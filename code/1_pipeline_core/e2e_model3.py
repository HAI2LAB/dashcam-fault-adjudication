#!/usr/bin/env python
"""
End-to-End 과실 판정 파이프라인 (역전파 가능)
=================================================
당신 아이디어 실현: 텍스트 중간단계 제거 → 확률/특징으로 연결 → 두 태스크 동시 역전파

구조:
  영상(32f) → VLM(LoRA) → hidden h (마지막 토큰)
              ├→ 분류헤드 → logits(3) : 과실비교 [B>A, A=B, A>B]
              └→ 회귀헤드 → A%        : 과실비율 (B=100-A)
  loss = CE(비교) + λ·SmoothL1(비율/100)
  → 하나의 loss로 VLM+두헤드 동시 학습 (end-to-end)

모델: internvl / qwen / qwen3  (--model)
프레임: 32 기본

핵심 설계:
  - 텍스트 생성 안 함 → 완전 미분가능
  - 단일 VLM 공유 + 2헤드 → 진짜 end-to-end
  - VLM은 LoRA, 헤드는 full 학습
"""
import os, sys, json, argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent1"))

# 💡 agent1_pipeline을 먼저 import → all_tied_weights_keys monkeypatch 적용 (import 시 실행됨)
import agent1_pipeline  # noqa: F401  (side-effect: PreTrainedModel 패치)

# 💡 추가 안전장치: get_total_byte_count가 로딩 초기에 all_tied_weights_keys를 참조하는데,
#    _finalize 패치는 그보다 늦게 실행됨. 그래서 클래스에 property를 직접 달아
#    all_tied_weights_keys가 없으면 _tied_weights_keys로 폴백하게 한다.
from transformers.modeling_utils import PreTrainedModel as _PTM
if not hasattr(_PTM, "_patched_tied_property"):
    def _all_tied_getter(self):
        v = self.__dict__.get("_all_tied_weights_keys_cache")
        if v is not None: return v
        tk = getattr(self, "_tied_weights_keys", None)
        if isinstance(tk, dict): return tk
        if isinstance(tk, (list, tuple)): return {k: None for k in tk}
        return {}
    def _all_tied_setter(self, value):
        self.__dict__["_all_tied_weights_keys_cache"] = value
    try:
        _PTM.all_tied_weights_keys = property(_all_tied_getter, _all_tied_setter)
        _PTM._patched_tied_property = True
    except Exception as _e:
        print(f"⚠️ tied property 패치 실패: {_e}")

# ─────────────────────────────────────────────────────────────
# 데이터/라벨 로직 (agent들에서 재활용)
# ─────────────────────────────────────────────────────────────
DATA_DIR = "/nas/Post-doc/허지웅/TS_Data"
MAP_FILE = os.path.join(DATA_DIR, "1.Training", "accident_mapping_en.csv")

import csv
def load_csv_maps():
    rows = list(csv.DictReader(open(MAP_FILE, encoding="utf-8-sig")))
    m = {}
    for r in rows:
        if not r["Accident_ID"].strip().isdigit(): continue
        m[int(r["Accident_ID"])] = {
            "Fault_Ratio_A": r.get("Fault_Ratio_A", ""),
            "Fault_Ratio_B": r.get("Fault_Ratio_B", ""),
            "A_Direction": r.get("A_Direction", ""),
        }
    return m
CSV_MAP = load_csv_maps()

# ── 상황(A움직임) 대분류 13개 — agent1_multi_field의 coarse_direction과 동일 ──
def coarse_direction(s):
    s = (s or "").lower()
    if any(k in s for k in ['walk','pedestrian','playing','lying','sidewalk']): return "Pedestrian/Crossing"
    if 'crossing' in s and 'car' not in s: return "Pedestrian/Crossing"
    if 'rear-end' in s or 'collided' in s or 'collision' in s: return "Rear-end/Collision"
    if 'u-turn' in s or 'u turn' in s or 'daewoo' in s: return "U-turn"
    if 'reverse' in s or 'backward' in s: return "Reverse"
    if 'overtaking' in s or 'passing' in s: return "Overtaking"
    if any(k in s for k in ['change course','lane change','career change','change route','merge','shoulder road']): return "Lane change/Merge"
    if 'center line' in s or 'one-way' in s or 'opposite' in s: return "Centerline violation"
    if 'stop and depart' in s or 'parked' in s or 'park' in s: return "Stop/Park"
    if 'left' in s and 'turn' in s: return "Left turn"
    if 'right' in s and 'turn' in s: return "Right turn"
    if 'straight' in s or 'trailing' in s or 'leading' in s or 'preceding' in s or 'falling behind' in s: return "Go straight"
    if 'left' in s: return "Left turn"
    if 'right' in s: return "Right turn"
    if 'roundabout' in s: return "Roundabout"
    if 'crossing' in s: return "Pedestrian/Crossing"
    return "Other"

SITUATION_CLASSES = ["Go straight","Left turn","Right turn","U-turn","Reverse",
    "Lane change/Merge","Overtaking","Rear-end/Collision","Roundabout",
    "Centerline violation","Stop/Park","Pedestrian/Crossing","Other"]
SIT2IDX = {c: i for i, c in enumerate(SITUATION_CLASSES)}

def get_situation_gt(jp):
    """JSON → accident_type → A_Direction → 대분류 인덱스(0~12)"""
    try:
        j = json.load(open(jp, encoding="utf-8"))
        at = _get_field(j, ["traffic_accident_type", "accident_type"])
        if at is None: return None
        adir = CSV_MAP.get(int(at), {}).get("A_Direction", "")
        return SIT2IDX.get(coarse_direction(adir), SIT2IDX["Other"])
    except Exception:
        return None

def load_split(txt_path):
    return [tuple(line.strip().split("\t")) for line in open(txt_path, encoding="utf-8") if line.strip()]

def _get_field(j, keys):
    """agent1_pipeline과 동일: j 및 j['video']에서 첫 매치, 없으면 None"""
    containers = []
    if isinstance(j, dict):
        containers.append(j)
        v = j.get("video")
        if isinstance(v, dict): containers.append(v)
    for cont in containers:
        for k in keys:
            if k in cont and cont[k] is not None:
                return cont[k]
    return None

def get_fault_gt(jp):
    """JSON → accident_type → (rate_a, rate_b). 비율 GT."""
    try:
        j = json.load(open(jp, encoding="utf-8"))
        at = _get_field(j, ["traffic_accident_type", "accident_type"])
        if at is None: return None
        at = int(at)
        m = CSV_MAP.get(at, {})
        ra, rb = int(float(m["Fault_Ratio_A"])), int(float(m["Fault_Ratio_B"]))
        return ra, rb
    except Exception:
        return None

def compare_label(ra, rb):
    """3-class: 0=B>A(블박차B 과실↑), 1=A=B, 2=A>B(상대A 과실↑)
       ra=상대(A) 과실, rb=블박(B) 과실"""
    if abs(ra - rb) <= 1: return 1        # 동등
    return 2 if ra > rb else 0            # A>B : B>A

COMPARE_NAMES = ["B>A (블랙박스차 과실↑)", "A=B (동등)", "A>B (상대차 과실↑)"]

# 프롬프트 (텍스트 생성 아님 — VLM이 영상+질문을 인코딩하게만)
PROMPT = (
    "You are analyzing a dashcam video of a car accident.\n"
    "Vehicle B is the dashcam (ego) car. Vehicle A is the other party.\n"
    "Assess the fault of each vehicle based on the video."
)

# ─────────────────────────────────────────────────────────────
# 모델별 백본 로딩 + hidden 추출 (핵심: 3개 모델 통일 인터페이스)
# ─────────────────────────────────────────────────────────────
class VLMBackbone(nn.Module):
    """3개 VLM을 통일 인터페이스로 감싸 hidden vector를 뽑는다."""
    def __init__(self, model_type, local_rank):
        super().__init__()
        self.model_type = model_type
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self._load()

    def _load(self):
        mt = self.model_type
        if mt == "internvl":
            from transformers import AutoModel, AutoTokenizer
            # 💡 InternVL은 device_map을 주면 tied_weights 에러 → device_map 없이 로딩 후 이동
            self.model = AutoModel.from_pretrained(
                "OpenGVLab/InternVL3-8B", torch_dtype=torch.bfloat16,
                trust_remote_code=True, low_cpu_mem_usage=True).eval()
            self.model = self.model.to(self.device)
            self.tok = AutoTokenizer.from_pretrained(
                "OpenGVLab/InternVL3-8B", trust_remote_code=True, use_fast=False)
            # 💡 [Fix 13] img_context_token_id 등록 (이미지 토큰 인식 필수)
            try:
                img_id = self.tok.convert_tokens_to_ids("<IMG_CONTEXT>")
                self.model.img_context_token_id = img_id
            except Exception as e:
                print(f"⚠️ img_context_token_id 등록 실패: {e}")
            self.hidden_size = self.model.language_model.config.hidden_size
        elif mt in ("qwen", "qwen3"):
            model_id = "Qwen/Qwen2.5-VL-7B-Instruct" if mt == "qwen" else "Qwen/Qwen3-VL-8B-Instruct"
            if mt == "qwen":
                from transformers import Qwen2_5_VLForConditionalGeneration as Loader
            else:
                try:
                    from transformers import Qwen3VLForConditionalGeneration as Loader
                except ImportError:
                    from transformers import AutoModelForImageTextToText as Loader
            from transformers import AutoProcessor
            self.model = Loader.from_pretrained(
                model_id, dtype=torch.bfloat16,
                device_map=self.device, attn_implementation="eager").eval()
            self.proc = AutoProcessor.from_pretrained(model_id)
            self.hidden_size = self.model.config.get_text_config().hidden_size \
                if hasattr(self.model.config, "get_text_config") else self.model.config.hidden_size
        else:
            raise ValueError(mt)

    def encode(self, frames, prompt):
        """영상 프레임 + 프롬프트 → 마지막 토큰 hidden (grad 흐름 유지)"""
        if self.model_type == "internvl":
            return self._encode_internvl(frames, prompt)
        return self._encode_qwen(frames, prompt)

    def _encode_internvl(self, frames, prompt):
        from agent1_pipeline import internvl_image_transform
        # 각 프레임 1타일 (448px) → pixel_values (n, 3, 448, 448)
        pv = torch.cat([internvl_image_transform(f).unsqueeze(0) for f in frames], dim=0)
        pv = pv.to(self.device, dtype=torch.bfloat16)
        n = pv.shape[0]
        # 💡 agent 방식: 프레임마다 개별 이미지 블록 (Fix 17)
        num_img_tok = getattr(self.model, "num_image_token", 256)
        IMG_START, IMG_END, IMG_CTX = "<img>", "</img>", "<IMG_CONTEXT>"
        image_block = IMG_START + IMG_CTX * num_img_tok + IMG_END
        frame_blocks = "".join([f"Frame-{i+1}: {image_block}\n" for i in range(n)])
        full = frame_blocks + prompt
        ids = self.tok(full, return_tensors="pt").input_ids.to(self.device)
        img_flags = torch.ones((n, 1), dtype=torch.long, device=self.device)
        out = self.model(pixel_values=pv, input_ids=ids, image_flags=img_flags,
                         output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1][:, -1, :]   # 마지막 토큰 (B=1, hidden)
        return h

    def _encode_qwen(self, frames, prompt):
        content = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": prompt})
        msgs = [{"role": "user", "content": content}]
        inputs = self.proc.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(self.device)
        out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1][:, -1, :]
        return h


N_SIT = len(SITUATION_CLASSES)   # 13 (상황=A움직임 대분류)
N_RATIO = 11                     # 비율 class 수 (11=정밀 / 5=실무구간) — set_ratio_classes로 변경
RATIO_VALUES = [i * 10 for i in range(N_RATIO)]  # [0,10,...,100]

def set_ratio_classes(n):
    """비율 class 수 변경 (11 또는 5). 모델 생성 전에 호출!
       11: 0,10,20,...,100 (정밀)
       5: 실무 구간 대표값 [5, 25, 50, 75, 95]"""
    global N_RATIO, RATIO_VALUES
    N_RATIO = n
    if n == 11:
        RATIO_VALUES = [i * 10 for i in range(11)]
    elif n == 5:
        RATIO_VALUES = [5, 25, 50, 75, 95]   # 구간 대표값
    else:
        raise ValueError(f"지원 안 함: {n}-class (11 또는 5)")

def ratio_to_class_5(ra):
    """A과실비율 → 5구간 인덱스 (실무 구간: 0-10/20-30/40-60/70-80/90-100)"""
    if ra <= 10: return 0
    if ra <= 30: return 1
    if ra <= 60: return 2
    if ra <= 80: return 3
    return 4

import math as _math
def ratio_to_class(ra):
    """A과실비율(0~100) → 현재 모드의 class 인덱스.
       11-class: 표준 반올림 (5→10 등)
       5-class: 실무 구간"""
    if N_RATIO == 5:
        return ratio_to_class_5(ra)
    idx = int(_math.floor(ra / 10.0 + 0.5))
    return max(0, min(N_RATIO - 1, idx))   # 0~10 범위 보장

class TripleHeadFaultModel(nn.Module):
    """VLM 백본 + 3헤드 순차 (상황→비교→비율). End-to-End 역전파.
       비율헤드는 11-class 분류(0,10,...,100) — 비율이 이산값이라 회귀보다 정확.
       각 헤드는 앞 헤드의 softmax 확률(숫자)을 hidden과 결합해 입력받음."""
    def __init__(self, backbone: VLMBackbone):
        super().__init__()
        self.backbone = backbone
        hs = backbone.hidden_size
        dev = backbone.device
        # 상황헤드: hidden → 13개 (A움직임 대분류)
        self.sit_head = nn.Sequential(
            nn.Linear(hs, hs // 2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hs // 2, N_SIT)).to(dev).to(torch.bfloat16)
        # 비교헤드: hidden + 상황확률(13) → 3개
        self.cmp_head = nn.Sequential(
            nn.Linear(hs + N_SIT, hs // 2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hs // 2, 3)).to(dev).to(torch.bfloat16)
        # 비율헤드: hidden + 상황확률(13) + 비교확률(3) → 11개 (분류)
        self.reg_head = nn.Sequential(
            nn.Linear(hs + N_SIT + 3, hs // 2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hs // 2, N_RATIO)).to(dev).to(torch.bfloat16)

    def forward(self, frames, prompt):
        import os as _os
        hard = _os.environ.get('E2E_CHAIN', 'soft') == 'hard'   # A안 ablation: hard(one-hot, detach) 전달
        h = self.backbone.encode(frames, prompt)          # (1, hs)
        # 1) 상황
        sit_logits = self.sit_head(h)                     # (1, 13)
        sit_soft = torch.softmax(sit_logits.float(), -1).to(h.dtype)  # (1,13) 미분가능
        if hard:
            sit_soft = torch.nn.functional.one_hot(
                sit_logits.argmax(-1), sit_logits.shape[-1]).to(h.dtype).detach()
        # 2) 비교 (hidden + 상황확률)
        cmp_in = torch.cat([h, sit_soft], dim=-1)         # (1, hs+13)
        cmp_logits = self.cmp_head(cmp_in)                # (1, 3)
        cmp_soft = torch.softmax(cmp_logits.float(), -1).to(h.dtype)  # (1,3)
        if hard:
            cmp_soft = torch.nn.functional.one_hot(
                cmp_logits.argmax(-1), cmp_logits.shape[-1]).to(h.dtype).detach()
        # 3) 비율 (hidden + 상황확률 + 비교확률) → 11-class
        reg_in = torch.cat([h, sit_soft, cmp_soft], dim=-1)  # (1, hs+13+3)
        ratio_logits = self.reg_head(reg_in)              # (1, 11)
        return sit_logits, cmp_logits, ratio_logits


def apply_lora(backbone: VLMBackbone, is_main, use_checkpoint=True):
    """VLM language_model에 LoRA."""
    from peft import LoraConfig, get_peft_model
    from agent1_pipeline import patch_internvl_for_training
    model = backbone.model
    if backbone.model_type == "internvl":
        patch_internvl_for_training(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    # gradient checkpointing (16프레임이면 꺼서 속도↑, OOM 시 켜기)
    if use_checkpoint:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if is_main: print("✅ gradient checkpointing 활성화")
        except Exception:
            try:
                model.gradient_checkpointing_enable()
                if is_main: print("✅ gradient checkpointing 활성화 (기본)")
            except Exception as e2:
                if is_main: print(f"⚠️ gradient checkpointing 실패: {e2}")
    else:
        if is_main: print("⏩ gradient checkpointing 비활성화 (속도 우선)")
    target = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and ("language_model" in name or "model.layers" in name):
            leaf = name.split(".")[-1]
            if leaf not in ("lm_head", "output", "embed_tokens"):
                target.add(leaf)
    target = sorted(target) or ["q_proj", "k_proj", "v_proj", "o_proj"]
    if is_main: print(f"🎯 LoRA 타겟: {target}")
    cfg = LoraConfig(r=16, lora_alpha=32, target_modules=target,
                     lora_dropout=0.1, task_type="CAUSAL_LM")
    backbone.model = get_peft_model(model, cfg)
    if is_main: backbone.model.print_trainable_parameters()
    return backbone
