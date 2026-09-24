"""
교통사고 분석 Multi-Agent 파이프라인 - [Agent 1: 상황 묘사 전문가]
🔥 진단 결과 반영 완전판:
   [Fix 1] JSON 키 두 가지(traffic_accident_type / accident_type) 모두 시도
   [Fix 2] 정규식 fallback 제거 (마지막 _숫자.mp4 는 비디오번호이지 사고유형이 아님)
   [Fix 3] InternVL는 최신 transformers 호환을 위해 InternVL3-8B로 교체
   [Fix 5] 오염 결과 자동 폭파 기준을 데이터 기반으로 강화
   [Fix 8] --phase {diag|zeroshot|all} 단계별 중단 옵션 추가
   [Fix 9] CSV 파일 자동 탐색 (accident_mapping_en.csv 우선)
   [Fix 10~11] Llama collator 패딩/stack
   [Fix 13~19] InternVL3 zero-shot/파인튜닝 호환 패치 일체
   ※ 패키지 자동 설치(auto_setup) 제거 — 컨테이너에 사전 설치 전제
      (Python 3.12 PEP 668 externally-managed 환경 대응)
"""
import os
import sys

# 💡 [핵심 방어 1] CUDA 메모리 단편화(Fragmentation) 원천 방지
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gc
import json
import csv
import random
import argparse
import difflib
import re
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from huggingface_hub import login, snapshot_download
from transformers.trainer_utils import get_last_checkpoint

import cv2
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor, AutoTokenizer, TrainingArguments, Trainer,
    Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM, AutoModel
)
from transformers.modeling_utils import PreTrainedModel

# flash-attn은 있으면 쓰고 없으면 sdpa로 폴백 (설치 시도 안 함)
try:
    import flash_attn  # noqa: F401
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False

# =========================================================================
# 🚨 [글로벌 몽키패치 1] DDP Tied Weights 로딩 에러 원천 차단
# =========================================================================
if not hasattr(PreTrainedModel, "_patched_for_tied_weights"):
    _orig_dict_item = PreTrainedModel.__dict__.get("_finalize_model_loading")
    if _orig_dict_item is not None:
        @classmethod
        def _patched_finalize(cls, *args, **kwargs):
            model = kwargs.get("model") if "model" in kwargs else (args[0] if args else None)
            if model is not None and not hasattr(model, 'all_tied_weights_keys'):
                tied_keys = getattr(model, '_tied_weights_keys', [])
                if isinstance(tied_keys, list): model.all_tied_weights_keys = {k: None for k in tied_keys}
                elif isinstance(tied_keys, dict): model.all_tied_weights_keys = tied_keys
                else: model.all_tied_weights_keys = {}
            if hasattr(_orig_dict_item, "__get__"): return _orig_dict_item.__get__(None, cls)(*args, **kwargs)
            else: return _orig_dict_item(*args, **kwargs)
        PreTrainedModel._finalize_model_loading = _patched_finalize
        PreTrainedModel._patched_for_tied_weights = True

# =========================================================================
# 🚨 [Fix 3] InternVL generate 복원 헬퍼 (InternVL3는 이미 정상이라 no-op로 통과)
#   - InternVL2를 쓰던 시절의 안전망. InternVL3에서는 자동으로 skip됨.
#   - 만약 향후 다른 trust_remote_code 모델로 회귀해도 동작하도록 보존.
# =========================================================================
def patch_internvl_generation(model, tokenizer=None):
    # 💡 [Fix 13] InternVL chat()의 필수 전제: img_context_token_id 등록
    #   이게 없으면 모델이 <image>를 visual token으로 치환 못 해서
    #   라벨만 따라 적고 값을 비워버리는 '빈 껍데기' 출력이 나온다.
    if tokenizer is not None:
        try:
            img_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
            if img_id is not None and img_id >= 0:
                model.img_context_token_id = img_id
        except Exception as e:
            print(f"⚠️ [InternVL patch] img_context_token_id 등록 실패: {e}")

    if not hasattr(model, "language_model"): return
    lm = model.language_model
    # 💡 [Fix 14] PEFT가 최상위 모델에서 prepare_inputs_for_generation을 찾는데
    #   InternVL은 이게 language_model에만 있음 → 최상위에 위임 메서드를 달아준다.
    #   (없으면 get_peft_model에서 AttributeError로 죽음)
    if not hasattr(model, "prepare_inputs_for_generation"):
        if hasattr(lm, "prepare_inputs_for_generation"):
            model.prepare_inputs_for_generation = lm.prepare_inputs_for_generation
    if not hasattr(model, "_supports_cache_class"):
        model._supports_cache_class = getattr(lm, "_supports_cache_class", False)

    if hasattr(lm, "generate") and callable(getattr(lm, "generate", None)):
        return  # 이미 정상
    try:
        from transformers.generation import GenerationMixin
        cls = lm.__class__
        if not issubclass(cls, GenerationMixin):
            new_cls = type(cls.__name__, (cls, GenerationMixin), {})
            lm.__class__ = new_cls
        if not hasattr(lm, "main_input_name"):
            lm.main_input_name = "input_ids"
        if not hasattr(lm, "_supports_cache_class"):
            lm._supports_cache_class = False
        # generate 주입 후 prepare_inputs_for_generation도 다시 위임
        if not hasattr(model, "prepare_inputs_for_generation") and hasattr(lm, "prepare_inputs_for_generation"):
            model.prepare_inputs_for_generation = lm.prepare_inputs_for_generation
    except Exception as e:
        print(f"⚠️ [InternVL patch] generate 주입 실패: {e}")

# =========================================================================
# 🚨 [Fix 15] InternVL 학습용 forward 래퍼
#   문제: InternVLChatModel.forward()는 표준 HF 인자(inputs_embeds 등)를 안 받고,
#         대신 image_flags 라는 자기만의 필수 인자를 요구한다.
#         Trainer가 model(**inputs)로 표준 호출하면 TypeError로 죽는다.
#   해결: forward를 래핑해서 (1) InternVL이 모르는 인자 제거,
#         (2) image_flags 자동 생성, (3) InternVL forward 시그니처에 맞게 전달.
# =========================================================================
def patch_internvl_for_training(model):
    import inspect
    # PEFT로 감싸기 "전"의 원본 InternVL 모델을 찾아야 함
    target = model
    # get_peft_model 적용 전이면 model 자체가 InternVLChatModel
    orig_forward = target.forward
    try:
        sig = inspect.signature(orig_forward)
        valid_keys = set(sig.parameters.keys())
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except (ValueError, TypeError):
        valid_keys, has_var_kw = set(), True

    def wrapped_forward(*args, **kwargs):
        # 1) InternVL이 모르는 인자 제거 (단, **kwargs를 받으면 그대로 둬도 됨)
        if not has_var_kw and valid_keys:
            kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        # 2) image_flags 보강 — InternVL forward의 필수 인자
        #    pixel_values의 첫 차원(=총 이미지 타일 수)만큼 1로 채움
        if "pixel_values" in kwargs and kwargs["pixel_values"] is not None:
            if "image_flags" in valid_keys and "image_flags" not in kwargs:
                pv = kwargs["pixel_values"]
                n_img = pv.shape[0]
                kwargs["image_flags"] = torch.ones((n_img, 1), dtype=torch.long, device=pv.device)
        return orig_forward(*args, **kwargs)

    target.forward = wrapped_forward
    print("✅ [Fix 15] InternVL forward 학습 래퍼 적용 완료")

def patch_mllama_is_causal(model):
    """💡 [Fix 24] transformers 5.3 버그 우회: Mllama의 vision attention 모듈에
    is_causal 속성이 없어서 sdpa 경로가 'has no attribute is_causal'로 실패한다.
    vision attention 모듈 전체에 is_causal=False를 주입하면 sdpa를 쓸 수 있고,
    sdpa는 eager보다 메모리를 훨씬 적게 써서 16프레임 OOM도 방지한다.
    (agent2/3에서 검증된 패치를 agent1에도 동일 적용)"""
    patched = 0
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if "Attention" in cls_name and not hasattr(module, "is_causal"):
            module.is_causal = False
            patched += 1
    print(f"✅ [Fix 24 / Llama sdpa 패치] is_causal 속성 주입: {patched}개 attention 모듈")
    return model


def patch_internvl_remote_code():
    """💡 [Fix 26] transformers 신버전의 meta-device 초기화와 InternVL 원격 코드의
    .item() 호출 충돌 해결 (Tensor.item() cannot be called on meta tensors).
    HF 캐시의 modeling_intern_vit.py에서 해당 줄을 순수 Python 산술로 교체.
    (수학적으로 동일: 0~drop_path_rate 균등분할. transformers 버전 무관 작동)
    모델 import 전에 호출해야 효과 있음."""
    import glob as _glob
    pats = _glob.glob(os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/**/modeling_intern_vit.py"),
        recursive=True)
    old = "dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]"
    new = "dpr = [config.drop_path_rate * i / max(config.num_hidden_layers - 1, 1) for i in range(config.num_hidden_layers)]  # [Fix 26] meta-safe"
    for p in pats:
        try:
            src = open(p).read()
            if old in src:
                open(p, "w").write(src.replace(old, new))
                print(f"✅ [Fix 26] InternVL meta-safe 패치: {p}")
        except Exception as e:
            print(f"⚠️ [Fix 26] 패치 실패 {p}: {e}")

AVAILABLE_MODELS = {
    "internvl": "OpenGVLab/InternVL3-8B",
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "llama": "meta-llama/Llama-3.2-11B-Vision-Instruct"
}

# =========================================================================
# 💡 JSON에서 키 후보 여러 개를 유연하게 찾는 헬퍼 (Fix 1의 기반)
# =========================================================================
def _get_field(j_data, candidates):
    """j_data 자체 및 j_data['video']에서 candidates 중 첫 매치 반환."""
    containers = []
    if isinstance(j_data, dict):
        containers.append(j_data)
        v = j_data.get("video")
        if isinstance(v, dict): containers.append(v)
    for cont in containers:
        for key in candidates:
            if key in cont and cont[key] is not None:
                return cont[key]
    return None

# =========================================================================
# 💡 강철의 CSV 파서 (Unknown 정답지 도배 완벽 방지)
# =========================================================================
def load_cot_map_and_prompt(mapping_file):
    cot_map = {}
    locations, loc_features, a_dirs, b_dirs, col_types = set(), set(), set(), set(), set()

    # 💡 [Fix 9] 어디서 깨지는지 즉시 보이도록 명시 (다음번 디버그 비용 절감)
    print(f"📂 [CSV Loader] path={mapping_file}")
    print(f"   파일 존재 여부: {os.path.exists(mapping_file) if mapping_file else False}")
    if mapping_file and os.path.exists(mapping_file):
        print(f"   파일 크기    : {os.path.getsize(mapping_file):,} bytes")

    if mapping_file and os.path.exists(mapping_file):
        with open(mapping_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = []
            for row in reader:
                if not row or len(row) < 8: continue
                # 헤더 인식
                if "Accident_ID" in row or "사고유형" in row:
                    header = [h.strip() for h in row]
                    continue
                if not header: continue

                row_dict = dict(zip(header, row))
                acc_id_str = row_dict.get("Accident_ID", row_dict.get("사고유형", ""))
                try: acc_id = int(acc_id_str)
                except ValueError: continue

                if "Collision_Type" in row_dict:
                    cot_map[acc_id] = {
                        "Location": row_dict.get("Location", ""),
                        "Location_Feature": row_dict.get("Location_Feature", ""),
                        "A_Direction": row_dict.get("A_Direction", ""),
                        "B_Direction": row_dict.get("B_Direction", ""),
                        "Collision_Type": row_dict.get("Collision_Type", ""),
                    }

    for row_dict in cot_map.values():
        locations.add(str(row_dict.get('Location', '')).strip())
        loc_features.add(str(row_dict.get('Location_Feature', '')).strip())
        a_dirs.add(str(row_dict.get('A_Direction', '')).strip())
        b_dirs.add(str(row_dict.get('B_Direction', '')).strip())
        col_types.add(str(row_dict.get('Collision_Type', '')).strip())

    for s in [locations, loc_features, a_dirs, b_dirs, col_types]:
        s.discard(""); s.discard("Unknown"); s.discard("None")

    sys_prompt = "You are a strict English-only assistant. Never generate Korean words. Answer strictly in English."

    prompt = (
        "🚨 [ABSOLUTE RULE]: Never mention the fault ratio, liability, or who is more at fault. Only classify the 'facts' shown in the video.\n\n"
        "🚨 [Definition of Vehicle A and B]\n"
        "- Vehicle B (Ego Vehicle): The 'dashcam vehicle' recording the video from a first-person perspective.\n"
        "- Vehicle A (Other Object): The 'opposing vehicle, pedestrian, or object' that collides with Vehicle B.\n\n"
        "From the [Classification Options] provided below, choose exactly one matching item that best describes the video situation for each category, and format your answer exactly according to the [Output Format]. Do not generate words outside of these options.\n\n"
        "=== [Classification Options] ===\n"
        f"- Accident Location: [{', '.join(sorted(locations)) if locations else 'straight road, Intersection, etc.'}]\n"
        f"- Location Feature: [{', '.join(sorted(loc_features)) if loc_features else 'equal width road, collision accident, etc.'}]\n"
        f"- Vehicle A Movement: [{', '.join(sorted(a_dirs)) if a_dirs else 'go straight ahead, left turn, etc.'}]\n"
        f"- Vehicle B Movement: [{', '.join(sorted(b_dirs)) if b_dirs else 'go straight ahead, parked, etc.'}]\n"
        f"- Collision Type: [{', '.join(sorted(col_types)) if col_types else 'vehicle to vehicle, chassis bike, etc.'}]\n"
        "================================\n\n"
        "[Output Format]\n"
        "1. Accident Location: \n2. Location Feature: \n3. Vehicle A Movement: \n4. Vehicle B Movement: \n5. Collision Type: "
    )
    return cot_map, sys_prompt, prompt

def build_stage1_answer_en(map_info):
    if not map_info:
        return ("1. Accident Location: Unknown\n2. Location Feature: Unknown\n3. Vehicle A Movement: Unknown\n4. Vehicle B Movement: Unknown\n5. Collision Type: Unknown")
    return (
        f"1. Accident Location: {map_info.get('Location', 'Unknown')}\n"
        f"2. Location Feature: {map_info.get('Location_Feature', 'Unknown')}\n"
        f"3. Vehicle A Movement: {map_info.get('A_Direction', 'Unknown')}\n"
        f"4. Vehicle B Movement: {map_info.get('B_Direction', 'Unknown')}\n"
        f"5. Collision Type: {map_info.get('Collision_Type', 'Unknown')}"
    )

def extract_frames(video_path, num_frames=16, max_size=448):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise ValueError("Cannot open video")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 2: cap.release(); raise ValueError("Video too short")
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img.thumbnail((max_size, max_size))
                frames.append(img)
            else: frames.append(Image.new('RGB', (max_size, max_size), color=(128, 128, 128)))
        cap.release()
        while len(frames) < num_frames: frames.append(frames[-1] if frames else Image.new('RGB', (max_size, max_size), color=(128, 128, 128)))
        return frames
    except Exception:
        return [Image.new('RGB', (max_size, max_size), color=(128, 128, 128))] * num_frames

def internvl_image_transform(img, max_size=448):
    transform = T.Compose([
        T.Lambda(lambda i: i.convert('RGB') if i.mode != 'RGB' else i),
        T.Resize((max_size, max_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transform(img)

def text_similarity_score(pred, gt):
    p = pred.replace(" ", "").replace("\n", "").strip().lower()
    g = gt.replace(" ", "").replace("\n", "").strip().lower()
    if not p or not g or "failed" in p or "error" in p: return 0.0
    match = sum(n for _, _, n in difflib.SequenceMatcher(None, p, g).get_matching_blocks())
    pr, rc = match / len(p), match / len(g)
    if pr + rc == 0: return 0.0
    return (2 * pr * rc) / (pr + rc)

def load_split(txt_path):
    if not os.path.exists(txt_path): raise FileNotFoundError(f"스플릿 파일이 없습니다: {txt_path}")
    return [tuple(line.strip().split("\t")) for line in open(txt_path, "r", encoding="utf-8") if line.strip()]

class Agent1Dataset(Dataset):
    # 💡 [Fix 7] JSON 파싱 실패율을 rank 0에서 한 번만 요약
    _diag_total = 0
    _diag_json_fail = 0
    _diag_no_id = 0
    _diag_no_map = 0
    _diag_printed = False

    def __init__(self, data_pairs, processor, tokenizer, cot_map, sys_prompt, prompt, model_type, num_frames=16):
        self.pairs, self.processor, self.tokenizer = data_pairs, processor, tokenizer
        self.m_type, self.nf, self.cot_map = model_type, num_frames, cot_map
        self.sys_prompt, self.prompt = sys_prompt, prompt
        # 💡 [Fix 18] _num_image_token을 생성자에서 인스턴스 속성으로 박아둠
        #   (나중에 외부에서 추가하면 num_workers>0 워커 복제 시 누락될 수 있음)
        self._num_image_token = 256

    def __len__(self): return len(self.pairs)

    @classmethod
    def _maybe_print_diag(cls, is_main):
        # 20개 이상 모이면 한 번만 요약 출력 (작은 sanity check도 잡히도록)
        if cls._diag_printed or not is_main: return
        if cls._diag_total < 20: return
        cls._diag_printed = True
        print("\n🩺 [Agent1Dataset Diagnostics]")
        print(f"   샘플 {cls._diag_total}개 점검 결과:")
        print(f"   - JSON 읽기 실패  : {cls._diag_json_fail} ({cls._diag_json_fail*100/cls._diag_total:.1f}%)")
        print(f"   - JSON에 ID 없음  : {cls._diag_no_id} ({cls._diag_no_id*100/cls._diag_total:.1f}%)")
        print(f"   - cot_map 미스    : {cls._diag_no_map} ({cls._diag_no_map*100/cls._diag_total:.1f}%)")
        ok = cls._diag_total - cls._diag_json_fail - cls._diag_no_id - cls._diag_no_map
        print(f"   - 정상 GT 매핑    : {ok} ({ok*100/cls._diag_total:.1f}%)")
        if ok < cls._diag_total * 0.5:
            print("   ⚠️ 정상 매핑이 50% 미만입니다! 데이터 경로/CSV/JSON 키를 다시 확인하세요.\n")
        else:
            print("   ✅ 정상 매핑이 충분합니다.\n")

    def get_info(self, idx):
        vp, jp = self.pairs[idx]
        acc_type = -1
        json_fail = False
        no_id = False

        # 💡 [Fix 1] JSON에서 ID 추출 — 두 가지 키 이름 모두 시도
        try:
            with open(jp, "r", encoding="utf-8") as f:
                j_data = json.load(f)
            val = _get_field(j_data, ["traffic_accident_type", "accident_type"])
            if val is None:
                no_id = True
            else:
                try:
                    acc_type = int(val)
                except (ValueError, TypeError):
                    no_id = True
        except Exception:
            json_fail = True

        # 💡 [Fix 2] 정규식 fallback 제거
        #   파일명 패턴은 {촬영방법}_{시점}_{날짜}_{사고대상}_{작업자ID}_{비디오번호}.mp4
        #   마지막 숫자는 '비디오번호'이지 '사고유형'이 아니므로 추출하면 안 됨.
        #   ID가 없으면 차라리 Unknown으로 두는 게 잘못된 매칭보다 안전하다.

        map_info = self.cot_map.get(acc_type, {})
        no_map = (acc_type != -1 and not map_info)

        # 진단 카운터 (스레드 안전성은 신경 쓰지 않음 — 근사값이면 충분)
        Agent1Dataset._diag_total += 1
        if json_fail: Agent1Dataset._diag_json_fail += 1
        elif no_id:   Agent1Dataset._diag_no_id += 1
        elif no_map:  Agent1Dataset._diag_no_map += 1

        ans = build_stage1_answer_en(map_info)
        return vp, ans

    def __getitem__(self, idx):
        v_path, answer = self.get_info(idx)
        frames = extract_frames(v_path, num_frames=self.nf)

        try:
            if self.m_type == "qwen":
                from qwen_vl_utils import process_vision_info
                # 💡 Qwen 비디오 코덱 우회 — 비디오 대신 사진 16장 묶음
                vision_content = [{"type": "image", "image": img} for img in frames]
                msgs = [
                    {"role": "system", "content": [{"type": "text", "text": self.sys_prompt}]},
                    {"role": "user", "content": vision_content + [{"type": "text", "text": self.prompt}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]}
                ]
                text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                img_in, vid_in = process_vision_info(msgs)
                inputs = self.processor(text=[text], images=img_in, videos=vid_in, padding=False, return_tensors="pt")

                p_text = self.processor.apply_chat_template([msgs[0], msgs[1]], tokenize=False, add_generation_prompt=True)
                prompt_inputs = self.processor(text=[p_text], images=img_in, videos=vid_in, padding=False, return_tensors="pt")

            elif self.m_type == "llama":
                cont = [{"type": "image"} for _ in range(self.nf)] + [{"type": "text", "text": self.prompt}]
                msgs = [
                    {"role": "system", "content": [{"type": "text", "text": self.sys_prompt}]},
                    {"role": "user", "content": cont},
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]}
                ]
                text = self.processor.apply_chat_template(msgs, tokenize=False)
                inputs = self.processor(images=frames, text=text, return_tensors="pt")
                p_text = self.processor.apply_chat_template([msgs[0], msgs[1]], tokenize=False, add_generation_prompt=True)
                prompt_inputs = self.processor(images=frames, text=p_text, return_tensors="pt")

            elif self.m_type == "internvl":
                # 💡 [Fix 17] InternVL 학습 입력: <image> 플레이스홀더를
                #   IMG_START + IMG_CONTEXT*(타일당 256토큰) + IMG_END 로 확장해야
                #   forward에서 visual embedding과 토큰 위치가 정확히 매칭된다.
                #   (이 확장을 안 하면 image token 수 불일치로 forward가 깨짐)
                IMG_START, IMG_END, IMG_CTX = '<img>', '</img>', '<IMG_CONTEXT>'
                num_image_token = getattr(self, '_num_image_token', 256)  # InternVL3 기본 256
                image_block = IMG_START + IMG_CTX * num_image_token + IMG_END

                # 프레임마다 명시적 마커 + 확장된 이미지 블록
                frame_blocks = "".join([f"Frame-{i+1}: {image_block}\n" for i in range(self.nf)])
                pmpt = frame_blocks + self.prompt

                text = f"<|im_start|>system\n{self.sys_prompt}<|im_end|>\n<|im_start|>user\n{pmpt}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n"
                p_text = f"<|im_start|>system\n{self.sys_prompt}<|im_end|>\n<|im_start|>user\n{pmpt}<|im_end|>\n<|im_start|>assistant\n"

                pixel_values = torch.cat([internvl_image_transform(f).unsqueeze(0) for f in frames], dim=0)
                input_ids = self.tokenizer(text, return_tensors="pt").input_ids
                p_input_ids = self.tokenizer(p_text, return_tensors="pt").input_ids

                inputs = {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
                prompt_inputs = {"input_ids": p_input_ids}

        except Exception as e:
            # 💡 [Fix 18] 폴백도 반드시 input_ids를 보장 (tokenizer 우선)
            text = f"System: {self.sys_prompt}\nUser: {self.prompt}\nAssistant: {answer}"
            tk = self.tokenizer
            if tk is None and self.processor is not None:
                tk = getattr(self.processor, "tokenizer", None)
            if tk is not None:
                inputs = tk(text, return_tensors="pt")
            else:
                inputs = self.processor(text=[text], return_tensors="pt")
            prompt_inputs = inputs
            # 💡 [Fix 23] InternVL은 forward에 pixel_values가 필수.
            #   폴백으로 빠지면 pixel_values가 없어서 학습 중 forward가
            #   'missing pixel_values'로 죽는다 → 더미 프레임으로라도 pixel_values 보장.
            if self.m_type == "internvl":
                try:
                    dummy = [Image.new('RGB', (448, 448), (128, 128, 128))] * self.nf
                    inputs["pixel_values"] = torch.cat(
                        [internvl_image_transform(f).unsqueeze(0) for f in dummy], dim=0
                    )
                except Exception:
                    pass

        inputs = {k: v.squeeze(0) for k, v in inputs.items() if v is not None and hasattr(v, "shape") and len(v.shape) > 0}
        # 💡 [Fix 18] input_ids가 없으면 이 샘플은 collator에서 KeyError를 일으키므로 방어
        if "input_ids" not in inputs:
            # 최후의 수단: answer만이라도 토큰화
            tk = self.tokenizer or getattr(self.processor, "tokenizer", None)
            ids = tk(f"{self.prompt}\n{answer}", return_tensors="pt").input_ids.squeeze(0)
            inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            prompt_inputs = {"input_ids": ids}
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        prompt_len = prompt_inputs["input_ids"].shape[-1]
        inputs["labels"] = inputs["input_ids"].clone()
        inputs["labels"][:prompt_len] = -100

        if "pixel_values_videos" in inputs: inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(torch.bfloat16)
        if "pixel_values" in inputs: inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return inputs

def dynamic_collate_fn(batch, pad_token_id, model_type):
    """모델별 입력 텐서를 안전하게 배치로 묶는다.
    
    🚨 [Fix 10] Llama-3.2-Vision의 cross_attention_mask는 시퀀스 길이가 가변이므로
       input_ids 패딩 길이에 맞춰 함께 패딩해야 한다. 안 그러면 RuntimeError.
    🚨 [Fix 11] Llama의 pixel_values는 [num_images, max_tiles, C, H, W] (5D)이므로
       cat이 아니라 stack으로 새 batch 차원을 만들어야 한다.
    """
    pad_id = pad_token_id or 0
    out = {
        "input_ids": torch.nn.utils.rnn.pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id),
        "attention_mask": torch.nn.utils.rnn.pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0),
        "labels": torch.nn.utils.rnn.pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100),
    }
    target_seq_len = out["input_ids"].shape[1]  # 패딩 후 시퀀스 길이

    # ---- pixel_values: 모델별로 다르게 묶기 ----
    # 💡 [Fix 23] batch[0]에만 의존하면, 첫 샘플이 폴백(pixel_values 없음)일 때
    #   배치 전체에서 pixel_values가 빠져 forward가 죽는다.
    #   → 하나라도 pixel_values를 가진 샘플이 있으면 처리.
    has_pv = any("pixel_values" in b for b in batch)
    if has_pv:
        pv_list = [b["pixel_values"] for b in batch if "pixel_values" in b]
        if model_type == "internvl":
            # InternVL: 샘플당 [num_frames, C, H, W] (4D) — frames 축으로 cat
            out["pixel_values"] = torch.cat(
                [p if p.dim() == 4 else p.unsqueeze(0) for p in pv_list], dim=0
            )
        elif model_type == "llama":
            # Llama: 샘플당 [num_images, max_tiles, C, H, W] (5D) — stack으로 batch 차원 추가
            try:
                out["pixel_values"] = torch.stack(pv_list, dim=0)
            except RuntimeError:
                # 샘플간 shape 불일치 시 fallback (보통은 발생 안 함)
                out["pixel_values"] = torch.cat([p.unsqueeze(0) for p in pv_list], dim=0)
        else:  # qwen — pixel_values가 flat [total_patches, dim] 형태 (image_grid_thw로 분기됨)
            out["pixel_values"] = torch.cat(
                [p.unsqueeze(0) if p.dim() == 4 else p for p in pv_list], dim=0
            )

    # ---- Qwen-only 키들 ----
    for k in ["pixel_values_videos", "image_grid_thw", "video_grid_thw"]:
        if k in batch[0] and batch[0][k] is not None:
            out[k] = torch.cat([b[k] for b in batch if k in b and b[k] is not None], dim=0)

    # ---- Llama-only: cross_attention_mask는 seq_len 패딩 필요 ----
    if "cross_attention_mask" in batch[0] and batch[0]["cross_attention_mask"] is not None:
        cam_list = []
        for b in batch:
            c = b["cross_attention_mask"]  # [seq_len, num_images, max_tiles]
            cur_seq = c.shape[0]
            if cur_seq < target_seq_len:
                pad_shape = (target_seq_len - cur_seq,) + tuple(c.shape[1:])
                pad_tensor = torch.zeros(pad_shape, dtype=c.dtype)
                c = torch.cat([c, pad_tensor], dim=0)
            elif cur_seq > target_seq_len:
                c = c[:target_seq_len]
            cam_list.append(c.unsqueeze(0))
        out["cross_attention_mask"] = torch.cat(cam_list, dim=0)

    # ---- Llama-only: aspect_ratio_*는 시퀀스와 무관 → 단순 stack ----
    for k in ["aspect_ratio_ids", "aspect_ratio_mask"]:
        if k in batch[0] and batch[0][k] is not None:
            try:
                out[k] = torch.stack([b[k] for b in batch if k in b], dim=0)
            except RuntimeError:
                out[k] = torch.cat([b[k].unsqueeze(0) for b in batch if k in b], dim=0)

    return out

@torch.no_grad()
def generate_descriptions(model, processor, tokenizer, dataset, local_rank, world_size, model_type, num_frames, result_dir):
    model.eval()
    device = torch.device(f"cuda:{local_rank}")
    my_indices = list(range(local_rank, len(dataset), world_size))
    prompt_text = dataset.prompt
    sys_prompt = dataset.sys_prompt

    temp_file = os.path.join(result_dir, f"temp_agent1_{model_type}_rank{local_rank}.json")
    my_preds = []

    if os.path.exists(temp_file):
        try:
            with open(temp_file, "r", encoding="utf-8") as f: my_preds = json.load(f)
            if len(my_preds) > 0 and local_rank == 0:
                print(f"🔄 [Auto-Resume] {model_type.upper()} 모델 이어하기를 진행합니다.")
        except Exception: my_preds = []

    processed_indices = {p["idx"] for p in my_preds}
    pbar = tqdm(my_indices, desc=f"[{model_type.upper()}] 평가/생성 중 GPU {local_rank}", position=local_rank)

    for idx in pbar:
        if idx in processed_indices: continue

        v_path, gt_answer = dataset.get_info(idx)
        frames = extract_frames(v_path, num_frames=num_frames)
        inputs = None; gen_ids = None; out_tokens = None
        pixel_values = None
        out_text = ""

        try:
            if model_type == "qwen":
                from qwen_vl_utils import process_vision_info
                vision_content = [{"type": "image", "image": img} for img in frames]
                msgs = [
                    {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                    {"role": "user", "content": vision_content + [{"type": "text", "text": prompt_text}]}
                ]
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                img_in, vid_in = process_vision_info(msgs)
                inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to(device)

            elif model_type == "llama":
                cont = [{"type": "image"} for _ in range(num_frames)] + [{"type": "text", "text": prompt_text}]
                msgs = [{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}, {"role": "user", "content": cont}]
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = processor(images=frames, text=[text], padding=True, return_tensors="pt").to(device)

            elif model_type == "internvl":
                # 💡 [Fix 13] InternVL3 멀티이미지 공식 방식
                #   원인 진단: 이전엔 "<image>\n"*16을 sys_prompt에 그냥 박아서
                #   chat() 내부 토큰 치환과 충돌 → '11.', '33.' 깨진 출력 + 빈 값.
                #   해결: 프레임마다 'Frame-N: <image>' 명시 마커 + num_patches_list 정확 매칭.
                #   (model.img_context_token_id는 모델 로드 직후 1회 등록되어 있어야 함)
                pixel_values = torch.cat(
                    [internvl_image_transform(f).unsqueeze(0) for f in frames], dim=0
                ).to(device, dtype=torch.bfloat16)
                num_patches_list = [1] * num_frames  # 프레임당 1타일 (448 리사이즈)

                # <image> 수동 삽입 금지 — 프레임 마커로만 표시, chat이 알아서 치환
                frame_markers = "".join([f"Frame-{i+1}: <image>\n" for i in range(num_frames)])
                pmpt = f"{sys_prompt}\n\n{frame_markers}\n{prompt_text}"

                generation_config = dict(max_new_tokens=128, do_sample=False)
                try:
                    chat_ret = model.chat(
                        tokenizer, pixel_values, pmpt, generation_config,
                        num_patches_list=num_patches_list, history=None, return_history=False
                    )
                except TypeError:
                    chat_ret = model.chat(tokenizer, pixel_values, pmpt, generation_config)
                out_text = chat_ret[0] if isinstance(chat_ret, tuple) else chat_ret
                gen_ids = None

            if inputs is not None and "pixel_values_videos" in inputs:
                inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(torch.bfloat16)
            if inputs is not None and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            if model_type == "internvl":
                out_text = (out_text or "").strip()
                out_text = re.sub(r'(?i)\[?(final\s*)?conclusion\]?.*', '', out_text, flags=re.DOTALL).strip()
            elif inputs is not None:
                try:
                    gen_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
                    input_len = inputs["input_ids"].shape[1]
                    out_tokens = gen_ids[:, input_len:] if gen_ids.shape[1] > input_len else gen_ids
                    out_text = processor.batch_decode(out_tokens, skip_special_tokens=True)[0]
                    out_text = re.sub(r'(?i)\[?(final\s*)?conclusion\]?.*', '', out_text, flags=re.DOTALL)
                    out_text = re.sub(r'(?i)(therefore|in conclusion|conclusion:|as a result).*', '', out_text, flags=re.DOTALL).strip()
                except RuntimeError as e:
                    if "memory" in str(e).lower() or "c10" in str(e).lower(): out_text = "Generation Failed: CUDA OOM / C++ Error"
                    else: out_text = f"Generation Failed: {str(e)}"
            else:
                if not out_text: out_text = "Generation Failed: Inputs empty"

        except Exception as e:
            out_text = f"Generation Failed: {str(e)}"

        my_preds.append({"idx": idx, "video": os.path.basename(v_path), "pred": out_text.strip(), "gt": gt_answer})

        if len(my_preds) % 5 == 0:
            with open(temp_file, "w", encoding="utf-8") as f: json.dump(my_preds, f, ensure_ascii=False)

        try:
            if inputs is not None: del inputs
            if gen_ids is not None: del gen_ids
            if out_tokens is not None: del out_tokens
            if pixel_values is not None: del pixel_values
            del frames
        except: pass

        gc.collect()
        torch.cuda.empty_cache()

    with open(temp_file, "w", encoding="utf-8") as f: json.dump(my_preds, f, ensure_ascii=False)
    if dist.is_initialized(): dist.barrier()

    if local_rank == 0:
        all_preds = []
        for r in range(world_size):
            r_file = os.path.join(result_dir, f"temp_agent1_{model_type}_rank{r}.json")
            if os.path.exists(r_file):
                all_preds.extend(json.load(open(r_file, "r", encoding="utf-8")))
                os.remove(r_file)
        all_preds.sort(key=lambda x: x["idx"])
        # 진단 출력
        Agent1Dataset._maybe_print_diag(is_main=True)
        return all_preds
    return None

# =========================================================================
# 💡 [Fix 5] 오염된 과거 결과를 데이터 기반으로 검출 (키워드 매칭 X)
# =========================================================================
def is_polluted_result(path):
    """이전 실행에서 망가진 결과인지 데이터 기반으로 판단"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        details = d.get("details", [])
        if not details: return True, "details 비어있음"
        preds = [x.get("model_prediction", "") for x in details]
        gts   = [x.get("ground_truth", "")    for x in details]
        # 1) Pred가 전부 같은 에러 메시지
        if len(set(preds)) == 1 and any(k in preds[0] for k in ("Generation Failed", "Error", "error")):
            return True, "모든 예측이 동일 에러"
        # 2) Generation Failed 비율 50% 초과
        fail_cnt = sum(1 for p in preds if "Generation Failed" in p or "Generation failed" in p)
        if fail_cnt * 2 > len(preds):
            return True, f"Generation Failed가 {fail_cnt}/{len(preds)}"
        # 3) GT 종류가 1개뿐 (데이터 파이프 폭사)
        if len(set(gts)) <= 1:
            return True, "GT 다양성=1 (데이터 파이프 깨짐)"
        return False, "OK"
    except Exception as e:
        return True, f"로드 실패: {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default="", help="Hugging Face 토큰")
    parser.add_argument("--num_frames", type=int, default=16, help="모든 모델 공통 추출 프레임수")
    parser.add_argument("--eval_samples", type=int, default=-1, help="-1이면 Val 전체 사용")
    parser.add_argument("--test_samples", type=int, default=-1, help="-1이면 Test 전체 사용")
    parser.add_argument("--train_samples", type=int, default=-1, help="-1이면 Train 전체 사용 (파인튜닝)")
    parser.add_argument("--epochs", type=int, default=5, help="최대 파인튜닝 epoch 수 (기본 5)")
    parser.add_argument("--patience", type=int, default=1,
                        help="early stopping patience: val 점수가 N번 연속 하락하면 중단 (기본 1)")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="per-device batch size. 0이면 모델별 기본값 사용")
    parser.add_argument("--force_clean", action="store_true", help="모든 과거 결과 강제 삭제")
    parser.add_argument("--result_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "result"),
                        help="결과 저장 디렉토리 (기본: 스크립트 위치/result — 예: TS_code/agent1/result)")
    # 💡 [Fix 8] 단계별 중단 옵션 — sanity check 시 fine-tuning 건너뛰기 가능
    parser.add_argument("--phase", choices=["diag", "zeroshot", "all"], default="all",
                        help="diag=데이터 점검만 / zeroshot=Phase1만 / all=전체(기본)")
    # 💡 [Fix 25] venv 분리 실행 지원: 이 모델만 zero-shot 평가 (다른 모델은 건너뜀)
    #   3개 모델을 각각 따로(또는 각자 환경에서) 평가하고, 결과 JSON이 다 모이면
    #   --phase all 로 우승자 파인튜닝을 진행하는 워크플로용.
    parser.add_argument("--only_model", choices=list(AVAILABLE_MODELS.keys()), default=None,
                        help="지정 시 이 모델만 zero-shot 평가하고 종료 (파인튜닝 안 함)")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available(): torch.cuda.set_device(local_rank)
    if "LOCAL_RANK" in os.environ and not dist.is_initialized(): dist.init_process_group(backend="nccl")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = (local_rank == 0)

    # 💡 [Fix 21] NAS 폴더가 다른 uid(ubuntu) 소유라 쓰기 불가 → 결과 저장은
    #   쓰기 가능한 곳으로. --result_dir로 덮어쓸 수 있게 함. 기본은 홈 아래.
    RESULT_DIR = args.result_dir
    if is_main: os.makedirs(RESULT_DIR, exist_ok=True)
    if is_main: print(f"💾 [결과 저장 위치] {RESULT_DIR}")

    # 💡 [Fix 5] 데이터 기반 오염 검출
    if is_main:
        print("\n🧹 [System] 과거 결과 폴루션 점검...")
        for m in AVAILABLE_MODELS.keys():
            for fname in [f"zeroshot_results_agent1_{m}.json", f"finetuned_results_agent1_{m}.json"]:
                z_path = os.path.join(RESULT_DIR, fname)
                if not os.path.exists(z_path): continue
                if args.force_clean:
                    os.remove(z_path); print(f"🗑️ [Force-Clean] {fname} 강제 삭제")
                    continue
                bad, reason = is_polluted_result(z_path)
                if bad:
                    os.remove(z_path)
                    print(f"🗑️ [오토 클리너] {fname} 삭제 — 사유: {reason}")
                else:
                    print(f"✅ {fname}는 유효합니다 — 그대로 둡니다.")
    if dist.is_initialized(): dist.barrier()

    if is_main and args.hf_token: login(token=args.hf_token)
    if is_main:
        # 💡 [Fix 25] --only_model이면 해당 모델만 다운로드 (시간/디스크 절약)
        dl_targets = ([AVAILABLE_MODELS[args.only_model]] if args.only_model
                      else list(AVAILABLE_MODELS.values()))
        for m_id in dl_targets:
            snapshot_download(repo_id=m_id, token=args.hf_token if args.hf_token else None)
    if dist.is_initialized(): dist.barrier()
    # 💡 [Fix 26] InternVL 원격 코드 meta-safe 패치 (모델 import 전, rank0만)
    if is_main: patch_internvl_remote_code()
    if dist.is_initialized(): dist.barrier()

    DATA_DIR = "/nas/Post-doc/허지웅/TS_Data"
    # 💡 [Fix 9] CSV 파일 자동 탐색
    #   - 파서는 영문 헤더(Accident_ID, Collision_Type, ...) 기준이라 _en 버전이 정답
    #   - 두 후보를 모두 시도하여 어느 쪽이든 작동하도록 안전망
    _MAP_CANDIDATES = [
        "/nas/Post-doc/허지웅/TS_Data/1.Training/accident_mapping_en.csv",
        "/nas/Post-doc/허지웅/TS_Data/1.Training/accident_mapping.csv",
    ]
    MAP_FILE = next((p for p in _MAP_CANDIDATES if os.path.exists(p)), _MAP_CANDIDATES[0])
    if is_main: print(f"📂 [CSV] 사용할 매핑 파일: {MAP_FILE}")

    global_cot_map, global_sys_prompt, global_prompt = load_cot_map_and_prompt(MAP_FILE)
    if is_main:
        print(f"\n📁 [데이터 로드] cot_map: {len(global_cot_map)}개 사고유형 매핑됨")
        if len(global_cot_map) < 100:
            print(f"⚠️ cot_map 사이즈가 비정상적으로 작습니다 — CSV 경로/파싱을 확인하세요: {MAP_FILE}")

    if is_main: print("\n📁 [데이터 로드] 분할된 txt 파일을 읽어옵니다...")
    # 💡 [4-way 분할 개편]
    #   split_train_v2.txt      : 학습 (기존 train에서 10% val 분리 후 90%)
    #   split_zeroshot.txt      : zero-shot 평가 (기존 split_val.txt 이름변경)
    #   split_finetune_val.txt  : fine-tuning epoch 선택 (train에서 stratified 10%) ← NEW
    #   split_test_finetuning.txt: 최종 평가용 (epoch 선택에 사용 안 함 → 누수 제거)
    # 하위호환: 새 파일이 없으면 기존 파일로 폴백
    def _pick(*names):
        for nm in names:
            p = os.path.join(DATA_DIR, nm)
            if os.path.exists(p):
                return p, nm
        return os.path.join(DATA_DIR, names[0]), names[0]

    train_path, train_nm = _pick("split_train_v2.txt", "split_train.txt")
    zs_path,    zs_nm    = _pick("split_zeroshot.txt", "split_val.txt")
    fval_path,  fval_nm  = _pick("split_finetune_val.txt", "split_test_finetuning.txt")
    testft_path, testft_nm = _pick("split_test_finetuning.txt")

    train_pairs    = load_split(train_path)      # 학습
    val_pairs      = load_split(zs_path)          # zero-shot 평가 (eval_subset의 소스)
    finetune_val_pairs = load_split(fval_path)    # epoch 선택 (NEW)
    test_ft_pairs  = load_split(testft_path)      # 최종 평가
    if is_main:
        print(f"   학습:        {train_nm}  ({len(train_pairs)})")
        print(f"   zero-shot:   {zs_nm}  ({len(val_pairs)})")
        print(f"   epoch선택:   {fval_nm}  ({len(finetune_val_pairs)})")
        print(f"   최종평가:    {testft_nm}  ({len(test_ft_pairs)})")

    if args.eval_samples == -1: eval_subset = val_pairs
    else: eval_subset = random.sample(val_pairs, min(len(val_pairs), args.eval_samples))

    if args.test_samples == -1: test_ft_subset = test_ft_pairs
    else: test_ft_subset = random.sample(test_ft_pairs, min(len(test_ft_pairs), args.test_samples))

    # 💡 epoch 선택용 val (train에서 stratified 분리한 10%). test_samples 옵션 공유.
    if args.test_samples == -1: finetune_val_subset = finetune_val_pairs
    else: finetune_val_subset = random.sample(finetune_val_pairs, min(len(finetune_val_pairs), args.test_samples))

    # 💡 [Fix 8] Train도 잘라낼 수 있게 — Phase 3에서 사용
    if args.train_samples == -1: train_subset = train_pairs
    else: train_subset = random.sample(train_pairs, min(len(train_pairs), args.train_samples))

    # ==========================================
    # Phase 0: 데이터 진단 (모델 로드 X, GPU 사용 X — 1분 안에 끝남)
    # ==========================================
    if is_main:
        print(f"\n🩺 [Phase 0] 데이터 파이프 진단 — 100개 샘플로 GT 매핑 점검...")
        diag_ds = Agent1Dataset(val_pairs[:100], None, None, global_cot_map,
                                global_sys_prompt, global_prompt, "qwen", args.num_frames)
        gts = [diag_ds.get_info(i)[1] for i in range(len(diag_ds))]
        unique_gts = len(set(gts))
        unknown_cnt = sum(1 for g in gts if g.count("Unknown") >= 5)
        print(f"   - 점검한 샘플 수    : {len(gts)}")
        print(f"   - 고유 GT 패턴 수   : {unique_gts}  (목표: 20+ 개)")
        print(f"   - Unknown 5줄 도배  : {unknown_cnt}  (목표: 0개)")
        Agent1Dataset._maybe_print_diag(is_main=True)
        if unique_gts < 5 or unknown_cnt > len(gts) * 0.5:
            print("\n⛔ 데이터 파이프가 여전히 망가져 있습니다. 본 실행을 중단합니다.")
            print("   → CSV 헤더, JSON의 traffic_accident_type/accident_type 키, 데이터 경로를 점검하세요.")
            if dist.is_initialized(): dist.destroy_process_group()
            return
        # 💡 [재발방지] finetune_val(epoch 선택용)도 점검 — 이게 Unknown이면 학습이 망가짐
        #   (NAS 경로 깨짐 등으로 GT가 전부 Unknown인 채 학습되는 사고 방지)
        fval_gts = []
        try:
            fval_diag = Agent1Dataset(finetune_val_pairs[:100], None, None, global_cot_map,
                                      global_sys_prompt, global_prompt, "qwen", args.num_frames)
            fval_gts = [fval_diag.get_info(i)[1] for i in range(len(fval_diag))]
        except Exception as e:
            print(f"   ⚠️ finetune_val 진단 실패: {e}")
        if fval_gts:
            fval_unknown = sum(1 for g in fval_gts if g.count("Unknown") >= 5)
            fval_unique = len(set(fval_gts))
            print(f"   - [finetune_val] 고유 GT: {fval_unique}, Unknown 도배: {fval_unknown}/{len(fval_gts)}")
            if fval_unique < 5 or fval_unknown > len(fval_gts) * 0.5:
                print("\n⛔ finetune_val의 GT가 망가져 있습니다 (Unknown 도배). 본 실행을 중단합니다.")
                print("   → NAS 경로/JSON 매핑을 점검하세요. 이대로 학습하면 모델이 Unknown만 출력합니다.")
                if dist.is_initialized(): dist.destroy_process_group()
                return
        print("   ✅ 데이터 파이프 양호 (zero-shot + finetune_val). 다음 단계로 진행합니다.\n")

    if args.phase == "diag":
        if is_main: print("🏁 [--phase diag] 데이터 점검만 수행 후 종료합니다.")
        if dist.is_initialized(): dist.destroy_process_group()
        return
    if dist.is_initialized(): dist.barrier()

    # ==========================================
    # Phase 1: Zero-Shot 평가
    # ==========================================
    if is_main: print(f"\n⚔️ [Phase 1] 3대 모델 Zero-Shot 블라인드 테스트 시작! (평가 개수: {len(eval_subset):,}개)")
    scores = {}
    has_flash_attn = _HAS_FLASH_ATTN

    for m_type, m_id in AVAILABLE_MODELS.items():
        zs_save_path = os.path.join(RESULT_DIR, f"zeroshot_results_agent1_{m_type}.json")

        # 💡 [Fix 25] --only_model: 지정 모델이 아니면 평가하지 않음
        #   (결과 파일이 이미 있으면 점수만 읽어 우승자 집계에는 포함)
        if args.only_model and m_type != args.only_model:
            if os.path.exists(zs_save_path) and is_main:
                try:
                    with open(zs_save_path, "r", encoding="utf-8") as f:
                        scores[m_type] = json.load(f)["summary"]["average_score"]
                    print(f"\n⏭️ [only_model] {m_type.upper()}는 평가 안 함 (기존 점수만 집계: {scores[m_type]:.4f})")
                except Exception: pass
            elif is_main:
                print(f"\n⏭️ [only_model] {m_type.upper()}는 건너뜀 (결과 없음)")
            if dist.is_initialized(): dist.barrier()
            continue

        if os.path.exists(zs_save_path):
            if is_main:
                try:
                    with open(zs_save_path, "r", encoding="utf-8") as f:
                        scores[m_type] = json.load(f)["summary"]["average_score"]
                    print(f"\n✅ [스킵 완료] {m_type.upper()} 모델은 이미 평가됨 (점수: {scores[m_type]:.4f})")
                except Exception: pass
            if dist.is_initialized(): dist.barrier()
            continue

        if is_main: print(f"\n🚀 {m_type.upper()} 모델 등판 및 평가 시작...")

        trust_remote = (m_type == "internvl")
        try: processor = AutoProcessor.from_pretrained(m_id, trust_remote_code=trust_remote)
        except: processor = None

        tokenizer = None
        if m_type == "internvl" or processor is None:
            tokenizer = AutoTokenizer.from_pretrained(m_id, trust_remote_code=trust_remote)
            if processor is None:
                from transformers import CLIPImageProcessor
                processor = CLIPImageProcessor.from_pretrained(m_id, trust_remote_code=trust_remote)

        model_kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": trust_remote}
        if m_type != "internvl": model_kwargs["device_map"] = {"": local_rank}

        if m_type == "qwen":
            model_kwargs["attn_implementation"] = "flash_attention_2" if has_flash_attn else "sdpa"
        elif m_type == "llama":
            # 💡 [Fix 27] Mllama의 flash_attention_2 경로가 Blackwell/현 transformers에서
            #   c10 에러 발생 ("CUDA OOM / C++ Error"로 표시). 단독 테스트로
            #   sdpa + is_causal 패치(Fix 24) 조합의 정상 작동 확인됨 → sdpa 강제.
            model_kwargs["attn_implementation"] = "sdpa"
        elif m_type == "internvl":
            model_kwargs["attn_implementation"] = "eager"
            if has_flash_attn: model_kwargs["use_flash_attn"] = True
            # 💡 [Fix 20] transformers 5.x는 meta device 로딩이 기본인데, InternVL의
            #   vision 코드가 초기화 중 .item()을 호출 → meta tensor에서 불가 → 에러.
            #   low_cpu_mem_usage=False로 예전처럼 실제 device에 바로 생성하게 함.
            model_kwargs["low_cpu_mem_usage"] = False

        try:
            if m_type == "qwen":
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(m_id, **model_kwargs)
            elif m_type == "llama":
                from transformers import MllamaForConditionalGeneration
                model = MllamaForConditionalGeneration.from_pretrained(m_id, **model_kwargs)
            elif m_type == "internvl":
                model = AutoModel.from_pretrained(m_id, **model_kwargs)
        except ValueError:
            model_kwargs["attn_implementation"] = "eager"
            model_kwargs.pop("use_flash_attn", None)
            if m_type == "qwen":
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(m_id, **model_kwargs)
            elif m_type == "llama":
                from transformers import MllamaForConditionalGeneration
                model = MllamaForConditionalGeneration.from_pretrained(m_id, **model_kwargs)
            elif m_type == "internvl":
                model = AutoModel.from_pretrained(m_id, **model_kwargs)

        if m_type == "internvl":
            model = model.to(torch.device(f"cuda:{local_rank}"))
            # 💡 [Fix 13] generate 복원 + img_context_token_id 등록 (tokenizer 필수)
            patch_internvl_generation(model, tokenizer)
        elif m_type == "llama":
            # 💡 [Fix 24] Llama sdpa의 is_causal 버그 우회 (transformers 5.3)
            #   이걸 안 하면 generation 시 'MllamaVisionAttention has no attribute is_causal'로
            #   3,281개 전부 Generation Failed 됨 (agent2/3와 동일 패치).
            patch_mllama_is_causal(model)

        eval_ds = Agent1Dataset(eval_subset, processor, tokenizer, global_cot_map, global_sys_prompt, global_prompt, m_type, args.num_frames)
        preds = generate_descriptions(model, processor, tokenizer, eval_ds, local_rank, world_size, m_type, args.num_frames, RESULT_DIR)

        if is_main and preds:
            detailed_results = []
            total_score = 0
            for p in preds:
                score = text_similarity_score(p['pred'], p['gt'])
                total_score += score
                detailed_results.append({
                    "sample_idx": p["idx"], "video_file": p["video"],
                    "ground_truth": p["gt"], "model_prediction": p["pred"],
                    "similarity_score": round(score, 4)
                })

            avg_score = total_score / len(preds) if preds else 0
            scores[m_type] = avg_score
            print(f"🏅 {m_type.upper()} 모델 Zero-Shot 점수: {avg_score:.4f}")

            with open(zs_save_path, "w", encoding="utf-8") as f:
                json.dump({
                    "summary": {"model_type": m_type, "phase": "Zero-Shot",
                                "average_score": round(avg_score, 4),
                                "total_samples": len(preds)},
                    "details": detailed_results
                }, f, indent=4, ensure_ascii=False)

        del model, processor, tokenizer, eval_ds
        gc.collect()
        torch.cuda.empty_cache()
        if dist.is_initialized(): dist.barrier()

    # ==========================================
    # Phase 2: 1등 모델 선정
    # ==========================================
    best_idx = torch.tensor([0], dtype=torch.long, device=torch.device(f"cuda:{local_rank}"))
    if is_main:
        if not scores:
            print("⛔ 어떤 모델도 점수를 산출하지 못했습니다. 데이터/환경을 확인하세요.")
            if dist.is_initialized(): dist.destroy_process_group()
            return
        best_m = max(scores, key=scores.get)
        print(f"\n🏆 [Phase 2] 경연 결과! 우승은 '{best_m.upper()}' (점수 {scores[best_m]:.4f})")
        best_idx[0] = list(AVAILABLE_MODELS.keys()).index(best_m)

    if dist.is_initialized(): dist.broadcast(best_idx, src=0)
    winner_type = list(AVAILABLE_MODELS.keys())[best_idx.item()]
    winner_id = AVAILABLE_MODELS[winner_type]

    # 💡 [Fix 8] --phase zeroshot 이면 Phase 2까지만 하고 종료
    # 💡 [Fix 25] --only_model 지정 시에도 파인튜닝 없이 종료 (3개 다 모은 후 --phase all로)
    if args.phase == "zeroshot" or args.only_model:
        if is_main:
            n_done = len(scores)
            print(f"\n🏁 Zero-Shot 단계 종료 (현재 결과 {n_done}/3개 모델).")
            if n_done >= 3:
                print(f"   🏆 현재 우승: {winner_type.upper()} — 파인튜닝은 --phase all (only_model 없이)로 진행하세요.")
            else:
                missing = [m for m in AVAILABLE_MODELS if m not in scores]
                print(f"   ⏳ 아직 평가 안 된 모델: {missing} — 마저 평가 후 파인튜닝 하세요.")
        if dist.is_initialized(): dist.destroy_process_group()
        return

    # ==========================================
    # Phase 3: 파인튜닝
    # ==========================================
    FT_MODEL_DIR = os.path.join(RESULT_DIR, f"finetuned_model_agent1_{winner_type}")
    os.makedirs(FT_MODEL_DIR, exist_ok=True)

    ft_save_path = os.path.join(RESULT_DIR, f"finetuned_results_agent1_{winner_type}.json")
    if os.path.exists(ft_save_path):
        if is_main: print("\n✅ 파인튜닝/최종 평가 결과가 이미 존재합니다. 종료합니다.")
        if dist.is_initialized(): dist.destroy_process_group()
        return

    if is_main: print(f"\n🚀 [Phase 3] 1등 {winner_type.upper()} 모델 파인튜닝 돌입... (학습 샘플: {len(train_subset):,}개)")

    trust_remote = (winner_type == "internvl")
    try: processor = AutoProcessor.from_pretrained(winner_id, trust_remote_code=trust_remote)
    except: processor = None

    tokenizer = None
    if winner_type == "internvl" or processor is None:
        tokenizer = AutoTokenizer.from_pretrained(winner_id, trust_remote_code=trust_remote)
        if processor is None:
            from transformers import CLIPImageProcessor
            processor = CLIPImageProcessor.from_pretrained(winner_id, trust_remote_code=trust_remote)

    model_kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": trust_remote}
    if winner_type in ["qwen", "llama"]:
        model_kwargs["device_map"] = {"": local_rank}
        if winner_type == "llama":
            model_kwargs["attn_implementation"] = "sdpa"  # [Fix 27]
        else:
            model_kwargs["attn_implementation"] = "flash_attention_2" if has_flash_attn else "sdpa"
    elif winner_type == "internvl":
        model_kwargs["attn_implementation"] = "eager"
        if has_flash_attn: model_kwargs["use_flash_attn"] = True
        # 💡 [Fix 20] transformers 5.x meta device 로딩 우회 (InternVL .item() 에러 방지)
        model_kwargs["low_cpu_mem_usage"] = False

    try:
        if winner_type == "qwen":
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(winner_id, **model_kwargs)
        elif winner_type == "llama":
            from transformers import MllamaForConditionalGeneration
            model = MllamaForConditionalGeneration.from_pretrained(winner_id, **model_kwargs)
        elif winner_type == "internvl":
            model = AutoModel.from_pretrained(winner_id, **model_kwargs)
    except ValueError:
        model_kwargs["attn_implementation"] = "eager"
        model_kwargs.pop("use_flash_attn", None)
        if winner_type == "qwen":
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(winner_id, **model_kwargs)
        elif winner_type == "llama":
            from transformers import MllamaForConditionalGeneration
            model = MllamaForConditionalGeneration.from_pretrained(winner_id, **model_kwargs)
        elif winner_type == "internvl":
            model = AutoModel.from_pretrained(winner_id, **model_kwargs)

    if winner_type == "internvl":
        model = model.to(torch.device(f"cuda:{local_rank}"))
        patch_internvl_generation(model, tokenizer)
        # 💡 [Fix 15] 학습용 forward 래퍼는 PEFT로 감싸기 전에 적용
        patch_internvl_for_training(model)
    elif winner_type == "llama":
        # 💡 [Fix 24] Llama sdpa is_causal 패치 (학습/추론 모두 필요)
        patch_mllama_is_causal(model)
    if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()

    # 💡 [Fix 16] InternVL은 vision+language 복합 구조라 all-linear가 vision tower까지
    #   LoRA를 씌워 메모리 폭증/불안정을 유발 → language_model의 Linear만 자동 탐지해 타겟팅.
    #   (모듈 이름을 추측하지 않고 실제 모델에서 찾아내므로 InternVL2/3 어느 쪽이든 동작)
    if winner_type == "internvl":
        import torch.nn as nn
        target_set = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "language_model" in name:
                leaf = name.split(".")[-1]
                # lm_head, output 등 최종 출력층은 제외 (LoRA 부적합)
                if leaf not in ("lm_head", "output", "embed_tokens"):
                    target_set.add(leaf)
        target_modules = sorted(target_set)
        if is_main:
            print(f"🎯 [Fix 16] InternVL LoRA 타겟 모듈 자동 탐지: {target_modules}")
        if not target_modules:
            # 혹시 못 찾으면 all-linear로 폴백 (최소한 죽지는 않게)
            target_modules = "all-linear"
            if is_main: print("⚠️ language_model Linear 미탐지 → all-linear 폴백")
        lora_cfg = LoraConfig(
            r=64, lora_alpha=128,
            target_modules=target_modules,
            task_type="CAUSAL_LM"
        )
    elif winner_type == "llama":
        # 💡 [Fix 28 v2] Mllama는 vision도 k_proj/q_proj 등 같은 leaf 이름을 사용
        #   → leaf 이름 타겟팅은 vision tower에도 LoRA를 붙여 OOM 유발 (94GB 폭발 확인).
        #   풀 경로 이름 리스트로 전달하면 PEFT가 정확히 그 모듈만 매치
        #   → vision 완전 프리즈 → vision activation이 grad 그래프에 안 잡혀 메모리 급감.
        import torch.nn as nn
        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "language_model" in name:
                leaf = name.split(".")[-1]
                if leaf not in ("lm_head", "output", "embed_tokens"):
                    target_modules.append(name)
        if is_main:
            leafs = sorted(set(n.split(".")[-1] for n in target_modules))
            print(f"🎯 [Fix 28 v2] Llama LoRA: language_model 내 Linear {len(target_modules)}개 정확 타겟")
            print(f"   leaf 종류: {leafs}")
        if not target_modules:
            target_modules = "all-linear"
            if is_main: print("⚠️ language_model Linear 미탐지 → all-linear 폴백")
        lora_cfg = LoraConfig(
            r=64, lora_alpha=128,
            target_modules=target_modules,
            task_type="CAUSAL_LM"
        )
    else:
        lora_cfg = LoraConfig(r=64, lora_alpha=128, target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)

    # 💡 [Fix 29] Llama: 프리즈된 vision tower의 activation 저장 차단
    #   vision은 LoRA 타겟이 아니라 완전 프리즈인데도, 학습 모드에서 중간 activation
    #   (32이미지 × 4타일 × 1601토큰 × 40레이어 ≈ 50~100GB)이 backward용으로 저장돼
    #   94GB OOM 유발. vision forward를 no_grad로 감싸면 원천 차단되고,
    #   cross_attention_states는 상수로 language에 전달돼 LoRA 학습은 정상 작동.
    if winner_type == "llama":
        import functools
        _vision = None
        for _name, _mod in model.named_modules():
            if _name.endswith("vision_model"):
                _vision = _mod
                break
        if _vision is not None:
            _orig_vfwd = _vision.forward
            @functools.wraps(_orig_vfwd)
            def _no_grad_vfwd(*a, **k):
                with torch.no_grad():
                    return _orig_vfwd(*a, **k)
            _vision.forward = _no_grad_vfwd
            for p in _vision.parameters():
                p.requires_grad_(False)
            if is_main:
                print("✅ [Fix 29] vision_model forward를 no_grad로 봉인 (activation 저장 차단)")
        else:
            if is_main: print("⚠️ [Fix 29] vision_model 미발견 — 건너뜀")

    train_ds = Agent1Dataset(train_subset, processor, tokenizer, global_cot_map, global_sys_prompt, global_prompt, winner_type, args.num_frames)
    # 💡 [Fix 17] InternVL의 실제 num_image_token을 Dataset에 주입 (모델마다 다를 수 있음)
    if winner_type == "internvl":
        base_model = model.base_model.model if hasattr(model, "base_model") else model
        nit = getattr(base_model, "num_image_token", 256)
        train_ds._num_image_token = nit
        if is_main: print(f"📐 [InternVL] num_image_token = {nit} (프레임당 visual token 수)")

    pad_token_id = 0
    if tokenizer and getattr(tokenizer, "pad_token_id", None) is not None:
        pad_token_id = tokenizer.pad_token_id
    elif processor and hasattr(processor, "tokenizer") and getattr(processor.tokenizer, "pad_token_id", None) is not None:
        pad_token_id = processor.tokenizer.pad_token_id

    # 💡 [Fix 18] InternVL은 시퀀스가 8400토큰(이미지 4096+텍스트)으로 매우 길어
    #   batch가 크면 OOM 위험 + num_workers>0이면 worker 복제 충돌.
    #   기본값은 보수적으로 두되, --batch_size로 덮어쓸 수 있게 함.
    if winner_type == "internvl":
        per_dev_batch, grad_accum, n_workers = 2, 2, 0
    else:
        per_dev_batch, grad_accum, n_workers = 2, 2, 4
    # --batch_size 인자가 주어지면 덮어쓰기 (grad_accum은 effective batch 4 유지하도록 자동 조정)
    if args.batch_size and args.batch_size > 0:
        per_dev_batch = args.batch_size
        grad_accum = max(1, 4 // per_dev_batch)

    if is_main:
        print(f"🔧 [학습 설정] batch={per_dev_batch}, grad_accum={grad_accum}, "
              f"max_epochs={args.epochs}, workers={n_workers} (effective batch={per_dev_batch*grad_accum})")

    # =========================================================================
    # 💡 [Fix 22] Epoch별 학습 + 평가 + Early Stopping (patience=1) + Best 저장 + 이어하기
    #   - 매 epoch마다 1 epoch씩 학습 → test 전체 평가 → 점수 비교
    #   - 점수가 best보다 떨어지면(patience=1) 즉시 중단, 직전 best epoch를 최종 채택
    #   - best 모델은 BEST_DIR에 별도 저장, 진행상황은 progress.json에 기록 (이어하기용)
    # =========================================================================
    BEST_DIR = os.path.join(RESULT_DIR, f"best_model_agent1_{winner_type}")
    progress_path = os.path.join(RESULT_DIR, f"train_progress_agent1_{winner_type}.json")
    # 💡 [누수 제거] epoch 선택(early stopping)은 finetune_val로 수행.
    #   test_ft(split_test_finetuning)는 최종 평가 전용으로 분리 → 모델 선택에 미사용.
    eval_sel_ds = Agent1Dataset(finetune_val_subset, processor, tokenizer, global_cot_map,
                                global_sys_prompt, global_prompt, winner_type, args.num_frames)
    if winner_type == "internvl":
        eval_sel_ds._num_image_token = nit

    def evaluate_current(tag):
        """현재 model 상태로 finetune_val 전체 평가 → 평균 점수 + 상세 반환 (epoch 선택용)"""
        preds = generate_descriptions(model, processor, tokenizer, eval_sel_ds,
                                       local_rank, world_size, winner_type,
                                       args.num_frames, RESULT_DIR)
        # rank0만 점수 집계, 나머지는 None
        if not (is_main and preds):
            return None, None
        details, total = [], 0.0
        for p in preds:
            s = text_similarity_score(p['pred'], p['gt'])
            total += s
            details.append({"sample_idx": p["idx"], "video_file": p["video"],
                            "ground_truth": p["gt"], "model_prediction": p["pred"],
                            "similarity_score": round(s, 4)})
        avg = total / len(preds) if preds else 0.0
        # temp 평가 파일 정리 (다음 epoch 평가가 이어쓰기 안 하도록)
        for r in range(world_size):
            tf = os.path.join(RESULT_DIR, f"temp_agent1_{winner_type}_rank{r}.json")
            if os.path.exists(tf):
                try: os.remove(tf)
                except: pass
        return avg, details

    # ---- zero-shot 점수를 모든 rank에 공유 (scores는 rank0에만 채워져 있음) ----
    zs_tensor = torch.tensor(
        [scores.get(winner_type, 0.0)], dtype=torch.float64,
        device=torch.device(f"cuda:{local_rank}")
    )
    if dist.is_initialized():
        dist.broadcast(zs_tensor, src=0)
    zero_shot_score = float(zs_tensor.item())

    # ---- 이어하기: 기존 진행상황 로드 ----
    start_epoch = 1
    best_score = zero_shot_score   # 시작 기준점 = zero-shot 점수
    best_details = None
    best_epoch = 0   # 0 = 아직 fine-tune best 없음 (zero-shot이 best)
    no_improve = 0   # 💡 연속 하락 횟수 (patience용)
    if os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                prog = json.load(f)
            start_epoch = prog.get("next_epoch", 1)
            best_score = prog.get("best_score", best_score)
            best_epoch = prog.get("best_epoch", 0)
            if is_main:
                print(f"\n🔄 [이어하기] epoch {start_epoch}부터 재개 "
                      f"(현재 best: epoch {best_epoch}, score {best_score:.4f})")
        except Exception as e:
            if is_main: print(f"⚠️ 진행상황 로드 실패, 처음부터: {e}")

    stopped_reason = "최대 epoch 도달"

    # =========================================================================
    # Epoch 루프
    # =========================================================================
    for epoch in range(start_epoch, args.epochs + 1):
        if is_main: print(f"\n{'='*50}\n🚀 [Epoch {epoch}/{args.epochs}] 학습 시작\n{'='*50}")

        # ---- 이전 epoch 체크포인트에서 이어서 학습 (가중치 연속성 유지) ----
        epoch_ckpt_dir = os.path.join(FT_MODEL_DIR, f"epoch_{epoch}")
        prev_ckpt_dir = os.path.join(FT_MODEL_DIR, f"epoch_{epoch-1}")

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=epoch_ckpt_dir,
                per_device_train_batch_size=per_dev_batch,
                gradient_accumulation_steps=grad_accum,
                num_train_epochs=1,          # 한 번에 1 epoch만
                learning_rate=2e-5,
                bf16=True,
                logging_steps=50,
                save_strategy="no",          # epoch 단위로 우리가 직접 저장
                eval_strategy="no",
                ddp_find_unused_parameters=False,
                gradient_checkpointing=True,
                # 💡 [Fix 23b] reentrant checkpointing이 DDP와 충돌해
                #   'parameter did not receive grad' 에러를 일으킨다.
                #   non-reentrant 모드로 켜면 DDP 충돌 해결 (agent3와 동일).
                gradient_checkpointing_kwargs={"use_reentrant": False},
                dataloader_num_workers=n_workers,
                remove_unused_columns=False,
                report_to="none",
            ),
            train_dataset=train_ds,
            data_collator=lambda x: dynamic_collate_fn(x, pad_token_id, winner_type),
        )
        trainer.train()

        # 💡 [Fix 23c] Trainer는 첫 호출 시 model을 DDP/multi-device로 감싼다.
        #   두 번째 epoch에서 새 Trainer가 이미 감싸진 model을 받으면
        #   "model is already on multiple devices" 경고 후 SIGABRT.
        #   해결: 매 epoch 종료 후 (a) trainer 내부의 wrap 해제,
        #         (b) 다음 Trainer가 새로 감쌀 수 있게 hf_device_map 흔적 제거.
        try:
            # Trainer가 wrap한 DDP 모듈에서 원본 PEFT 모델만 추출
            wrapped = getattr(trainer, "model_wrapped", None)
            if wrapped is not None and wrapped is not model:
                # DDP/FSDP 래퍼에는 보통 .module 속성에 원본이 있음
                if hasattr(wrapped, "module"):
                    pass   # model은 이미 원본을 가리키므로 추가 작업 불필요
            # 모델에 "이미 분산됨" 표식이 있으면 제거 (다음 Trainer가 다시 처리하게)
            if hasattr(model, "hf_device_map"):
                try: del model.hf_device_map
                except: pass
            base = getattr(model, "base_model", None)
            if base is not None and hasattr(base, "hf_device_map"):
                try: del base.hf_device_map
                except: pass
        except Exception as e:
            if is_main: print(f"⚠️ unwrap 정리 중 경고: {e}")

        del trainer
        gc.collect(); torch.cuda.empty_cache()
        # 모든 rank가 정리 끝나길 기다린 뒤 다음 epoch 진입
        if dist.is_initialized(): dist.barrier()

        # ---- 이번 epoch 평가 (test 전체) ----
        if is_main: print(f"\n📈 [Epoch {epoch}] 평가 — Test 세트 전체...")
        avg_score, details = evaluate_current(f"epoch{epoch}")

        # rank0가 점수 판정 후 모든 rank에 brodcast (중단 동기화)
        stop_signal = torch.tensor([0], dtype=torch.long, device=torch.device(f"cuda:{local_rank}"))
        if is_main:
            print(f"   → Epoch {epoch} 점수: {avg_score:.4f} (이전 best: {best_score:.4f}, "
                  f"zero-shot: {zero_shot_score:.4f})")
            improved = avg_score > best_score
            if improved:
                best_score = avg_score
                best_epoch = epoch
                best_details = details
                no_improve = 0   # 💡 개선되면 카운터 리셋
                # best 모델 별도 저장
                os.makedirs(BEST_DIR, exist_ok=True)
                model.save_pretrained(BEST_DIR)
                try:
                    if processor: processor.save_pretrained(BEST_DIR)
                    if tokenizer: tokenizer.save_pretrained(BEST_DIR)
                except: pass
                print(f"   ✅ 신기록! best 모델 저장 (epoch {epoch}, {best_score:.4f})")
            else:
                # 💡 patience: N번 연속 하락하면 중단 (best는 유지)
                no_improve += 1
                print(f"   ⚠️ 점수 하락 (연속 {no_improve}/{args.patience}). best=epoch {best_epoch}({best_score:.4f}) 유지")
                if no_improve >= args.patience:
                    stop_signal[0] = 1
                    stopped_reason = f"epoch {epoch}에서 {no_improve}번 연속 하락 (patience={args.patience}) → epoch {best_epoch} 채택"
                    print(f"   ⛔ patience 소진 — 직전 best(epoch {best_epoch})를 최종 채택하고 중단합니다.")

            # 진행상황 저장 (이어하기용)
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump({
                    "next_epoch": epoch + 1,
                    "best_epoch": best_epoch,
                    "best_score": round(best_score, 4),
                    "last_epoch_score": round(avg_score, 4) if avg_score else None,
                }, f, indent=2, ensure_ascii=False)

        if dist.is_initialized(): dist.broadcast(stop_signal, src=0)
        if stop_signal.item() == 1:
            break

    # =========================================================================
    # Phase 4: 최종 결과 정리 (best epoch 기준)
    # =========================================================================
    if is_main:
        print("\n==================================================")
        print("🎉🎉 [Agent 1 파이프라인 완주 대성공!] 🎉🎉")
        print(f"👑 최종 선정 모델 : {winner_type.upper()}")
        print(f"🛑 종료 사유      : {stopped_reason}")
        if best_epoch == 0:
            # fine-tuning이 zero-shot을 한 번도 못 넘음
            print(f"📊 Zero-Shot 정확도 : {zero_shot_score:.4f}")
            print(f"⚠️ 파인튜닝이 zero-shot을 넘지 못했습니다. zero-shot 모델을 권장합니다.")
            final_score = zero_shot_score
        else:
            print(f"🏅 최종 채택 Epoch : {best_epoch}")
            print(f"📊 Fine-Tuning 전(Zero-Shot) 정확도 : {zero_shot_score:.4f}")
            print(f"📈 Fine-Tuning 후 (Best Epoch {best_epoch}) 정확도 : {best_score:.4f}")
            print(f"🔥 상황 묘사 정확도 향상폭          : +{(best_score - zero_shot_score):.4f}")
            final_score = best_score
        print("==================================================")

        with open(ft_save_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "model_type": winner_type, "phase": "Fine-Tuned (Best Epoch)",
                    "best_epoch": best_epoch,
                    "stopped_reason": stopped_reason,
                    "average_score_before(Zero-Shot)": round(zero_shot_score, 4),
                    "average_score_after(Best)": round(final_score, 4),
                    "improvement_gap": round(final_score - zero_shot_score, 4),
                    "total_samples": len(best_details) if best_details else 0
                },
                "details": best_details if best_details else []
            }, f, indent=4, ensure_ascii=False)
        print(f"💾 결과 저장: {ft_save_path}")
        print(f"💾 Best 모델: {BEST_DIR}")

    # =========================================================================
    # 💡 [최종 평가] best 모델(finetune_val로 선택됨)로 test_finetuning을 단 한 번 평가.
    #   epoch 선택엔 finetune_val을, 최종 보고엔 test_ft를 씀 → 누수 없는 정직한 test 점수.
    #   (best_epoch==0이면 fine-tuning 실패라 생략)
    # =========================================================================
    final_test_score = None
    if best_epoch and best_epoch > 0 and os.path.exists(BEST_DIR):
        if is_main:
            print("\n🧪 [최종 평가] best 모델로 test_finetuning 평가 (epoch 선택에 미사용 → 누수 없음)...")
        try:
            from peft import PeftModel as _PeftModel
            # best 어댑터 로드 (현재 model에 덮어쓰기)
            if hasattr(model, "load_adapter"):
                try:
                    model.load_adapter(BEST_DIR, adapter_name="best_final")
                    model.set_adapter("best_final")
                except Exception:
                    pass
            test_ft_ds_final = Agent1Dataset(test_ft_subset, processor, tokenizer, global_cot_map,
                                             global_sys_prompt, global_prompt, winner_type, args.num_frames)
            if winner_type == "internvl":
                test_ft_ds_final._num_image_token = nit
            preds = generate_descriptions(model, processor, tokenizer, test_ft_ds_final,
                                          local_rank, world_size, winner_type,
                                          args.num_frames, RESULT_DIR)
            if is_main and preds:
                tot, cnt = 0.0, 0
                test_details = []
                for p in preds:
                    sc = text_similarity_score(p["pred"], p["gt"])
                    tot += sc; cnt += 1
                    test_details.append({"video": p.get("video", ""),
                                         "model_prediction": p["pred"],
                                         "ground_truth": p["gt"],
                                         "score": round(sc, 4)})
                final_test_score = tot / cnt if cnt else 0.0
                print(f"📊 [최종 test 점수] test_finetuning = {final_test_score:.4f} "
                      f"(val 기준 best={best_score:.4f})")
                # 최종 test 결과 별도 저장
                test_report_path = os.path.join(RESULT_DIR, f"final_test_agent1_{winner_type}.json")
                with open(test_report_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "summary": {
                            "model_type": winner_type,
                            "best_epoch": best_epoch,
                            "val_score(finetune_val, for selection)": round(best_score, 4),
                            "test_score(test_finetuning, final report)": round(final_test_score, 4),
                            "zero_shot_score": round(zero_shot_score, 4),
                            "total_samples": cnt,
                        },
                        "details": test_details,
                    }, f, indent=4, ensure_ascii=False)
                print(f"💾 최종 test 결과 저장: {test_report_path}")
        except Exception as e:
            if is_main: print(f"⚠️ 최종 test 평가 중 오류(무시 가능, val 점수는 유효): {e}")
        if dist.is_initialized(): dist.barrier()

    if dist.is_initialized(): dist.destroy_process_group()
if __name__ == "__main__":
    main()
