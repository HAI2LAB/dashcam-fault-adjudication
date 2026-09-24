#!/usr/bin/env python
"""
End-to-End 과실 파이프라인 학습/평가
사용법:
  python -m torch.distributed.run --nproc_per_node=4 e2e_train.py \
      --model internvl --num_frames 32 --epochs 10 --patience 2

  --model : internvl | qwen | qwen3
"""
import os, sys, json, argparse, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_model3_ms import (
    VLMBackbone, TripleHeadFaultModel, apply_lora,
    load_split, get_fault_gt, compare_label, PROMPT, DATA_DIR,
    _get_field, CSV_MAP, get_situation_gt, SITUATION_CLASSES,
    ratio_to_class, RATIO_VALUES, N_RATIO,
)
from agent1_pipeline import extract_frames

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          os.environ.get("E2E_RESULT_DIR", "result_e2e3"))
os.makedirs(RESULT_DIR, exist_ok=True)

def setup():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        lr = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
        return lr, dist.get_world_size()
    return 0, 1

def pick(*names):
    here = os.path.dirname(os.path.abspath(__file__))
    for nm in names:
        for base in (DATA_DIR, here):          # NAS 우선, 없으면 TS_code
            p = os.path.join(base, nm)
            if os.path.exists(p): return p
    return os.path.join(DATA_DIR, names[0])

def build_samples(pairs, nframes, cap=-1, debug=False):
    """(frames_path, jp, cmp_label, ratio_a) 사전 라벨만 (프레임은 학습 때 로드)"""
    out = []
    n_pairs = len(pairs)
    n_gt_none = 0
    for vp, jp in pairs:
        gt = get_fault_gt(jp)
        if gt is None:
            n_gt_none += 1
            continue
        ra, rb = gt
        sit = get_situation_gt(jp)
        if sit is None: sit = len(SITUATION_CLASSES) - 1  # Other
        out.append((vp, jp, compare_label(ra, rb), ra, sit))
        if cap > 0 and len(out) >= cap: break
    if debug:
        print(f"   [build_samples] pairs={n_pairs}, gt_none={n_gt_none}, ok={len(out)}")
        if n_pairs > 0 and len(out) == 0:
            vp0, jp0 = pairs[0]
            print(f"   [진단] 첫 pair: vp={vp0[:50]}, jp={jp0[:50]}")
            print(f"   [진단] jp 존재: {os.path.exists(jp0)}")
            try:
                jj = json.load(open(jp0, encoding='utf-8'))
                print(f"   [진단] JSON 키: {list(jj.keys())[:8]}")
                at = _get_field(jj, ['traffic_accident_type','accident_type'])
                print(f"   [진단] accident_type: {at}")
                print(f"   [진단] CSV에 {at} 있나: {int(at) in CSV_MAP if at else 'N/A'}")
            except Exception as e:
                print(f"   [진단] JSON 로드 실패: {e}")
    return out

@torch.no_grad()
def evaluate(dh, samples, local_rank, world_size, nframes, is_main, tag):
    dh.eval()
    idxs = list(range(local_rank, len(samples), world_size))
    recs = []
    ev_cmp_ok, ev_sit_ok, ev_w10 = 0, 0, 0
    pbar = tqdm(idxs, desc=f"[{tag}] GPU{local_rank}", position=local_rank, disable=not is_main)
    for cnt, i in enumerate(pbar, 1):
        vp, jp, cmp_gt, ra_gt, sit_gt = samples[i]
        try:
            frames = extract_frames(vp, nframes)
            sit_logits, cmp_logits, ratio_logits = dh(frames, PROMPT)
            sit_pred = int(sit_logits.argmax(-1).item())
            cmp_pred = int(cmp_logits.argmax(-1).item())
            ra_cls = int(ratio_logits.argmax(-1).item())     # 0~10
            ra_pred = RATIO_VALUES[ra_cls]                   # → 0,10,...,100
        except Exception:
            sit_pred, cmp_pred, ra_pred = 12, 1, 50
        recs.append({"idx": i, "cmp_pred": cmp_pred, "cmp_gt": cmp_gt,
                     "ra_pred": ra_pred, "ra_gt": ra_gt,
                     "sit_pred": sit_pred, "sit_gt": sit_gt})
        if sit_pred == sit_gt: ev_sit_ok += 1
        if cmp_pred == cmp_gt: ev_cmp_ok += 1
        if abs(ra_pred - ra_gt) <= 10: ev_w10 += 1
        if is_main and cnt % 20 == 0:
            pbar.set_postfix({"상황": f"{ev_sit_ok/cnt:.2f}", "비교": f"{ev_cmp_ok/cnt:.2f}",
                              "비율w10": f"{ev_w10/cnt:.2f}"})
    tf = os.path.join(RESULT_DIR, f"temp_{tag}_rank{local_rank}.json")
    json.dump(recs, open(tf, "w"))
    if dist.is_initialized(): dist.barrier()
    if local_rank == 0:
        allr = []
        for r in range(world_size):
            f = os.path.join(RESULT_DIR, f"temp_{tag}_rank{r}.json")
            if os.path.exists(f): allr.extend(json.load(open(f))); os.remove(f)
        n = len(allr)
        cmp_acc = sum(1 for x in allr if x["cmp_pred"] == x["cmp_gt"]) / max(n, 1)
        # macro-F1
        from collections import defaultdict
        tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
        for x in allr:
            if x["cmp_pred"] == x["cmp_gt"]: tp[x["cmp_gt"]] += 1
            else: fp[x["cmp_pred"]] += 1; fn[x["cmp_gt"]] += 1
        f1s = []
        for c in range(3):
            p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0
            r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0
            f1s.append(2 * p * r / (p + r) if (p + r) else 0)
        macro_f1 = sum(f1s) / 3
        mae = sum(abs(x["ra_pred"] - x["ra_gt"]) for x in allr) / max(n, 1)
        within10 = sum(1 for x in allr if abs(x["ra_pred"] - x["ra_gt"]) <= 10) / max(n, 1)
        sit_acc = sum(1 for x in allr if x["sit_pred"] == x["sit_gt"]) / max(n, 1)
        # 5구간 실무 평가 (A압도열세/열세/대등/우세/압도우세)
        def band5(ra):
            if ra <= 10: return 0     # A 압도적 열세 (B 주책임)
            if ra <= 30: return 1     # A 열세
            if ra <= 60: return 2     # 대등
            if ra <= 80: return 3     # A 우세
            return 4                  # A 압도적 우세
        band_correct = sum(1 for x in allr if band5(x["ra_pred"]) == band5(x["ra_gt"]))
        band5_acc = band_correct / max(n, 1)
        band_pred_dist = dict(Counter(band5(x["ra_pred"]) for x in allr))
        band_gt_dist = dict(Counter(band5(x["ra_gt"]) for x in allr))
        return {"n": n, "cmp_acc": cmp_acc, "cmp_macro_f1": macro_f1,
                "ratio_mae": mae, "ratio_within10": within10, "sit_acc": sit_acc,
                "ratio_band5_acc": band5_acc,
                "band5_pred_dist": band_pred_dist, "band5_gt_dist": band_gt_dist,
                "cmp_pred_dist": dict(Counter(x["cmp_pred"] for x in allr)),
                "cmp_gt_dist": dict(Counter(x["cmp_gt"] for x in allr)),
                "preds": allr}   # 개별 예측 저장 (추가 분석용)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["internvl", "qwen", "qwen3"], default="internvl")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_reg", type=float, default=1.0)
    ap.add_argument("--lambda_sit", type=float, default=0.2)
    ap.add_argument("--lambda_cmp", type=float, default=1.0, help="[ms] 비교 loss 가중치; 0이면 ratio-only baseline")
    ap.add_argument("--select", default="f1_w10", choices=["f1_w10","w10"], help="[ms] best 선택 기준; ratio-only는 w10")
    ap.add_argument("--samples", type=int, default=-1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--no_checkpoint", action="store_true", help="gradient checkpointing 끄기 (16프레임 속도↑)")
    ap.add_argument("--eval_only", action="store_true", help="학습 없이 저장된 best로 test만 평가 (5구간 등 추가지표)")
    ap.add_argument("--cmp_loss", default="ce", choices=["ce","focal"], help="비교 헤드 손실 (focal: gamma=2, 동등클래스 대응 ablation)")
    ap.add_argument("--train_split", default="split_train_v2.txt", help="학습 스플릿 (bb_1 ablation: split_train_v2_bb1.txt)")
    ap.add_argument("--val_split", default="split_finetune_val.txt", help="검증 스플릿")
    ap.add_argument("--test_split", default="split_test_finetuning.txt", help="평가용 스플릿 파일명 (eval_only 2차 평가: split_zeroshot.txt)")
    ap.add_argument("--seed", type=int, default=42, help="재현성용 시드 (모든 실험 동일)")
    ap.add_argument("--ratio_classes", type=int, default=11, choices=[11, 5],
                    help="비율 class 수 (11=정밀 / 5=실무구간 직접학습)")
    args = ap.parse_args()

    # 비율 class 모드 설정 (모델 생성 전!)
    import e2e_model3_ms as _m3
    _m3.set_ratio_classes(args.ratio_classes)
    global RATIO_VALUES, N_RATIO
    from e2e_model3_ms import RATIO_VALUES, N_RATIO
    cls_tag = "" if args.ratio_classes == 11 else "_5c"   # 파일명 구분 (11class는 기본)
    split_tag = "" if args.test_split == "split_test_finetuning.txt" else "_" + args.test_split.replace("split_","").replace(".txt","")

    # 시드 고정 (논문 재현성)
    import random as _random
    _random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    local_rank, world_size = setup()
    is_main = (local_rank == 0)
    if is_main:
        print(f"🚀 End-to-End 과실 파이프라인 | 모델: {args.model} | {args.num_frames}프레임 | 비율 {args.ratio_classes}-class")
        print(f"   loss = CE(비교) + {args.lambda_reg}·SmoothL1(비율)")

    # 데이터
    train_pairs = load_split(pick(args.train_split, "split_train.txt"))
    val_pairs   = load_split(pick(args.val_split, "split_val.txt"))
    test_pairs  = load_split(pick(args.test_split))
    cap = args.samples
    train_s = build_samples(train_pairs, args.num_frames, cap, debug=is_main)
    val_s   = build_samples(val_pairs, args.num_frames, max(40, cap // 4) if cap > 0 else -1)
    test_s  = build_samples(test_pairs, args.num_frames, cap)
    if is_main:
        print(f"📊 train {len(train_s)} / val {len(val_s)} / test {len(test_s)}")
        print(f"   비교 분포(train): {dict(Counter(s[2] for s in train_s))}")

    # 클래스 가중치 (비교 3-class만, sqrt로 완화 — 극단 방지)
    # 클래스 가중치는 사용 안 함 (F1을 오히려 악화시킴 — 실험으로 확인)
    if is_main:
        print(f"   (클래스 가중치 미사용 — 균형 학습)")

    # 모델
    backbone = VLMBackbone(args.model, local_rank)
    backbone = apply_lora(backbone, is_main, use_checkpoint=not args.no_checkpoint)
    dh = TripleHeadFaultModel(backbone)

    # 옵티마이저 (LoRA + 3헤드)
    params = [p for p in dh.backbone.model.parameters() if p.requires_grad]
    params += list(dh.sit_head.parameters()) + list(dh.cmp_head.parameters()) + list(dh.reg_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    # cosine LR 스케줄링 (전체 step 기준)
    steps_per_epoch = max(1, len(range(local_rank, len(train_s), world_size)) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=args.lr * 0.1)

    # ── eval_only: 학습 없이 저장된 best로 test만 (5구간 등 추가지표 재계산) ──
    BEST = os.path.join(RESULT_DIR, f"best_{args.model}_{args.num_frames}f{cls_tag}")
    if args.eval_only:
        if is_main: print("\n📊 [eval_only] 저장된 best 로드 → test 평가만")
        heads_path = os.path.join(BEST, "heads.pt")
        if os.path.exists(heads_path):
            ck = torch.load(heads_path, map_location=backbone.device)
            dh.sit_head.load_state_dict(ck["sit"]); dh.cmp_head.load_state_dict(ck["cmp"]); dh.reg_head.load_state_dict(ck["reg"])
            if is_main: print("   ✅ best heads 로드")
        else:
            if is_main: print(f"   ⚠️ heads.pt 없음: {heads_path} — 학습 먼저 필요")
        # LoRA 로드 — 저장된 가중치를 '기존 default 어댑터'에 주입 (학습과 동일 구조 유지, Fix15 래퍼 보존)
        try:
            from peft.utils import set_peft_model_state_dict
            ad_st = os.path.join(BEST, "adapter_model.safetensors")
            ad_bin = os.path.join(BEST, "adapter_model.bin")
            if os.path.exists(ad_st):
                from safetensors.torch import load_file
                sd = load_file(ad_st)
            elif os.path.exists(ad_bin):
                sd = torch.load(ad_bin, map_location=backbone.device)
            else:
                raise FileNotFoundError(f"adapter 가중치 없음: {BEST}")
            res = set_peft_model_state_dict(dh.backbone.model, sd, adapter_name="default")
            missing = getattr(res, "missing_keys", [])
            lora_missing = [k for k in missing if "lora_" in k]
            if is_main:
                print(f"   ✅ best LoRA 가중치 주입 (keys {len(sd)}, lora누락 {len(lora_missing)})")
                if lora_missing: print(f"   ⚠️ 누락 예시: {lora_missing[:2]}")
            dh.backbone.model.eval()
        except Exception as e:
            if is_main: print(f"   ❌ LoRA 로드 실패: {str(e)[:100]}")
            raise
        # 전체 eval 모드 (헤드 dropout 0.3까지 완전 차단 — 결정론적 평가)
        dh.sit_head.eval(); dh.cmp_head.eval(); dh.reg_head.eval()
        if is_main: print("   ✅ 전체 eval 모드 (dropout off)")
        t = evaluate(dh, test_s, local_rank, world_size, args.num_frames, is_main, "test")
        if is_main and t:
            print(f"\n{'='*56}")
            print(f"🎯 [E2E {args.model} {args.num_frames}f] eval_only 결과")
            print(f"{'='*56}")
            print(f"  상황 {t['sit_acc']:.3f} | 비교 acc {t['cmp_acc']:.4f} / F1 {t['cmp_macro_f1']:.4f}")
            print(f"  비율 MAE {t['ratio_mae']:.2f} / within±10 {t['ratio_within10']:.4f}")
            print(f"\n  📊 비율 5구간(실무) 정확도: {t['ratio_band5_acc']:.4f}")
            print(f"     5구간 예측분포: {t['band5_pred_dist']}")
            print(f"     5구간 정답분포: {t['band5_gt_dist']}")
            print(f"     (0:A압도열세 1:A열세 2:대등 3:A우세 4:A압도우세)")
            out = dict(t); ev_preds = out.pop("preds", None)
            json.dump({"model": args.model, "num_frames": args.num_frames, "eval_only": True, "test": out},
                      open(os.path.join(RESULT_DIR, f"e2e_{args.model}_{args.num_frames}f{cls_tag}{split_tag}_eval.json"), "w"),
                      ensure_ascii=False, indent=2)
            print(f"\n💾 저장: e2e_{args.model}_{args.num_frames}f_eval.json")
            if ev_preds:
                import csv as _csv
                tag = f"{args.model}_{args.num_frames}f{cls_tag}{split_tag}_eval"
                with open(os.path.join(RESULT_DIR, f"preds_{tag}.csv"), "w", newline="", encoding="utf-8") as f:
                    w = _csv.writer(f)
                    w.writerow(["video", "sit_gt", "sit_pred", "cmp_gt", "cmp_pred",
                                "ratio_gt", "ratio_pred", "ratio_abs_err", "within10"])
                    for x in sorted(ev_preds, key=lambda r: r["idx"]):
                        vp = test_s[x["idx"]][0] if x["idx"] < len(test_s) else ""
                        err = abs(x["ra_pred"] - x["ra_gt"])
                        w.writerow([os.path.basename(vp), x["sit_gt"], x["sit_pred"],
                                    x["cmp_gt"], x["cmp_pred"], x["ra_gt"], x["ra_pred"],
                                    err, int(err <= 10)])
                print(f"💾 개별예측 저장: preds_{tag}.csv ({len(ev_preds)}건)")
        if dist.is_initialized():
            dist.barrier(); os._exit(0)
        return

    # zero-shot (학습 전) 평가
    if is_main: print("\n📊 [학습 전] zero-shot 평가...")
    z = evaluate(dh, test_s, local_rank, world_size, args.num_frames, is_main, "zeroshot")
    if is_main and z:
        print(f"   상황 acc {z['sit_acc']:.4f} | 비교 acc {z['cmp_acc']:.4f} / macro-F1 {z['cmp_macro_f1']:.4f} | "
              f"비율 MAE {z['ratio_mae']:.2f} / within10 {z['ratio_within10']:.4f}")

    # 학습 루프
    my_idx = list(range(local_rank, len(train_s), world_size))
    best_score, best_epoch, no_imp = -1, 0, 0
    os.makedirs(BEST, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        dh.train()
        if is_main: print(f"\n{'='*46}\n🚀 [Epoch {epoch}/{args.epochs}]\n{'='*46}")
        import random; random.Random(epoch).shuffle(my_idx)
        opt.zero_grad()
        running = 0.0
        n_ok, n_skip = 0, 0
        run_cmp_correct, run_sit_correct, run_ra_correct = 0, 0, 0
        pbar = tqdm(my_idx, desc=f"train GPU{local_rank} Ep{epoch}", disable=not is_main)
        for step, i in enumerate(pbar):
            vp, jp, cmp_gt, ra_gt, sit_gt = train_s[i]
            try:
                frames = extract_frames(vp, args.num_frames)
                sit_logits, cmp_logits, ratio_logits = dh(frames, PROMPT)
            except Exception as e:
                n_skip += 1
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                if is_main and n_skip <= 2: print(f"   ⚠️ forward 실패: {type(e).__name__}: {str(e)[:80]}")
                continue
            sit_t = torch.tensor([sit_gt], device=backbone.device)
            cmp_t = torch.tensor([cmp_gt], device=backbone.device)
            ra_cls_idx = ratio_to_class(ra_gt)
            ra_cls = torch.tensor([ra_cls_idx], device=backbone.device)  # 0~10
            loss_sit = F.cross_entropy(sit_logits.float(), sit_t)
            if args.cmp_loss == "focal":
                ce_n = F.cross_entropy(cmp_logits.float(), cmp_t, reduction="none")
                pt = torch.exp(-ce_n)
                loss_cmp = ((1 - pt) ** 2 * ce_n).mean()   # focal gamma=2
            else:
                loss_cmp = F.cross_entropy(cmp_logits.float(), cmp_t)   # 가중치 없음 (F1 안정)
            loss_reg = F.cross_entropy(ratio_logits.float(), ra_cls)   # 비율 11-class 분류
            # 3 loss 합 (상황은 보조라 가중 낮게)
            loss = (args.lambda_sit * loss_sit + args.lambda_cmp * loss_cmp + args.lambda_reg * loss_reg) / args.grad_accum
            loss.backward()
            running += loss.item() * args.grad_accum
            n_ok += 1
            # 실시간 정확도 (학습 진행 중 예측이 맞아가는지)
            with torch.no_grad():
                if int(sit_logits.argmax(-1).item()) == sit_gt: run_sit_correct += 1
                if int(cmp_logits.argmax(-1).item()) == cmp_gt: run_cmp_correct += 1
                if int(ratio_logits.argmax(-1).item()) == ra_cls_idx: run_ra_correct += 1
            if is_main and n_ok % 20 == 0:
                pbar.set_postfix({
                    "loss": f"{running/n_ok:.3f}",
                    "상황": f"{run_sit_correct/n_ok:.2f}",
                    "비교": f"{run_cmp_correct/n_ok:.2f}",
                    "비율": f"{run_ra_correct/n_ok:.2f}",
                })
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(); scheduler.step()
        # 남은 gradient 반영 (grad_accum 안 채운 마지막)
        if n_ok % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad()
        if is_main: print(f"   평균 loss: {running/max(n_ok,1):.4f} (ok={n_ok}, skip={n_skip})")

        # val 평가 → best 선택 (비교 macro-F1 + 비율 within10 조합)
        v = evaluate(dh, val_s, local_rank, world_size, args.num_frames, is_main, f"val{epoch}")
        stop = torch.tensor([0], device=backbone.device)
        if is_main and v:
            score = v["ratio_within10"] if args.select == "w10" else v["cmp_macro_f1"] + v["ratio_within10"]   # [ms] select 스위치
            print(f"   val: 상황 {v['sit_acc']:.3f} | 비교 acc {v['cmp_acc']:.4f}/F1 {v['cmp_macro_f1']:.4f} | "
                  f"비율 MAE {v['ratio_mae']:.2f}/w10 {v['ratio_within10']:.4f} | score {score:.4f}")
            if score > best_score:
                best_score, best_epoch, no_imp = score, epoch, 0
                torch.save({"sit": dh.sit_head.state_dict(), "cmp": dh.cmp_head.state_dict(),
                            "reg": dh.reg_head.state_dict()}, os.path.join(BEST, "heads.pt"))
                dh.backbone.model.save_pretrained(BEST)
                print(f"   ✅ best 갱신 (epoch {epoch}, score {score:.4f})")
            else:
                no_imp += 1
                print(f"   ⚠️ 하락 {no_imp}/{args.patience}")
                if no_imp >= args.patience: stop = torch.tensor([1], device=backbone.device)
        if dist.is_initialized(): dist.broadcast(stop, src=0)
        if stop.item() == 1:
            if is_main: print(f"⏹️ Early stopping (best epoch {best_epoch})")
            break

    # 최종 test — best heads 로드
    if is_main: print(f"\n🏁 [학습 후] test 최종 평가 (best epoch {best_epoch})")
    heads_path = os.path.join(BEST, "heads.pt")
    if os.path.exists(heads_path):
        try:
            ck = torch.load(heads_path, map_location=backbone.device)
            dh.sit_head.load_state_dict(ck["sit"]); dh.cmp_head.load_state_dict(ck["cmp"]); dh.reg_head.load_state_dict(ck["reg"])
            if is_main: print(f"   ✅ best heads 로드")
        except Exception as e:
            if is_main: print(f"   ⚠️ heads 로드 실패: {e}")
    if dist.is_initialized(): dist.barrier()
    t = evaluate(dh, test_s, local_rank, world_size, args.num_frames, is_main, "test")
    if is_main and t:
        print(f"\n{'='*56}")
        print(f"🎯 [E2E {args.model} {args.num_frames}f] 최종 결과")
        print(f"{'='*56}")
        print(f"  [학습 전] 상황 {z['sit_acc']:.3f} | 비교 acc {z['cmp_acc']:.4f} / F1 {z['cmp_macro_f1']:.4f} | MAE {z['ratio_mae']:.2f} / w10 {z['ratio_within10']:.4f} / 5구간 {z['ratio_band5_acc']:.4f}")
        print(f"  [학습 후] 상황 {t['sit_acc']:.3f} | 비교 acc {t['cmp_acc']:.4f} / F1 {t['cmp_macro_f1']:.4f} | MAE {t['ratio_mae']:.2f} / w10 {t['ratio_within10']:.4f} / 5구간 {t['ratio_band5_acc']:.4f}")
        print(f"\n  📊 비율 5구간(실무) 정확도: {t['ratio_band5_acc']:.4f}")
        print(f"     5구간 예측분포: {t['band5_pred_dist']}")
        print(f"     5구간 정답분포: {t['band5_gt_dist']}")
        print(f"     (0:A압도열세 1:A열세 2:대등 3:A우세 4:A압도우세)")
        print(f"\n  비교 예측분포: {t['cmp_pred_dist']}")
        print(f"  비교 정답분포: {t['cmp_gt_dist']}")
        if args.samples > 0:
            print("\n⚠️ smoke 모드(--samples) — 결과 파일 저장 안 함 (본 학습 결과 보호)")
        else:
            _do_save = True
        z_out = dict(z); z_preds = z_out.pop("preds", None)
        t_out = dict(t); t_preds = t_out.pop("preds", None)
        tag = f"{args.model}_{args.num_frames}f{cls_tag}"
        if args.samples <= 0: json.dump({"model": args.model, "num_frames": args.num_frames,
                   "config": {"epochs": args.epochs, "patience": args.patience, "lr": args.lr,
                              "lambda_sit": args.lambda_sit, "lambda_reg": args.lambda_reg,
                              "grad_accum": args.grad_accum, "seed": args.seed, "ratio_classes": args.ratio_classes},
                   "zeroshot": z_out, "finetuned": t_out, "best_epoch": best_epoch},
                  open(os.path.join(RESULT_DIR, f"e2e_{args.model}_{args.num_frames}f{cls_tag}.json"), "w"),
                  ensure_ascii=False, indent=2)
        if args.samples <= 0: print(f"\n💾 저장: e2e_{args.model}_{args.num_frames}f{cls_tag}.json")

        # 개별 영상별 예측 저장 (논문 분석/재현용)
        if t_preds and args.samples <= 0:
            import csv as _csv
            pred_csv = os.path.join(RESULT_DIR, f"preds_{tag}.csv")
            with open(pred_csv, "w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f)
                w.writerow(["video", "sit_gt", "sit_pred", "cmp_gt", "cmp_pred",
                            "ratio_gt", "ratio_pred", "ratio_abs_err", "within10"])
                for x in sorted(t_preds, key=lambda r: r["idx"]):
                    vp = test_s[x["idx"]][0] if x["idx"] < len(test_s) else ""
                    err = abs(x["ra_pred"] - x["ra_gt"])
                    w.writerow([os.path.basename(vp), x["sit_gt"], x["sit_pred"],
                                x["cmp_gt"], x["cmp_pred"], x["ra_gt"], x["ra_pred"],
                                err, int(err <= 10)])
            print(f"💾 개별예측 저장: preds_{tag}.csv ({len(t_preds)}건)")
            json.dump(t_preds, open(os.path.join(RESULT_DIR, f"preds_{tag}.json"), "w"),
                      ensure_ascii=False)

    if dist.is_initialized():
        dist.barrier(); os._exit(0)

if __name__ == "__main__":
    main()
