#!/bin/bash
# ══════════════════════════════════════════════════════════
# bb_1 18개 저장 best 전체 재평가 (깨끗한 eval 모드, 결정론적)
# → 논문의 공식 숫자 산출. 결과: e2e_*_eval.json / preds_*_eval.csv
# 사용: tmux new -s reeval; bash run_reeval_all.sh
# ══════════════════════════════════════════════════════════
set -e
cd ~/TS_code
source ~/venv_ts/bin/activate
export E2E_RESULT_DIR=result_e2e3_bb1
export LANG=C.UTF-8 LC_ALL=C.UTF-8 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
       OPENCV_FFMPEG_LOGLEVEL=-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FILTER='Warning|pad_token|deprecated|Fetching|Loading|Download|checkpoint|NCCL|Watchdog|using GPU|rank[0-9]'

reeval () {
  local M=$1; local NF=$2; local RC=$3
  local TAG="${M}_${NF}f"; [ "$RC" == "5" ] && TAG="${TAG}_5c"
  local OUT=~/TS_code/result_e2e3_bb1/e2e_${TAG}_eval.json
  if [ -e "$OUT" ] && [ "$FORCE" != "1" ]; then
    echo "⏭️  [$TAG] 재평가 있음 → 건너뜀"; return
  fi
  echo ""
  echo "═══ 🔁 재평가 [$TAG]  $(date '+%m-%d %H:%M') ═══"
  python -m torch.distributed.run --nproc_per_node=4 ~/TS_code/e2e_train3.py \
      --model $M --num_frames $NF --ratio_classes $RC --eval_only \
      --train_split split_train_v2_bb1.txt --val_split split_finetune_val_bb1.txt --test_split split_test_finetuning_bb1.txt \
      2>&1 | stdbuf -oL grep --line-buffered -v -E "$FILTER"
}

# 우선순위: 32f 11c (메인표) → 16f/8f 11c → 5c 전부
for M in internvl qwen qwen3; do reeval $M 32 11; done
for M in internvl qwen qwen3; do reeval $M 16 11; done
for M in internvl qwen qwen3; do reeval $M 8 11; done
for M in internvl qwen qwen3; do for NF in 32 16 8; do reeval $M $NF 5; done; done

echo ""
echo "═══ 📊 재평가 요약 ═══"
python3 - <<'PYEOF'
import json, glob, os
rows = []
for f in sorted(glob.glob(os.path.expanduser('~/TS_code/result_e2e3_bb1/e2e_*_eval.json'))):
    if 'zeroshot' in f: continue
    try:
        d = json.load(open(f)); t = d['test']
        name = os.path.basename(f).replace('e2e_','').replace('_eval.json','')
        rows.append((name, t['sit_acc'], t['cmp_acc'], t['cmp_macro_f1'],
                     t['ratio_mae'], t['ratio_within10'], t['ratio_band5_acc']))
    except Exception: pass
print(f"{'설정':18s} {'상황':>5s} {'비교acc':>7s} {'F1':>5s} {'MAE':>5s} {'w10':>5s} {'5구간':>5s}")
for n,s,ca,f1,m,w,b in rows:
    print(f"{n:18s} {s:5.3f} {ca:7.3f} {f1:5.3f} {m:5.1f} {w:5.3f} {b:5.3f}")
print(f"\n완료: {len(rows)}/18")
PYEOF
echo "🎉 재평가 전체 완료"
