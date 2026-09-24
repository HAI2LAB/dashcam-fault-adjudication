#!/bin/bash
# ══════════════════════════════════════════════════════════
# bb_1(1인칭 블랙박스) 전용 — 본 실험 18런 전체 파이프라인
# 기존 결과(result_e2e3/)와 완전 분리: result_e2e3_bb1/ 에 저장
# 사용:
#   bash run_paper_bb1.sh splits    # bb_1 스플릿 파일 생성 (1회)
#   tmux new -s bb1; bash run_paper_bb1.sh priority
#   bash run_paper_bb1.sh summary
# ══════════════════════════════════════════════════════════
set -e
cd ~/TS_code
source ~/venv_ts/bin/activate
export LANG=C.UTF-8 LC_ALL=C.UTF-8 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
       OPENCV_FFMPEG_LOGLEVEL=-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export E2E_RESULT_DIR=result_e2e3_bb1

TRAIN_SPLIT=split_train_v2_bb1.txt
VAL_SPLIT=split_finetune_val_bb1.txt
TEST_SPLIT=split_test_finetuning_bb1.txt
FILTER='Warning|pad_token|deprecated|Fetching|Loading|Download|checkpoint|NCCL|Watchdog|using GPU|rank[0-9]'

make_splits () {
  # 원본은 NAS(DATA_DIR)에서 읽고, bb1 필터 결과는 ~/TS_code/ 에 저장
  # (NAS 반영은 사용자가 직접 업로드 — pick()이 TS_code도 탐색하므로 미업로드여도 동작)
  python3 - <<'PYSPLIT'
import os
from e2e_model3 import DATA_DIR
out_dir = os.path.expanduser('~/TS_code')
for s in ['split_train_v2','split_finetune_val','split_test_finetuning','split_zeroshot','split_test_overall']:
    src = os.path.join(DATA_DIR, s + '.txt')
    dst = os.path.join(out_dir, s + '_bb1.txt')
    lines = [l for l in open(src, encoding='utf-8') if '/bb_1_' in l]
    open(dst, 'w', encoding='utf-8').writelines(lines)
    print('  ' + s + '_bb1.txt: ' + str(len(lines)) + '건 → ~/TS_code/')
PYSPLIT
  echo "✅ bb_1 스플릿 5종 생성 (~/TS_code/) — 필요 시 NAS로 직접 업로드"
}

run_one () {
  local M=$1; local NF=$2; local RC=$3
  local TAG="${M}_${NF}f"; [ "$RC" == "5" ] && TAG="${TAG}_5c"
  local OUT=~/TS_code/result_e2e3_bb1/e2e_${TAG}.json
  if [ -e "$OUT" ]; then echo "⏭️  [$TAG] 결과 있음 → 건너뜀"; return; fi
  echo ""
  echo "════════════════════════════════════════════════"
  echo "🚀 [bb1 | $TAG] 학습  $(date '+%m-%d %H:%M')"
  echo "════════════════════════════════════════════════"
  python -m torch.distributed.run --nproc_per_node=4 ~/TS_code/e2e_train3.py \
      --model $M --num_frames $NF --ratio_classes $RC \
      --train_split $TRAIN_SPLIT --val_split $VAL_SPLIT --test_split $TEST_SPLIT \
      --epochs 8 --patience 3 --lr 1e-4 --grad_accum 8 --seed 42 \
      2>&1 | stdbuf -oL grep --line-buffered -v -E "$FILTER"
  echo "✅ [$TAG] 완료  $(date '+%m-%d %H:%M')"
}

summary () {
  echo "════════════════════════════════════════════════════════════"
  echo "📊 bb_1 전용 결과 요약  ($(date '+%m-%d %H:%M'))"
  echo "════════════════════════════════════════════════════════════"
  python3 - <<'PYEOF'
import json, glob, os
rows = []
for f in sorted(glob.glob(os.path.expanduser('~/TS_code/result_e2e3_bb1/e2e_*.json'))):
    if '_eval' in f: continue
    try:
        d = json.load(open(f)); t = d['finetuned']
        rc = d['config'].get('ratio_classes', 11)
        rows.append((d['model'], d['num_frames'], rc, t['sit_acc'], t['cmp_acc'],
                     t['cmp_macro_f1'], t['ratio_mae'], t['ratio_within10'], t['ratio_band5_acc']))
    except Exception: pass
rows.sort(key=lambda r: (r[2], r[0], r[1]))
print(f"{'모델':10s} {'f':>3s} {'cls':>3s} | {'상황':>6s} {'비교acc':>7s} {'비교F1':>6s} | {'MAE':>5s} {'w10':>5s} {'5구간':>5s}")
print('-'*70)
for m,nf,rc,s,ca,f1,mae,w,b in rows:
    print(f"{m:10s} {nf:>3d} {rc:>3d} | {s:6.3f} {ca:7.3f} {f1:6.3f} | {mae:5.1f} {w:5.3f} {b:5.3f}")
print(f"\n진행: {len(rows)}/18 실험 완료")
PYEOF
}

case "${1:-priority}" in
  splits) make_splits ;;
  summary) summary ;;
  priority)
    # 1단계: 32f 11c (메인) → 2단계: 16f/8f 11c → 3단계: 5c 전부
    for M in internvl qwen qwen3; do run_one $M 32 11; done
    for M in internvl qwen qwen3; do run_one $M 16 11; done
    for M in internvl qwen qwen3; do run_one $M 8 11; done
    for M in internvl qwen qwen3; do for NF in 32 16 8; do run_one $M $NF 5; done; done
    summary
    echo "🎉 bb_1 전체 18개 실험 완료"
    ;;
esac
