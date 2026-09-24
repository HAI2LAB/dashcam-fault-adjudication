#!/bin/bash
# ══════════════════════════════════════════════════════════
# M3 seed 검증: 핵심 4설정 × seed {44,45} 학습+재평가
#   (seed 42는 본 실험 result_e2e3_bb1에 이미 존재)
# 설정: internvl_32f / qwen_16f / qwen3_16f / qwen3_32f (11c)
# 결과: result_e2e3_bb1_s44/, result_e2e3_bb1_s45/
# 사용: tmux new -s seeds; bash run_seeds.sh          (~4일)
#       bash run_seeds.sh agg                          (집계)
# ══════════════════════════════════════════════════════════
set -e
cd ~/TS_code
source ~/venv_ts/bin/activate
export LANG=C.UTF-8 LC_ALL=C.UTF-8 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
       OPENCV_FFMPEG_LOGLEVEL=-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SPLITS="--train_split split_train_v2_bb1.txt --val_split split_finetune_val_bb1.txt --test_split split_test_finetuning_bb1.txt"
FILTER='Warning|pad_token|deprecated|Fetching|Loading|Download|checkpoint|NCCL|Watchdog|using GPU|rank[0-9]'
CONFIGS="internvl:32 qwen:16 qwen3:16 qwen3:32"

run_seed () {
  local M=$1; local NF=$2; local SEED=$3
  local DIR=result_e2e3_bb1_s${SEED}
  local OUT=~/TS_code/${DIR}/e2e_${M}_${NF}f_test_finetuning_bb1_eval.json
  if [ -e "$OUT" ]; then echo "⏭️  [$M ${NF}f s$SEED] 완료됨 → 건너뜀"; return; fi
  echo ""
  echo "════ 🎲 [$M ${NF}f seed $SEED] 학습  $(date '+%m-%d %H:%M') ════"
  E2E_RESULT_DIR=$DIR python -m torch.distributed.run --nproc_per_node=4 e2e_train3.py \
      --model $M --num_frames $NF --ratio_classes 11 $SPLITS \
      --epochs 8 --patience 3 --lr 1e-4 --grad_accum 8 --seed $SEED \
      2>&1 | stdbuf -oL grep --line-buffered -v -E "$FILTER"
  echo "── eval-mode 재평가 ──"
  E2E_RESULT_DIR=$DIR python -m torch.distributed.run --nproc_per_node=4 e2e_train3.py \
      --model $M --num_frames $NF --ratio_classes 11 --eval_only $SPLITS \
      2>&1 | tail -8
}

aggregate () {
  python3 - <<'PYEOF'
import json, os, statistics
CFG = [('internvl',32),('qwen',16),('qwen3',16),('qwen3',32)]
DIRS = {42:'result_e2e3_bb1', 44:'result_e2e3_bb1_s44', 45:'result_e2e3_bb1_s45'}
print(f"{'설정':14s} {'지표':6s} {'s42':>6s} {'s44':>6s} {'s45':>6s} {'mean±std':>14s}")
print('-'*56)
for m, nf in CFG:
    vals = {}
    for s, d in DIRS.items():
        p = os.path.expanduser(f'~/TS_code/{d}/e2e_{m}_{nf}f_test_finetuning_bb1_eval.json')
        if os.path.exists(p):
            t = json.load(open(p)).get('test', json.load(open(p)).get('finetuned'))
            vals[s] = t
    for k, lbl in [('cmp_macro_f1','F1'), ('ratio_within10','w10'), ('sit_acc','sit')]:
        xs = [vals[s][k] for s in (42,44,45) if s in vals]
        cell = lambda s: f"{vals[s][k]:.3f}" if s in vals else '  -  '
        ms = f"{statistics.mean(xs):.3f}±{statistics.stdev(xs):.3f}" if len(xs)>=2 else '(대기)'
        print(f"{m+'_'+str(nf)+'f':14s} {lbl:6s} {cell(42):>6s} {cell(44):>6s} {cell(45):>6s} {ms:>14s}")
    print()
PYEOF
}

case "${1:-run}" in
  agg) aggregate ;;
  run)
    for SEED in 44 45; do
      for C in $CONFIGS; do
        M=${C%%:*}; NF=${C##*:}
        run_seed $M $NF $SEED
      done
    done
    aggregate
    echo "🎉 seed 캠페인 완료"
    ;;
esac
