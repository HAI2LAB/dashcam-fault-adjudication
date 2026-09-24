#!/bin/bash
# ratio-only baseline: single ratio head supervision (E2E_CHAIN=none, lambda_sit=0, lambda_cmp=0, select=w10). seeds 42 -> 44 -> 45, sequential.
cd ~/TS_code
source ~/venv_ts/bin/activate
export LANG=C.UTF-8 LC_ALL=C.UTF-8 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
       OPENCV_FFMPEG_LOGLEVEL=-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export E2E_CHAIN=none

SPLITS="--train_split split_train_v2_bb1.txt --val_split split_finetune_val_bb1.txt --test_split split_test_finetuning_bb1.txt"
FILTER='Warning|pad_token|deprecated|Fetching|Loading|Download|checkpoint|NCCL|Watchdog|using GPU|rank[0-9]'
M=internvl; NF=32

run_seed () {
  local SEED=$1
  local DIR=result_e2e3_bb1_ratioonly_s${SEED}
  local OUT=~/TS_code/${DIR}/e2e_${M}_${NF}f_test_finetuning_bb1_eval.json
  if [ -e "$OUT" ]; then echo "⏭️   [ratioonly s$SEED] 완료됨 → 건너뜀"; return 0; fi
  echo ""
  echo "════ 🎲 [ratioonly $M ${NF}f seed $SEED] 학습  $(date '+%m-%d %H:%M') ════"
  E2E_RESULT_DIR=$DIR python -m torch.distributed.run --nproc_per_node=4 e2e_train3_ms.py \
      --model $M --num_frames $NF --ratio_classes 11 $SPLITS \
      --epochs 8 --patience 3 --lr 1e-4 --grad_accum 8 --seed $SEED --lambda_sit 0 --lambda_cmp 0 --select w10 \
      2>&1 | stdbuf -oL grep --line-buffered -v -E "$FILTER"
  echo "── eval-mode 재평가 ──"
  E2E_RESULT_DIR=$DIR python -m torch.distributed.run --nproc_per_node=4 e2e_train3_ms.py \
      --model $M --num_frames $NF --ratio_classes 11 --eval_only $SPLITS \
      2>&1 | tail -8
  if [ -e "$OUT" ]; then echo "✅ [ratioonly s$SEED] 완료 $(date '+%m-%d %H:%M')"; else echo "❌ [ratioonly s$SEED] 결과 JSON 없음 — 실패, 다음 시드 진행" | tee -a ratioonly_failures_ms.log; fi
}

aggregate () {
  python3 - <<'PYEOF'
import json, os, statistics
DIRS = {'soft':   {42:'result_e2e3_bb1',      44:'result_e2e3_bb1_s44',      45:'result_e2e3_bb1_s45'},
        'hard':   {42:'result_e2e3_bb1_hard', 44:'result_e2e3_bb1_hard_s44', 45:'result_e2e3_bb1_hard_s45'},
        'nochain':{42:'result_e2e3_bb1_nochain_s42', 44:'result_e2e3_bb1_nochain_s44', 45:'result_e2e3_bb1_nochain_s45'},
        'ratioonly':{42:'result_e2e3_bb1_ratioonly_s42', 44:'result_e2e3_bb1_ratioonly_s44', 45:'result_e2e3_bb1_ratioonly_s45'}}
KEYS = [('cmp_acc','CmpAcc'),('cmp_macro_f1','F1'),('ratio_within10','W10'),('ratio_mae','MAE'),('sit_acc','Sit')]
print(f"{'variant':8s} {'metric':7s} {'s42':>7s} {'s44':>7s} {'s45':>7s} {'mean±std':>15s}")
print('-'*50)
for v, dd in DIRS.items():
    vals = {}
    for s, d in dd.items():
        p = os.path.expanduser(f'~/TS_code/{d}/e2e_internvl_32f_test_finetuning_bb1_eval.json')
        if os.path.exists(p):
            j = json.load(open(p)); vals[s] = j.get('test', j.get('finetuned', j))
    for k, lbl in KEYS:
        xs = [vals[s][k] for s in (42,44,45) if s in vals and k in vals[s]]
        cell = lambda s: f"{vals[s][k]:.3f}" if s in vals and k in vals[s] else '   -   '
        ms = f"{statistics.mean(xs):.3f}±{statistics.stdev(xs):.3f}" if len(xs)>=2 else ('(1 seed)' if len(xs)==1 else '(none)')
        print(f"{v:8s} {lbl:7s} {cell(42):>7s} {cell(44):>7s} {cell(45):>7s} {ms:>15s}")
    print()
PYEOF
}

case "${1:-run}" in
  agg) aggregate ;;
  run)
    for SEED in 42 44 45; do run_seed $SEED; done
    aggregate
    echo "🎉 no-chain 3-seed 캠페인 완료 $(date '+%m-%d %H:%M')"
    ;;
esac
