#!/usr/bin/env python3
"""E: 라벨 폐기 수 카운트 (논문 III-C 'excludes 8 of 7,574' 근거)
학습 코드(e2e_model3)의 get_fault_gt와 동일 로직으로 4개 스플릿 검산.
교차검증: test ok=2837(본문 일치), zeroshot ok=2840(본문 일치) 확인됨."""
from e2e_model3 import load_split, get_fault_gt
for name in ["split_train_v2_bb1.txt","split_finetune_val_bb1.txt",
             "split_test_finetuning_bb1.txt","split_zeroshot_bb1.txt"]:
    pairs = load_split(name); n=len(pairs); bad=0
    for vp, jp in pairs:
        if get_fault_gt(jp) is None: bad+=1
    print(f"{name}: pairs={n}, discarded={bad}, ok={n-bad}")
