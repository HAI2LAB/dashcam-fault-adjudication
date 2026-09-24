#!/usr/bin/env python3
"""추론 지연시간 실측 — 단일 영상, 단일 GPU, 단일 forward
사용: CUDA_VISIBLE_DEVICES=0 python3 latency_bench.py --model qwen3 --num_frames 32
     (기본: test 앞쪽 50개 영상, warmup 3개 제외)
출력: 디코딩/포워드 분리 시간 + 총 지연 (mean/median/p95)
"""
import sys, os, time, json, argparse, statistics
sys.path.insert(0, os.path.expanduser('~/TS_code'))
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3', choices=['internvl','qwen','qwen3'])
    ap.add_argument('--num_frames', type=int, default=32)
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--ratio_classes', type=int, default=11)
    args = ap.parse_args()

    import e2e_model3 as m3
    m3.set_ratio_classes(args.ratio_classes)
    from e2e_model3 import VLMBackbone, TripleHeadFaultModel, PROMPT, DATA_DIR
    from agent1_pipeline import extract_frames

    cls_tag = '' if args.ratio_classes == 11 else '_5c'
    BEST = os.path.expanduser(f'~/TS_code/result_e2e3/best_{args.model}_{args.num_frames}f{cls_tag}')
    assert os.path.isdir(BEST), f'best 폴더 없음: {BEST}'

    print(f'🔧 로드: {args.model} {args.num_frames}f  ({BEST})')
    backbone = VLMBackbone(args.model, 0)
    # LoRA 어댑터 (PEFT 표준 API — apply_lora 불필요)
    try:
        from peft import PeftModel
        if os.path.exists(os.path.join(BEST, 'adapter_config.json')):
            backbone.model = PeftModel.from_pretrained(backbone.model, BEST)
            backbone.model.eval()
            print('   ✅ LoRA 어댑터 로드')
    except Exception as e:
        print(f'   ⚠️ LoRA 로드 스킵: {str(e)[:80]}')
    dh = TripleHeadFaultModel(backbone)
    hp = os.path.join(BEST, 'heads.pt')
    if os.path.exists(hp):
        ck = torch.load(hp, map_location=backbone.device)
        dh.sit_head.load_state_dict(ck['sit']); dh.cmp_head.load_state_dict(ck['cmp']); dh.reg_head.load_state_dict(ck['reg'])
        print('   ✅ heads 로드')
    dh.eval()

    # test 영상 목록
    vids = []
    for line in open(os.path.join(DATA_DIR, 'split_test_finetuning.txt'), encoding='utf-8'):
        p = line.strip().split('\t')[0]
        if p: vids.append(p)
        if len(vids) >= args.n + args.warmup: break
    print(f'📊 대상 {len(vids)}개 (warmup {args.warmup} 제외 후 측정)')

    dec, fwd, tot = [], [], []
    with torch.no_grad():
        for i, vp in enumerate(vids):
            try:
                t0 = time.perf_counter()
                frames = extract_frames(vp, args.num_frames)
                t1 = time.perf_counter()
                _ = dh(frames, PROMPT)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t2 = time.perf_counter()
            except Exception as e:
                print(f'   skip {os.path.basename(vp)}: {str(e)[:50]}'); continue
            if i >= args.warmup:
                dec.append(t1 - t0); fwd.append(t2 - t1); tot.append(t2 - t0)

    def stat(x):
        s = sorted(x)
        return f'mean {statistics.mean(x):.2f}s | median {statistics.median(x):.2f}s | p95 {s[int(len(s)*0.95)-1]:.2f}s'
    print(f'\n═══ 지연시간 ({len(tot)}개 영상, batch=1, 1×GPU) ═══')
    print(f'  디코딩(±샘플링): {stat(dec)}')
    print(f'  모델 forward:    {stat(fwd)}')
    print(f'  합계(end-to-end): {stat(tot)}')
    print(f'  처리량: {1/statistics.mean(tot):.2f} videos/sec/GPU')

    out = {'model': args.model, 'num_frames': args.num_frames, 'n': len(tot),
           'decode_mean_s': statistics.mean(dec), 'forward_mean_s': statistics.mean(fwd),
           'total_mean_s': statistics.mean(tot), 'total_median_s': statistics.median(tot)}
    path = os.path.expanduser(f'~/TS_code/result_e2e3/latency_{args.model}_{args.num_frames}f.json')
    json.dump(out, open(path, 'w'), indent=2)
    print(f'💾 저장: {path}')

if __name__ == '__main__':
    main()
