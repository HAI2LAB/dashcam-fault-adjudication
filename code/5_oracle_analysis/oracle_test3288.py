#!/usr/bin/env python3
"""Oracle 상한 재계산 @ split_test_finetuning (3,288)
- Oracle-13:  정답 13-class 상황 → train에서 학습한 최적 비율
- Oracle-fine: 정답 A_Direction 원문(세분) → train 최적 비율
→ "상황 인식이 완벽해도 비율 예측의 천장" 을 우리 test셋 위에서 직접 측정
CPU만 사용. 학습과 병행 실행 무해.
"""
import sys, os, json
from collections import defaultdict, Counter
sys.path.insert(0, os.path.expanduser('~/TS_code'))
from e2e_model3 import get_fault_gt, get_situation_gt, CSV_MAP, DATA_DIR, _get_field

def load_pairs(name):
    p = os.path.join(DATA_DIR, name)
    if not os.path.exists(p):
        p = os.path.expanduser(os.path.join('~/TS_code', name))
    out = []
    for line in open(p, encoding='utf-8'):
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            out.append((parts[0], parts[1]))
    return out

def find_accident_type(obj):
    """e2e_model3와 동일한 필드 추출 (traffic_accident_type 우선, j['video'] 포함)"""
    at = _get_field(obj, ["traffic_accident_type", "accident_type"])
    try:
        return int(at) if at is not None else None
    except Exception:
        return None

def get_keys(jp):
    """(13class, fine A_Direction 문자열) 반환"""
    sit13 = get_situation_gt(jp)
    fine = None
    try:
        at = find_accident_type(json.load(open(jp, encoding='utf-8')))
        if at is not None:
            fine = CSV_MAP.get(int(at), {}).get('A_Direction', '') or None
    except Exception:
        pass
    return sit13, fine

def band5(r):
    return 0 if r <= 10 else 1 if r <= 30 else 2 if r <= 60 else 3 if r <= 80 else 4

def best_value_for(ras):
    """관측 비율 목록에서 within±10을 최대화하는 예측값 (동률 시 MAE 최소)"""
    cands = sorted(set(list(range(0, 101, 10)) + list(ras)))
    best, bw, bm = 50, -1, 1e9
    for c in cands:
        w = sum(1 for r in ras if abs(r - c) <= 10)
        m = sum(abs(r - c) for r in ras)
        if w > bw or (w == bw and m < bm):
            best, bw, bm = c, w, m
    return best

def evaluate(pred_map, fallback, samples, keyfn, label):
    n = len(samples); w10 = 0; mae = 0
    pb = defaultdict(lambda: [0, 0])
    miss = 0
    for ra, key in samples:
        v = pred_map.get(key)
        if v is None:
            v = fallback; miss += 1
        w10 += (abs(ra - v) <= 10)
        mae += abs(ra - v)
        bg, bp = band5(ra), band5(v)
        pb[bg][1] += 1; pb[bg][0] += (bg == bp)
    macro = sum(c / t for c, t in pb.values()) / len(pb)
    micro = sum(c for c, _ in pb.values()) / n
    print(f"[{label}] within±10 {w10/n:.3f} | MAE {mae/n:.1f} | band5 micro {micro:.3f} / macro {macro:.3f} | 미등록키 {miss}건")
    return {'w10': w10/n, 'mae': mae/n, 'band5_micro': micro, 'band5_macro': macro, 'unseen': miss}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_split', default='split_train_v2_bb1.txt')
    ap.add_argument('--test_split', default='split_test_finetuning_bb1.txt')
    ap.add_argument('--out', default='~/TS_code/result_e2e3_bb1/oracle_bb1.json')
    args = ap.parse_args()
    print(f"📥 train 라벨 수집 ({args.train_split})...")
    train = load_pairs(args.train_split)
    by13, byfine = defaultdict(list), defaultdict(list)
    all_ras = []
    for vp, jp in train:
        gt = get_fault_gt(jp)
        if gt is None: continue
        ra = gt[0]
        s13, fine = get_keys(jp)
        all_ras.append(ra)
        if s13 is not None: by13[s13].append(ra)
        if fine: byfine[fine].append(ra)
    print(f"   train 유효 {len(all_ras)} | 13class 키 {len(by13)} | fine 키 {len(byfine)}")

    map13 = {k: best_value_for(v) for k, v in by13.items()}
    mapfine = {k: best_value_for(v) for k, v in byfine.items()}
    fallback = best_value_for(all_ras)
    print(f"   전역 fallback 예측값: {fallback}%")

    print(f"\n📥 test 로드 ({args.test_split})...")
    test = load_pairs(args.test_split)
    t13, tfine = [], []
    for vp, jp in test:
        gt = get_fault_gt(jp)
        if gt is None: continue
        ra = gt[0]
        s13, fine = get_keys(jp)
        t13.append((ra, s13)); tfine.append((ra, fine))
    print(f"   test 유효 {len(t13)}")

    print("\n═══ Oracle 결과 (test 3,288) ═══")
    r_const = evaluate({}, fallback, t13, None, f"상수 {fallback}% (참고)")
    r13 = evaluate(map13, fallback, t13, None, "Oracle-13 (13class 완벽)")
    rfine = evaluate(mapfine, fallback, tfine, None, "Oracle-fine (A방향 원문 완벽)")

    out = {'const': r_const, 'oracle_13': r13, 'oracle_fine': rfine,
           'fallback_value': fallback, 'n_test': len(t13),
           'n_keys_13': len(map13), 'n_keys_fine': len(mapfine)}
    path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n💾 저장: {path}")

if __name__ == '__main__':
    main()
