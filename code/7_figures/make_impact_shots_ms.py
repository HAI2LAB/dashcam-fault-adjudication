#!/usr/bin/env python3
"""사고 시점 자동 감지 + 전후 타임스탬프 프레임 추출
원리: 클립이 사고시점 ±5초로 구축됨 → 중앙(2~8초) 구간에서
      연속 프레임 차이(카메라 충격 흔들림)가 최대인 지점 = 충돌 시점
출력: impact_shots_ms_0907/{영상이름}/ 에 t-1.5s ~ t+1.0s 6장 + 검출 시점 로그
사용: python3 make_impact_shots_ms_0907.py            # supp_video_list.txt의 경로 전부
      python3 make_impact_shots_ms_0907.py 영상경로   # 특정 영상 1개
"""
import sys, os, re
import cv2
import numpy as np

OFFSETS = [-1.5, -1.0, -0.5, 0.0, +0.5, +1.0]
SEARCH = (0.2, 0.8)          # 전체 길이 대비 탐색 구간 (중앙 60%)
OUT_ROOT = os.path.expanduser('~/TS_code/impact_shots_ms_0907')

def detect_impact(vp):
    """프레임 diff 스파이크로 충돌 시점(초) 추정. 실패 시 중앙값."""
    cap = cv2.VideoCapture(vp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps
    lo, hi = int(total*SEARCH[0]), int(total*SEARCH[1])
    prev, diffs = None, []
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    for i in range(lo, hi):
        ok, fr = cap.read()
        if not ok: break
        g = cv2.cvtColor(cv2.resize(fr, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.mean(np.abs(g - prev))))
        prev = g
    cap.release()
    if len(diffs) < 10:
        return dur/2, fps, dur, 'fallback(중앙)'
    d = np.array(diffs)
    # 이동평균 대비 급증 지점 (충격 = 짧고 강한 스파이크)
    k = max(3, int(fps*0.4))
    base = np.convolve(d, np.ones(k)/k, mode='same')
    spike = d - base
    idx = int(np.argmax(spike)) + lo + 1
    return idx/fps, fps, dur, f'spike(강도 {spike.max():.1f})'

def grab(vp, t_sec):
    cap = cv2.VideoCapture(vp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t_sec*fps))))
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None

def process(vp):
    name = os.path.basename(vp).replace('.mp4','')
    t0, fps, dur, how = detect_impact(vp)
    outdir = os.path.join(OUT_ROOT, name)
    os.makedirs(outdir, exist_ok=True)
    n_ok = 0
    for off in OFFSETS:
        t = min(max(t0 + off, 0), dur - 1/fps)
        fr = grab(vp, t)
        if fr is None: continue
        tag = f"t{'+' if off>=0 else ''}{off:.1f}s"
        cv2.imwrite(os.path.join(outdir, f"{tag}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n_ok += 1
    print(f"  ✅ {name}: 충돌 {t0:.2f}s ({how}) → {n_ok}장 저장")

def main():
    if len(sys.argv) > 1:
        vids = [sys.argv[1]]
    else:
        lst = os.path.expanduser('~/TS_code/supp_video_list.txt')
        txt = open(lst, encoding='utf-8').read()
        vids = re.findall(r'(/nas/\S+\.mp4)', txt)
    print(f"🎬 {len(vids)}개 영상 처리")
    for vp in vids:
        if not os.path.exists(vp):
            print(f"  ⚠️ 없음: {vp}"); continue
        try: process(vp)
        except Exception as e: print(f"  ❌ {os.path.basename(vp)}: {str(e)[:60]}")
    print(f"\n📁 결과: {OUT_ROOT}/<영상이름>/t-1.5s.jpg ... t+1.0s.jpg")
    print("   검출이 이상한 영상은 개별 재실행 또는 수동 캡처로 보완")

if __name__ == '__main__':
    main()
