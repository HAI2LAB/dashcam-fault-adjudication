#!/usr/bin/env python3
"""B: 서론 '2%-9%' 산출 방식 재현 (S-VIII 근거)
당시 zero-shot 결과(backup/agent1/result/*.json, n=3281)에서
Vehicle A Movement를 13클래스로 매핑해 exact-match / 13-class acc / macro recall 산출.
결론: exact-match ~2% (하한), 13-class macro recall 7.7~8.4% (상한) = 본문 '2%-9%'."""
import json, glob, re
from collections import defaultdict, Counter
RULES = [("u-turn","U-turn"),("roundabout","Roundabout"),("revers","Reverse"),
 ("overtak","Overtaking"),("lane chang","Lane change/Merge"),("merge","Lane change/Merge"),
 ("centerline","Centerline violation"),("center line","Centerline violation"),
 ("stop","Stop/Park"),("park","Stop/Park"),
 ("pedestrian","Pedestrian/Crossing"),("crossing","Pedestrian/Crossing"),("crosswalk","Pedestrian/Crossing"),
 ("rear-end","Rear-end/Collision"),("rear end","Rear-end/Collision"),("collision","Rear-end/Collision"),("preceding","Rear-end/Collision"),
 ("left turn","Left turn"),("right turn","Right turn"),
 ("go straight","Go straight"),("straight","Go straight")]
def move_line(t):
    m = re.search(r"Vehicle A Movement:\s*(.*)", t or "")
    s=(m.group(1) if m else "").strip().lower()
    return re.sub(r"\s+"," ",re.sub(r"[\[\(](facing|leading)[\]\)]","",s)).strip()
def to13(t):
    s = move_line(t)
    for k,c in RULES:
        if k in s: return c
    return "Other" if s else None
for p in sorted(glob.glob("backup/agent1/result/zeroshot_results_agent1_*.json")):
    d=json.load(open(p)); det=d["details"]; name=p.split("_")[-1].replace(".json","")
    em_t=em_h=0; c13_t=c13_h=0; hit=defaultdict(int); tot=defaultdict(int); dist=Counter()
    for x in det:
        gm,pm = move_line(x["ground_truth"]), move_line(x["model_prediction"])
        if gm: em_t+=1; em_h+=(gm==pm and gm!="")
        g,pr = to13(x["ground_truth"]), to13(x["model_prediction"])
        if g is None: continue
        c13_t+=1; c13_h+=(g==pr); tot[g]+=1; hit[g]+=(g==pr); dist[pr]+=1
    mr = sum(hit[c]/tot[c] for c in tot)/len(tot)
    print(f"{name:9s} n={len(det)} | exact-match={em_h/max(em_t,1):.3f} | 13cls-acc={c13_h/max(c13_t,1):.3f} | macro-recall={mr:.3f} | top-pred={dist.most_common(2)}")
