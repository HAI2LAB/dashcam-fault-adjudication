#!/usr/bin/env python3
"""M2 외부 baseline: VideoMAE-base + 삼중 헤드 (soft chain 동일 구조)
- 같은 bb_1 스플릿/라벨/선택기준/지표 — "VLM 백본의 가치" 입증용
- 단일 GPU: CUDA_VISIBLE_DEVICES=0 python3 videomae_baseline.py
- 결과: result_e2e3_bb1/e2e_videomae_16f.json + preds_videomae_16f_eval.csv
"""
import os, sys, json, csv, time, random, argparse
sys.path.insert(0, os.path.expanduser('~/TS_code'))
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

from e2e_model3 import DATA_DIR, get_fault_gt, get_situation_gt
from agent1_pipeline import extract_frames

RESULT_DIR = os.path.expanduser('~/TS_code/result_e2e3_bb1')
os.makedirs(RESULT_DIR, exist_ok=True)
N_SIT, N_CMP, N_RA = 13, 3, 11

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_pairs(name):
    p1 = os.path.join(DATA_DIR, name); p2 = os.path.expanduser(f'~/TS_code/{name}')
    p = p1 if os.path.exists(p1) else p2
    out = []
    for line in open(p, encoding='utf-8'):
        parts = line.strip().split('\t')
        if len(parts) >= 2: out.append((parts[0], parts[1]))
    return out

def labels_of(jp):
    gt = get_fault_gt(jp)
    if gt is None: return None
    ra, rb = gt
    sit = get_situation_gt(jp)
    if sit is None: return None
    cmp_l = 1 if abs(ra - rb) <= 1 else (2 if ra > rb else 0)
    ra_cls = min(10, int(round(ra / 10)))
    return sit, cmp_l, ra_cls, ra

class VidDS(Dataset):
    def __init__(self, pairs, nframes, processor):
        self.items = []
        for vp, jp in pairs:
            lab = labels_of(jp)
            if lab is not None: self.items.append((vp,) + lab)
        self.nf = nframes; self.proc = processor
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        vp, sit, cmp_l, ra_cls, ra = self.items[i]
        frames = extract_frames(vp, self.nf)
        # PIL/np 모두 수용 → np.uint8 HWC 리스트로 통일
        arrs = []
        for fr in frames:
            a = np.array(fr)
            if a.ndim == 2: a = np.stack([a]*3, -1)
            arrs.append(a[..., :3].astype(np.uint8))
        while len(arrs) < self.nf: arrs.append(arrs[-1])
        px = self.proc(arrs, return_tensors='pt')['pixel_values'][0]  # (T,3,224,224)
        return px, sit, cmp_l, ra_cls, ra, os.path.basename(vp)

def collate(b):
    px = torch.stack([x[0] for x in b])
    t = lambda k: torch.tensor([x[k] for x in b])
    return px, t(1), t(2), t(3), t(4), [x[5] for x in b]

class TripleHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        def mlp(i, o): return nn.Sequential(nn.Linear(i, d//2), nn.GELU(), nn.Dropout(0.3), nn.Linear(d//2, o))
        self.sit = mlp(d, N_SIT)
        self.cmp = mlp(d + N_SIT, N_CMP)
        self.reg = mlp(d + N_SIT + N_CMP, N_RA)
    def forward(self, h):
        ls = self.sit(h); ps = ls.softmax(-1)
        lc = self.cmp(torch.cat([h, ps], -1)); pc = lc.softmax(-1)
        lr = self.reg(torch.cat([h, ps, pc], -1))
        return ls, lc, lr

def evaluate(bb, heads, loader, dev, save_preds=None):
    bb.eval(); heads.eval()
    agg = defaultdict(float); n = 0
    cls_stat = defaultdict(lambda: [0,0,0])  # cmp per-class: tp, gt_n, pred_n
    pb = defaultdict(lambda: [0,0])
    rows = []
    with torch.no_grad():
        for px, sit, cmp_t, ra_cls, ra, names in loader:
            px = px.to(dev, dtype=torch.bfloat16)
            h = bb(pixel_values=px).last_hidden_state.mean(1).float()
            ls, lc, lr = heads(h)
            sp, cp = ls.argmax(-1).cpu(), lc.argmax(-1).cpu()
            rp_cls = lr.argmax(-1).cpu(); rp = rp_cls * 10
            for i in range(len(names)):
                n += 1
                agg['sit'] += int(sp[i]==sit[i]); agg['cmp'] += int(cp[i]==cmp_t[i])
                err = abs(int(rp[i]) - int(ra[i]))
                agg['mae'] += err; agg['w10'] += int(err <= 10)
                b_g = 0 if ra[i]<=10 else 1 if ra[i]<=30 else 2 if ra[i]<=60 else 3 if ra[i]<=80 else 4
                b_p = 0 if rp[i]<=10 else 1 if rp[i]<=30 else 2 if rp[i]<=60 else 3 if rp[i]<=80 else 4
                agg['b5'] += int(b_g==b_p); pb[b_g][1]+=1; pb[b_g][0]+=int(b_g==b_p)
                c_g, c_p = int(cmp_t[i]), int(cp[i])
                cls_stat[c_g][1]+=1; cls_stat[c_p][2]+=1
                if c_g==c_p: cls_stat[c_g][0]+=1
                if save_preds is not None:
                    rows.append([names[i], int(sit[i]), int(sp[i]), c_g, c_p, int(ra[i]), int(rp[i]), err, int(err<=10)])
    f1s = []
    for c in range(3):
        tp, gn, pn = cls_stat[c]
        p = tp/pn if pn else 0; r = tp/gn if gn else 0
        f1s.append(2*p*r/(p+r) if p+r else 0)
    res = {'n': n, 'sit_acc': agg['sit']/n, 'cmp_acc': agg['cmp']/n,
           'cmp_macro_f1': sum(f1s)/3, 'ratio_mae': agg['mae']/n,
           'ratio_within10': agg['w10']/n, 'ratio_band5_acc': agg['b5']/n,
           'band5_macro_recall': sum(c/t for c,t in pb.values())/len(pb)}
    if save_preds is not None:
        with open(save_preds, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['video','sit_gt','sit_pred','cmp_gt','cmp_pred','ratio_gt','ratio_pred','ratio_abs_err','within10'])
            w.writerows(rows)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_frames', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    seed_all(args.seed)
    dev = 'cuda'

    from transformers import VideoMAEModel, VideoMAEImageProcessor
    proc = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base')
    bb = VideoMAEModel.from_pretrained('MCG-NJU/videomae-base', torch_dtype=torch.bfloat16).to(dev)
    heads = TripleHead(bb.config.hidden_size).to(dev)
    print(f'🔧 VideoMAE-base + 삼중헤드 | frames {args.num_frames} | params {sum(p.numel() for p in bb.parameters())/1e6:.0f}M')

    tr = VidDS(load_pairs('split_train_v2_bb1.txt'), args.num_frames, proc)
    va = VidDS(load_pairs('split_finetune_val_bb1.txt'), args.num_frames, proc)
    te = VidDS(load_pairs('split_test_finetuning_bb1.txt'), args.num_frames, proc)
    print(f'📊 train {len(tr)} / val {len(va)} / test {len(te)}')
    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch, shuffle=sh, num_workers=8,
                                   collate_fn=collate, pin_memory=True, drop_last=sh)
    tr_l, va_l, te_l = dl(tr, True), dl(va, False), dl(te, False)

    opt = torch.optim.AdamW(list(bb.parameters()) + list(heads.parameters()), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(tr_l), eta_min=args.lr * 0.1)

    best_score, best_ep, bad = -1, 0, 0
    BEST = os.path.join(RESULT_DIR, 'best_videomae_16f.pt')
    for ep in range(1, args.epochs + 1):
        bb.train(); heads.train()
        t0 = time.time(); tot = 0
        for step, (px, sit, cmp_t, ra_cls, ra, _) in enumerate(tr_l):
            px = px.to(dev, dtype=torch.bfloat16)
            h = bb(pixel_values=px).last_hidden_state.mean(1).float()
            ls, lc, lr_ = heads(h)
            loss = 0.2 * F.cross_entropy(ls, sit.to(dev)) + F.cross_entropy(lc, cmp_t.to(dev)) \
                 + F.cross_entropy(lr_, ra_cls.to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(bb.parameters()) + list(heads.parameters()), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item()
            if step % 100 == 0:
                print(f'  Ep{ep} {step}/{len(tr_l)} loss {tot/(step+1):.3f} ({(time.time()-t0)/60:.0f}m)', flush=True)
        v = evaluate(bb, heads, va_l, dev)
        score = v['cmp_macro_f1'] + v['ratio_within10']
        print(f'Ep{ep} val: sit {v["sit_acc"]:.3f} F1 {v["cmp_macro_f1"]:.3f} w10 {v["ratio_within10"]:.3f} | score {score:.3f}')
        if score > best_score:
            best_score, best_ep, bad = score, ep, 0
            torch.save({'bb': bb.state_dict(), 'heads': heads.state_dict()}, BEST)
            print('  💾 best 갱신')
        else:
            bad += 1
            if bad >= args.patience: print('  ⏹ early stop'); break

    ck = torch.load(BEST, map_location=dev)
    bb.load_state_dict(ck['bb']); heads.load_state_dict(ck['heads'])
    t = evaluate(bb, heads, te_l, dev,
                 save_preds=os.path.join(RESULT_DIR, 'preds_videomae_16f_eval.csv'))
    print('=' * 50)
    print(f'🎯 [VideoMAE-base 16f, best ep {best_ep}] test:')
    print(f'  상황 {t["sit_acc"]:.3f} | 비교 {t["cmp_acc"]:.3f}/F1 {t["cmp_macro_f1"]:.3f}')
    print(f'  MAE {t["ratio_mae"]:.1f} | w10 {t["ratio_within10"]:.3f} | 5구간 {t["ratio_band5_acc"]:.3f} | macroB5 {t["band5_macro_recall"]:.3f}')
    json.dump({'model': 'videomae', 'num_frames': args.num_frames,
               'config': {'ratio_classes': 11, 'seed': args.seed}, 'best_epoch': best_ep,
               'test': t, 'finetuned': t},
              open(os.path.join(RESULT_DIR, 'e2e_videomae_16f.json'), 'w'), indent=2)
    print('💾 저장 완료')

if __name__ == '__main__':
    main()
