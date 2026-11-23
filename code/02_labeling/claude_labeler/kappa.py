"""Agreement between pass-1 (claude) and pass-2 (codex) on the double-labeled images.
Matches people by box IoU>=0.3, then Cohen's kappa per field. usage: python3 kappa.py EXPORT.json"""
import json, sys, collections
d=json.load(open(sys.argv[1]))
by=collections.defaultdict(dict)
for t in d:
    for a in t['annotations']:
        people={}
        for r in a['result']:
            if r.get('from_name')=='box' and r['value']['rectanglelabels'][0]=='person': people.setdefault(r['id'],{})['box']=r['value']
            elif r.get('type')=='choices' and r.get('id') in people or (r.get('type')=='choices' and r.get('from_name') in ('support','helmet_state','harness_state','tied_off')):
                people.setdefault(r['id'],{})[r['from_name']]=r['value']['choices'][0]
        by[t['data']['global_id']][t['data'].get('pass')]=people
def iou(a,b):
    ax,ay,aw,ah=a['x'],a['y'],a['width'],a['height']; bx,by_,bw,bh=b['x'],b['y'],b['width'],b['height']
    ix=max(0,min(ax+aw,bx+bw)-max(ax,bx)); iy=max(0,min(ay+ah,by_+bh)-max(ay,by_)); i=ix*iy
    return i/(aw*ah+bw*bh-i+1e-9)
def kappa(pairs):
    n=len(pairs); 
    if not n: return None
    po=sum(a==b for a,b in pairs)/n; ca=collections.Counter(a for a,_ in pairs); cb=collections.Counter(b for _,b in pairs)
    pe=sum(ca[k]*cb[k] for k in set(ca)|set(cb))/n/n
    return round((po-pe)/(1-pe+1e-9),3), round(po,3), n
F=('support','helmet_state','harness_state','tied_off'); pairs={f:[] for f in F}; imgs=0; unmatched=0
for g,p in by.items():
    if 1 not in p or 2 not in p: continue
    imgs+=1; A=[v for v in p[1].values() if 'box' in v]; B=[v for v in p[2].values() if 'box' in v]; used=set()
    for a in A:
        best=max(((iou(a['box'],b['box']),j) for j,b in enumerate(B) if j not in used),default=(0,None))
        if best[0]<0.3: unmatched+=1; continue
        used.add(best[1]); b=B[best[1]]
        for f in F:
            if f in a and f in b: pairs[f].append((a[f],b[f]))
    unmatched+=len(B)-len(used)
print('images',imgs,'unmatched people',unmatched)
for f in F: print(f,'kappa/agree/n =',kappa(pairs[f]))
