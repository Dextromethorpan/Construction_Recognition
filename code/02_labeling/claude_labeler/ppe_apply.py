"""Apply PPE codes for one sheet. usage: ppe_apply.py N "codes"
codes: 20 space-separated tokens, one per tile, 4 chars each: helmet harness tied support
 helmet W=worn N=not_worn H=held V=not_visible ; harness Y=worn N=not_worn V=not_visible ;
 tied Y=yes N=no V=not_visible ; support S=stands_on C=climbing N=none U=unclear ; '.' keeps current value.
 A tile token 'X' means: not a person (box removed)."""
import json, os, sys
H=os.path.expanduser('~/GenIA-data'); OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'_render')
SP=f'{H}/labels/relations/v1/claude_specs'
n=int(sys.argv[1]); codes=sys.argv[2].split(); idx=json.load(open(f'{OUT}/ppe_index.json'))[n*20:(n+1)*20]
assert len(codes)==len(idx),(len(codes),len(idx))
HM=dict(W='helmet:worn',N='helmet:not_worn',H='helmet:held',V='helmet:not_visible')
HA=dict(Y='harness:worn',N='harness:not_worn',V='harness:not_visible')
TI=dict(Y='tied:yes',N='tied:no',V='tied:not_visible')
SU=dict(S='stands_on',C='climbing',N='none',U='unclear')
touched=set(); log=[]
for p,c in zip(idx,codes):
    f=f"{SP}/{p['task']:04d}.json"; s=json.load(open(f))
    for b in list(s['boxes']):
        if b['id']!=p['id']: continue
        if c=='X': s['boxes'].remove(b); log.append((p['task'],p['id'],'removed')); break
        h,a,t,u=c
        new=dict(b)
        if h!='.': new['helmet_state']=HM[h]
        if a!='.': new['harness_state']=HA[a]
        if t!='.': new['tied_off']=TI[t]
        if a=='N' and t=='.': new['tied_off']='tied:no'
        if u!='.':
            new['support']=SU[u]
            if SU[u] in ('none','unclear'): new.pop('on',None)
            if SU[u]=='none' and new.get('height_band') in (None,'h:unknown'): new['height_band']='h:ground'
        if new!=b: log.append((p['task'],p['id'],{k:new.get(k) for k in ('helmet_state','harness_state','tied_off','support') if new.get(k)!=b.get(k)}))
        b.clear(); b.update(new)
    s['replace']=True; s['ppe_checked']=True
    json.dump(s,open(f,'w'),indent=1); touched.add(f)
open(f'{OUT}/ppe_touched.txt','a').write('\n'.join(sorted(touched))+'\n')
print(len(touched),'specs updated;',len(log),'changes'); [print(' ',x) for x in log]
