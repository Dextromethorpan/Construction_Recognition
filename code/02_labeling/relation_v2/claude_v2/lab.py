"""Claude v2 batch labeler: compact per-task lines -> spec JSON (labels/relations/v2/claude_specs/NNNN.json) -> write_ann_v2.py.
usage: lab.py BATCHFILE   (or '-' for stdin)
Line:  TASK [bx=y|n] [flag=bad|none] [re] | item ; item ; ...
Items: ID CODE              person code (every kept person needs one)
       del ID[>KEEP]        delete box (move its ankles to KEEP if KEEP has none)
       add x,y,w,h CODE     new person (ids a0,a1..; no ankles)
       box ID x,y,w,h       refit a box
       pl x,y x,y x,y ...   new platform polygon (ids q0,q1..)   | plr y,x0,x1[,t] rectangle platform
       delpl PLID ; nok ID (drop ankles) ; ign x,y,w,h ; on ID PLID (explicit link) ; sc x,y,w,h (scaffold box)
CODE = SUPPORT FV [PROX] - HELMET HARNESS [ANCHOR] [:FEET]
  SUPPORT S platform F frame/attached ladder G ground E other_elevated U unclear
  FV 2/1/0  PROX f front b behind u under n next_to x far o no_scaffold  HELMET W N H(eld) V   HARNESS W N V  ANCHOR y n
  FEET override p f g o v  (default from SUPPORT; fv 0 => not_visible)"""
import json, os, re, sys, subprocess
H=os.path.expanduser('~/GenIA-data'); SPEC=f'{H}/labels/relations/v2/claude_specs'; os.makedirs(SPEC,exist_ok=True)
tasks={t['id']:t for t in json.load(open(f'{H}/labeling_queue/relation_v2/claude_tasks_dump_v2.json'))}
SUP=dict(S='scaffold_platform',F='scaffold_frame_or_attached_ladder',G='ground',E='other_elevated',U='unclear')
FEET=dict(p='platform',f='frame_or_ladder',g='ground',o='other_surface',v='not_visible')
DEF=dict(S='p',F='f',G='g',E='o',U='v')
PROX=dict(f='in_front_of',b='behind',u='under',n='next_to',x='far',o='no_scaffold_in_image')
HEL=dict(W='worn',N='not_worn',H='held',V='not_visible'); HAR=dict(W='worn',N='not_worn',V='not_visible')
RX=re.compile(r'^([SFGEU])([012])([fbunxo])?-([WNHV])([WNV])([yn])?(?::([pfgov]))?$')
def code(c):
    m=RX.match(c); assert m, f'bad code {c}'
    s,fv,pr,he,ha,an,fo=m.groups()
    if s in 'GEU': assert pr, f'{c}: proximity required for {SUP[s]}'
    if ha=='W': assert an, f'{c}: anchor needed when harness worn'
    fo=fo or ('v' if fv=='0' else DEF[s])
    if fv=='0': fo='v'
    assert not (s=='U' and fo!='v' and not m.group(7)), c
    d=dict(support_surface=SUP[s],feet_on='feet:'+FEET[fo],feet_visible='fv:'+{'2':'both','1':'one','0':'none'}[fv],
           helmet_state='helmet:'+HEL[he],harness_state='harness:'+HAR[ha])
    if pr: d['proximity']='prox:'+PROX[pr]
    if an: d['anchor_visible']='anchor:'+('yes' if an=='y' else 'no')
    return d
def owner(rid):
    for s in ('_left_ankle_occluded','_right_ankle_occluded','_left_ankle','_right_ankle'):
        if rid.endswith(s): return rid[:-len(s)],s[1:]
    return None,None
def f4(s): return [float(v) for v in s.split(',')]
def build(line):
    head,_,body=line.partition('|'); hp=head.split(); tid=int(hp[0]); t=tasks[tid]
    assert t['data']['pass']==1, f'{tid} is pass 2'
    opts=dict(x.split('=') for x in hp[1:] if '=' in x); rep='re' in hp[1:]
    boxes={}; order=[]; plats=[]; kp={}; img={}
    for r in t['pred']:
        v=r.get('value',{}); fn=r.get('from_name')
        if fn=='box': boxes[r['id']]=dict(id=r['id'],label=v['rectanglelabels'][0],x=v['x'],y=v['y'],w=v['width'],h=v['height']); order.append(r['id'])
        elif fn=='platform': plats.append(dict(id=r['id'],points=v['points']))
        elif r.get('type')=='keypointlabels':
            o,k=owner(r['id']); kp.setdefault(o,{})[v['keypointlabels'][0]]=[v['x'],v['y']]
        elif r.get('id','').startswith('img_'): img[fn]=v['choices'][0]
    codes={}; links={}; ign=[]; na=0; nq=0; nok=set()
    for it in [i.strip() for i in body.split(';') if i.strip()]:
        p=it.split()
        if p[0]=='del':
            a,_,k=p[1].partition('>'); assert a in boxes, f'{tid}: no box {a}'
            if k and not kp.get(k) and kp.get(a): kp[k]=kp[a]
            del boxes[a]; order.remove(a)
        elif p[0]=='add':
            i=f'a{na}'; na+=1; x,y,w,h=f4(p[1]); boxes[i]=dict(id=i,label='person',x=x,y=y,w=w,h=h); order.append(i); codes[i]=p[2]
        elif p[0]=='sc':
            i=f's{na}'; na+=1; x,y,w,h=f4(p[1]); boxes[i]=dict(id=i,label='scaffold_structure',x=x,y=y,w=w,h=h); order.append(i)
        elif p[0]=='box':
            x,y,w,h=f4(p[2]); boxes[p[1]].update(x=x,y=y,w=w,h=h)
        elif p[0]=='pl': plats.append(dict(id=f'q{nq}',points=[f4(q) for q in p[1:]])); nq+=1
        elif p[0]=='plr':
            v=f4(p[1]); y,x0,x1=v[:3]; th=v[3] if len(v)>3 else 2.5
            plats.append(dict(id=f'q{nq}',points=[[x0,y],[x1,y],[x1,y+th],[x0,y+th]])); nq+=1
        elif p[0]=='delpl': plats=[q for q in plats if q['id']!=p[1]]
        elif p[0]=='nok': nok.add(p[1])
        elif p[0]=='ign': x,y,w,h=f4(p[1]); ign.append(dict(x=x,y=y,w=w,h=h))
        elif p[0]=='on': links[p[1]]=p[2]
        else:
            assert p[0] in boxes, f'{tid}: no box {p[0]}'; assert boxes[p[0]]['label']=='person', f'{tid}: {p[0]} not person'; codes[p[0]]=p[1]
    out=[]
    for i in order:
        b=dict(boxes[i])
        if b['label']=='person':
            assert i in codes, f'{tid}: person {i} has no code'
            b.update(code(codes[i]))
            if i in kp and i not in nok:
                b['kp']={k:[x,y] for k,(x,y) in kp[i].items() if b['x']-2<=x<=b['x']+b['w']+2 and b['y']-2<=y<=b['y']+b['h']+4}
                if not b['kp']: del b['kp']
            if b['support_surface']=='scaffold_platform':
                if i in links: b['on']=links[i]
                elif plats:
                    fx,fy=b['x']+b['w']/2,b['y']+b['h']
                    def dist(q):
                        xs=[a for a,_ in q['points']]; ys=[c for _,c in q['points']]
                        dx=max(min(xs)-fx,0,fx-max(xs)); dy=max(min(ys)-fy,0,fy-max(ys)); return dx+dy
                    best=min(plats,key=dist)
                    if dist(best)<=8: b['on']=best['id']
        out.append(b)
    s=dict(task=tid,guideline_version='v2.0',boxes=out,platforms=plats,ignore=ign,all_people_boxed='boxed:'+{'y':'yes','n':'no'}[opts.get('bx','y')])
    for k in ('viewpoint','distance','lighting','occlusion'):
        if k in img: s[k]=img[k]
    if 'flag' in opts: s['image_flag']={'bad':'skip:bad_image','none':'skip:no_person_no_scaffold'}[opts['flag']]
    if rep: s['replace']=True
    fn=f'{SPEC}/{tid:04d}.json'; json.dump(s,open(fn,'w'),indent=1); return fn
src=sys.stdin if sys.argv[1]=='-' else open(sys.argv[1])
files=[build(l.strip()) for l in src if l.strip() and not l.startswith('#')]
env=dict(os.environ,ANNOTATOR='claude',SPEC_FILES=':'.join(files))
r=subprocess.run(['zsh',os.path.expanduser('~/GenIA-code/02_labeling/claude_labeler/ls_env.sh'),os.path.expanduser('~/GenIA-code/02_labeling/relation_v2/write_ann_v2.py')],env=env,capture_output=True,text=True)
ok=[l for l in r.stdout.splitlines() if l.startswith(('ok','skip'))]; print('\n'.join(ok)); print(len([l for l in ok if l.startswith('ok')]),'written of',len(files))
if len(ok)<len(files): print(r.stdout[-2000:],r.stderr[-2000:])
