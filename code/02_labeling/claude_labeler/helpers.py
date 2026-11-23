"""Spec helpers for CS10k-style tasks (no keypoints: missing, not estimated)."""
def Pp(i,box,sup,on=None,helmet="helmet:worn",harness="harness:not_worn",tied="tied:no",h="h:unknown",post="standing",vis="vis:full",feet=None,prox=None):
    x,y,w,hh=box
    d={"id":i,"label":"person","x":x,"y":y,"w":w,"h":hh,"support":sup,"helmet_state":helmet,"harness_state":harness,"tied_off":tied,"height_band":h,"posture":post,"visibility":vis}
    if on: d["on"]=on
    if feet: d["feet_on"]=feet
    if prox: d["proximity"]=[prox]
    return d
B=lambda i,lab,x,y,w,h:{"id":i,"label":lab,"x":x,"y":y,"w":w,"h":h}
img=lambda v="view:ground",d="dist:mid",l="light:day",o="occ:partial":{"viewpoint":v,"distance":d,"lighting":l,"occlusion":o}
G=lambda i,box,**k:Pp(i,box,"none",h="h:ground",feet="feet:ground",**k)
U=lambda i,box,**k:Pp(i,box,"unclear",vis="vis:partial",feet="feet:not_visible",**k)
def save(T):
    import json
    for t,(b,p,im) in T.items():
        json.dump(dict(task=t,boxes=b,platforms=p,**im),open(f'{t:04d}.json','w'),indent=1)

def from_pred(task, person_defaults=None, keep=("person","helmet","head","harness","scaffold_structure","ladder")):
    """Start a spec from the task's existing boxes (after Claude has looked at the image)."""
    import json, os
    d={t['id']:t for t in json.load(open(os.path.expanduser('~/GenIA-data/labeling_queue/relation_v1/claude_tasks_dump.json')))}
    boxes=[]
    for k,r in enumerate(d[task]['pred']):
        if r.get('type')!='rectanglelabels': continue
        v=r['value']; lab=v['rectanglelabels'][0]
        if lab not in keep: continue
        b={"id":f"b{k}","label":lab,"x":v['x'],"y":v['y'],"w":v['width'],"h":v['height']}
        if lab=="person": b.update(person_defaults or {})
        boxes.append(b)
    return boxes
NEG=dict(support="none",helmet_state="helmet:not_worn",harness_state="harness:not_worn",tied_off="tied:no",height_band="h:ground",posture="standing",visibility="vis:full",feet_on="feet:ground")

def setp(boxes,i,**kw):
    for b in boxes:
        if b["id"]==i: b.update(kw)
    return boxes
ON=dict(support="stands_on",helmet_state="helmet:worn",harness_state="harness:not_visible",tied_off="tied:not_visible",height_band="h:mid",posture="standing",visibility="vis:full",feet_on="feet:platform")
CL=dict(support="climbing",helmet_state="helmet:worn",harness_state="harness:not_visible",tied_off="tied:not_visible",height_band="h:mid",posture="climbing_posture",visibility="vis:full",feet_on="feet:not_visible")
def pl(y,x0,x1,i="pl1",t=2.5): return {"id":i,"points":[[x0,y],[x1,y],[x1,y+t],[x0,y+t]]}

CODES={"S":dict(ON,height_band="h:unknown"),"C":dict(CL,height_band="h:unknown"),"N":dict(NEG,helmet_state="helmet:worn"),
       "U":dict(NEG,support="unclear",helmet_state="helmet:worn",height_band="h:unknown",feet_on="feet:not_visible",visibility="vis:partial")}
def quick(task, codes, default="U", extra=None, view="view:ground", dist="dist:mid", occ="occ:partial", flag=None):
    """codes: {'b1':'S',...}; any person not listed gets default. Per-image fields only."""
    b=from_pred(task)
    for x in b:
        if x["label"]=="person":
            x.update(CODES[codes.get(x["id"],default)])
    b+= [dict(Pp(f"e{k}",bb,"none"),**CODES[c]) for k,(bb,c) in enumerate(extra or [])]
    im=img(view,dist,o=occ)
    if flag: im["image_flag"]=flag
    return (b,[],im)
