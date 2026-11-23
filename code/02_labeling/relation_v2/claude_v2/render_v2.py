"""v2 render for Claude (project 2). Copy of claude_labeler/render.py adapted to project 2's dump.
Makes one composite per task: full image with % grid, person boxes (id), other boxes, platform polygons, ankles,
plus a panel of zoomed person crops (helmet/harness/feet). usage: render_v2.py TASK [TASK..] [--crop x0,y0,x1,y1]"""
import json, os, sys
from urllib.parse import unquote
from PIL import Image, ImageDraw, ImageFont
H=os.path.expanduser('~/GenIA-data')
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'_render'); os.makedirs(OUT,exist_ok=True)
DUMP=f'{H}/labeling_queue/relation_v2/claude_tasks_dump_v2.json'
tasks={t['id']:t for t in json.load(open(DUMP))}
def path(t): return os.path.join(H, unquote(t['data']['image'].split('?d=')[1]))
try: font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',15); sfont=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',12)
except Exception: font=sfont=ImageFont.load_default()
COL={'person':(255,40,40),'scaffold_structure':(60,120,255),'helmet':(255,230,0),'head':(255,150,0),'harness':(255,0,255),'ladder':(0,255,255)}
def parse(t):
    P=[];O=[];PL=[];KP={};CH={}
    for r in t['pred']:
        v=r.get('value',{})
        if r.get('from_name')=='box':
            lab=v['rectanglelabels'][0]; (P if lab=='person' else O).append((r['id'],lab,v['x'],v['y'],v['width'],v['height']))
        elif r.get('from_name')=='platform': PL.append((r['id'],v['points']))
        elif r.get('type')=='keypointlabels': KP.setdefault(r['id'].rsplit('_',2)[0] if r['id'].endswith('ankle') else r['id'],[]).append((v['keypointlabels'][0],v['x'],v['y']))
        elif r.get('type')=='choices' and r.get('id') and not r['id'].startswith('img_'): CH.setdefault(r['id'],{})[r['from_name']]=v['choices'][0]
    return P,O,PL,KP,CH
def kpowner(rid): 
    for s in ('_left_ankle_occluded','_right_ankle_occluded','_left_ankle','_right_ankle'):
        if rid.endswith(s): return rid[:-len(s)]
    return rid
def render(tid,crop=None):
    t=tasks[tid]; im0=Image.open(path(t)).convert('RGB'); W,Hh=im0.size
    P,O,PL,_,CH=parse(t)
    KP={}
    for r in t['pred']:
        if r.get('type')=='keypointlabels': v=r['value']; KP.setdefault(kpowner(r['id']),[]).append((v['keypointlabels'][0],v['x'],v['y']))
    x0,y0,x1,y1=crop or (0,0,100,100)
    im=im0.crop((W*x0/100,Hh*y0/100,W*x1/100,Hh*y1/100))
    s=1000/max(im.size); im=im.resize((max(1,int(im.width*s)),max(1,int(im.height*s))))
    d=ImageDraw.Draw(im,'RGBA'); cw,ch=im.size
    def px(x,y): return ((x-x0)/(x1-x0)*cw,(y-y0)/(y1-y0)*ch)
    step=5 if crop else 10; v=0
    while v<=100:
        if x0<=v<=x1:
            X,_=px(v,0); d.line([(X,0),(X,ch)],fill=(255,255,255,110) if v%10==0 else (160,160,160,80),width=1); d.text((X+2,2),str(v),fill='white',font=sfont,stroke_width=2,stroke_fill='black')
        if y0<=v<=y1:
            _,Y=px(0,v); d.line([(0,Y),(cw,Y)],fill=(255,255,255,110) if v%10==0 else (160,160,160,80),width=1); d.text((2,Y+2),str(v),fill='white',font=sfont,stroke_width=2,stroke_fill='black')
        v+=step
    for pid,pts in PL:
        d.polygon([px(*q) for q in pts],fill=(0,255,0,45),outline=(0,255,0,255)); d.text(px(*pts[0]),pid,fill=(0,255,0),font=sfont,stroke_width=2,stroke_fill='black')
    for rid,lab,x,y,w,h in O:
        if lab=='scaffold_structure': d.rectangle([px(x,y),px(x+w,y+h)],outline=COL[lab]+(150,),width=1)
        else: d.rectangle([px(x,y),px(x+w,y+h)],outline=COL.get(lab,(0,200,0)),width=1)
    for rid,lab,x,y,w,h in P:
        d.rectangle([px(x,y),px(x+w,y+h)],outline=COL['person'],width=2)
        d.text((px(x,y)[0]+2,px(x,y)[1]+1),rid,fill=(255,60,60),font=font,stroke_width=2,stroke_fill='black')
        for k,kx,ky in KP.get(rid,[]):
            X,Y=px(kx,ky); c=(255,255,0) if k.startswith('left') else (0,255,255); d.ellipse([X-4,Y-4,X+4,Y+4],outline=c,width=2 if 'occl' in k else 3)
    fn=f'{OUT}/t{tid}'+('_crop' if crop else '')+'.jpg'
    # person crop panel
    crops=[]
    if not crop:
        for rid,lab,x,y,w,h in P:
            if w*W/100<6 or h*Hh/100<10: pass
            mx,my=w*0.25,h*0.08
            c=im0.crop((max(0,(x-mx))*W/100,max(0,(y-my))*Hh/100,min(100,x+w+mx)*W/100,min(100,y+h+my)*Hh/100))
            sc=260/max(1,c.height); c=c.resize((max(1,min(260,int(c.width*sc))),260))
            dc=ImageDraw.Draw(c); dc.text((3,2),rid,fill=(255,60,60),font=font,stroke_width=2,stroke_fill='black'); crops.append(c)
    if crops:
        cols=max(1,min(len(crops),1000//190)); 
        rows=[crops[i:i+cols] for i in range(0,len(crops),cols)]
        pw=max(sum(c.width for c in r)+4*len(r) for r in rows); ph=264*len(rows)
        pan=Image.new('RGB',(max(pw,cw),ph),(30,30,30)); yy=0
        for r in rows:
            xx=0
            for c in r: pan.paste(c,(xx,yy)); xx+=c.width+4
            yy+=264
        out=Image.new('RGB',(max(cw,pan.width),ch+ph+4),(0,0,0)); out.paste(im,(0,0)); out.paste(pan,(0,ch+4))
    else: out=im
    out.save(fn,quality=85)
    info=[f'{fn} | {tid} {t["data"]["hint"]} | {W}x{Hh}']
    for rid,lab,x,y,w,h in P:
        c=CH.get(rid,{}); info.append(f'  {rid}: [{x:.1f},{y:.1f},{w:.1f},{h:.1f}] {h*Hh/100:.0f}px '+' '.join(c.values())+' kp:'+','.join(k[0][0]+('o' if 'occ' in k[0] else '') for k in KP.get(rid,[])))
    oth={}
    for rid,lab,x,y,w,h in O: oth.setdefault(lab,[]).append(rid)
    info.append('  other: '+' '.join(f'{k}:{len(v)}' for k,v in oth.items())+f' | platforms: {[p for p,_ in PL]}')
    print('\n'.join(info))
if __name__=='__main__':
    a=sys.argv[1:]; crop=None
    if '--crop' in a: i=a.index('--crop'); crop=[float(v) for v in a[i+1].split(',')]; a=a[:i]+a[i+2:]
    for x in a: render(int(x),crop)
