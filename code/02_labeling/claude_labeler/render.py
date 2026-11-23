"""Render a task image with a % grid and numbered pre-annotation boxes, for Claude to read.
usage: python render.py TASK_ID [TASK_ID...] [--crop x0,y0,x1,y1 (percent)]"""
import json, os, sys
from urllib.parse import unquote
from PIL import Image, ImageDraw, ImageFont
H=os.path.expanduser('~/GenIA-data')
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'_render')
os.makedirs(OUT,exist_ok=True)
tasks={t['id']:t for t in json.load(open(f'{H}/labeling_queue/relation_v1/claude_tasks_dump.json'))}
def path(t): return os.path.join(H, unquote(t['data']['image'].split('?d=')[1]))
args=sys.argv[1:]; crop=None
if '--crop' in args:
    i=args.index('--crop'); crop=[float(v) for v in args[i+1].split(',')]; args=args[:i]+args[i+2:]
try: font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14)
except Exception: font=ImageFont.load_default()
COL={'person':'red','scaffold_structure':'blue','helmet':'yellow','head':'orange','harness':'magenta','ladder':'cyan'}
for a in args:
    t=tasks[int(a)]; im=Image.open(path(t)).convert('RGB'); W,Hh=im.size
    x0,y0,x1,y1=crop or (0,0,100,100)
    im=im.crop((W*x0/100,Hh*y0/100,W*x1/100,Hh*y1/100))
    s=max(1.0, 1000/max(im.size)); im=im.resize((int(im.width*s),int(im.height*s)))
    d=ImageDraw.Draw(im); cw,ch=im.size
    def px(x,y): return ((x-x0)/(x1-x0)*cw,(y-y0)/(y1-y0)*ch)
    step=5 if crop else 10
    v=0
    while v<=100:
        if x0<=v<=x1:
            X,_=px(v,0); d.line([(X,0),(X,ch)],fill=(255,255,255) if v%10==0 else (160,160,160),width=1); d.text((X+2,2),str(v),fill='white',font=font,stroke_width=2,stroke_fill='black')
        if y0<=v<=y1:
            _,Y=px(0,v); d.line([(0,Y),(cw,Y)],fill=(255,255,255) if v%10==0 else (160,160,160),width=1); d.text((2,Y+2),str(v),fill='white',font=font,stroke_width=2,stroke_fill='black')
        v+=step
    for k,r in enumerate(t['pred']):
        if r.get('type')!='rectanglelabels': continue
        vv=r['value']; lab=vv['rectanglelabels'][0]
        a1=px(vv['x'],vv['y']); a2=px(vv['x']+vv['width'],vv['y']+vv['height'])
        d.rectangle([a1,a2],outline=COL.get(lab,'green'),width=2)
        d.text((a1[0]+3,a1[1]+3),f'#{k} {lab}',fill=COL.get(lab,'green'),font=font,stroke_width=2,stroke_fill='black')
    fn=f'{OUT}/t{a}'+('_crop' if crop else '')+'.jpg'; im.save(fn,quality=88)
    print(fn, t['data']['global_id'], t['data']['hint'], 'orig', W,Hh)
