"""Draw a spec over its image to check it. usage: python check.py SPEC.json"""
import json, os, sys
from urllib.parse import unquote
from PIL import Image, ImageDraw, ImageFont
H=os.path.expanduser('~/GenIA-data')
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'_render')
os.makedirs(OUT,exist_ok=True)
tasks={t['id']:t for t in json.load(open(f'{H}/labeling_queue/relation_v1/claude_tasks_dump.json'))}
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14)
for fn in sys.argv[1:]:
    s=json.load(open(fn)); t=tasks[s['task']]
    im=Image.open(os.path.join(H,unquote(t['data']['image'].split('?d=')[1]))).convert('RGB')
    sc=1000/max(im.size); im=im.resize((int(im.width*sc),int(im.height*sc))); W,Hh=im.size
    d=ImageDraw.Draw(im,'RGBA'); P=lambda x,y:(x*W/100,y*Hh/100)
    for p in s.get('platforms',[]):
        d.polygon([P(*q) for q in p['points']],fill=(0,255,0,90),outline=(0,255,0,255))
    for b in s.get('boxes',[]):
        c={'person':(255,0,0,255)}.get(b['label'],(0,120,255,255))
        d.rectangle([P(b['x'],b['y']),P(b['x']+b['w'],b['y']+b['h'])],outline=c,width=2)
        d.text(P(b['x']+0.3,b['y']+0.3),f"{b['id']} {b.get('support','')}",fill=c,font=font,stroke_width=2,stroke_fill='black')
        for k,(x,y) in (b.get('kp') or {}).items():
            X,Y=P(x,y); d.ellipse([X-4,Y-4,X+4,Y+4],fill=(255,255,0,255) if k.startswith('left') else (0,255,255,255))
    o=f"{OUT}/check_{s['task']}.jpg"; im.save(o); print(o)
