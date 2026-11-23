"""Crop sheets of people currently harness:not_worn, for the strict-rule re-check. usage: harness_sheet.py START N
Prints index -> task/person. Crop = person box (whole), 220px tall, 8 per row, 24 per sheet."""
import csv,json,os,sys
from urllib.parse import unquote
from PIL import Image,ImageDraw,ImageFont
H=os.path.expanduser('~/GenIA-data'); S=H+'/labels/relations/v2/claude_specs'
T={t['id']:t for t in json.load(open(H+'/labeling_queue/relation_v2/claude_tasks_dump_v2.json'))}
R=list(csv.DictReader(open(H+'/labels/relations/v2/harness_recheck_people.csv')))
a,n=int(sys.argv[1]),int(sys.argv[2]); R=R[a:a+n]
f=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',16)
cache={}; tiles=[]
for k,r in enumerate(R):
    t=int(r['task_id'])
    if t not in cache: cache[t]=Image.open(os.path.join(H,unquote(T[t]['data']['image'].split('?d=')[1]))).convert('RGB')
    im=cache[t]; W,Hh=im.size
    b=[x for x in json.load(open(f'{S}/{t:04d}.json'))['boxes'] if x['id']==r['person_id']][0]
    c=im.crop((max(0,b['x']-2)*W/100,max(0,b['y']-2)*Hh/100,min(100,b['x']+b['w']+2)*W/100,min(100,b['y']+b['h']*0.75)*Hh/100))
    sc=240/max(1,c.height); c=c.resize((max(1,min(300,int(c.width*sc))),240))
    d=ImageDraw.Draw(c); d.text((3,2),str(a+k),fill=(255,40,40),font=f,stroke_width=2,stroke_fill='black')
    d.text((3,220),f"{b['h']*Hh/100:.0f}px",fill=(255,255,0),font=f,stroke_width=2,stroke_fill='black')
    tiles.append(c)
rows=[tiles[i:i+6] for i in range(0,len(tiles),6)]
Wd=max(sum(x.width+4 for x in r) for r in rows); out=Image.new('RGB',(Wd,244*len(rows)),(30,30,30)); y=0
for r in rows:
    x=0
    for c in r: out.paste(c,(x,y)); x+=c.width+4
    y+=244
fn=os.path.join(os.path.dirname(os.path.abspath(__file__)),'_render',f'hs_{a}.jpg'); out.save(fn,quality=85); print(fn)
