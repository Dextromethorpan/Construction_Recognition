"""PPE review: crop every person from claude_specs into numbered sheets of 20.
usage: ppe_sheet.py build  -> writes _render/ppe_index.json
       ppe_sheet.py sheet N -> renders sheet N to _render/ppe_N.jpg and prints the index map"""
import json, os, sys, glob
from urllib.parse import unquote
from PIL import Image, ImageDraw, ImageFont
H=os.path.expanduser('~/GenIA-data'); OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'_render'); os.makedirs(OUT,exist_ok=True)
SP=f'{H}/labels/relations/v1/claude_specs'
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16)
if sys.argv[1]=='build':
    tasks={t['id']:t for t in json.load(open(f'{H}/labeling_queue/relation_v1/claude_tasks_dump.json'))}
    idx=[]
    for f in sorted(glob.glob(f'{SP}/*.json')):
        s=json.load(open(f))
        for b in s['boxes']:
            if b['label']=='person': idx.append(dict(task=s['task'],id=b['id'],box=[b['x'],b['y'],b['w'],b['h']],img=tasks[s['task']]['data']['image'],
                cur=[b.get('helmet_state'),b.get('harness_state'),b.get('tied_off'),b.get('support')]))
    json.dump(idx,open(f'{OUT}/ppe_index.json','w')); print(len(idx),'people,',(len(idx)+19)//20,'sheets')
else:
    n=int(sys.argv[2]); idx=json.load(open(f'{OUT}/ppe_index.json'))[n*20:(n+1)*20]
    T=300; sheet=Image.new('RGB',(T*5,(T+22)*4),'black'); d=ImageDraw.Draw(sheet)
    cache={}
    for k,p in enumerate(idx):
        path=os.path.join(H,unquote(p['img'].split('?d=')[1]))
        im=cache.get(path) or Image.open(path).convert('RGB'); cache[path]=im
        W,Hh=im.size; x,y,w,h=p['box']
        m=0.35; x0=max(0,(x-w*m)/100*W); y0=max(0,(y-h*0.15)/100*Hh); x1=min(W,(x+w*(1+m))/100*W); y1=min(Hh,(y+h*1.15)/100*Hh)
        c=im.crop((int(x0),int(y0),int(max(x1,x0+2)),int(max(y1,y0+2))))
        r=min(T/c.width,T/c.height); c=c.resize((max(1,int(c.width*r)),max(1,int(c.height*r)))); cx,cy=(k%5)*T,(k//5)*(T+22)
        sheet.paste(c,(cx+(T-c.width)//2,cy+22+(T-c.height)//2))
        # mark the person box inside the crop
        sx=c.width/(x1-x0); sy=c.height/(y1-y0); ox=cx+(T-c.width)//2; oy=cy+22+(T-c.height)//2
        d.rectangle([ox+(x/100*W-x0)*sx,oy+(y/100*Hh-y0)*sy,ox+((x+w)/100*W-x0)*sx,oy+((y+h)/100*Hh-y0)*sy],outline='red',width=2)
        d.text((cx+3,cy+2),f"{k}  t{p['task']}",fill='yellow',font=font)
    sheet.save(f'{OUT}/ppe_{n}.jpg',quality=82)
    print(f'{OUT}/ppe_{n}.jpg', [(k,p['task'],p['id'],'/'.join((v or '-').split(':')[-1] for v in p['cur'])) for k,p in enumerate(idx)])
