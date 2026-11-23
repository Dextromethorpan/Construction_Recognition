"""Write Claude-made annotations into Label Studio (project 1) from spec JSON files.
Run through ls_env.sh. Reads SPEC_FILES (colon-separated paths). Coordinates are percent of the image.
Every task written is appended to ~/GenIA-data/labels/relations/v1/claude_labeled_tasks.csv."""
import json, os, csv, datetime
from projects.models import Project
from tasks.models import Task, Annotation
from users.models import User
P=Project.objects.get(id=1); U=User.objects.get(id=1)
WHO=os.environ.get('ANNOTATOR','claude')
assert WHO in ('claude','codex'), WHO
MAN=os.path.expanduser(f'~/GenIA-data/labels/relations/v1/{WHO}_labeled_tasks.csv')
os.makedirs(os.path.dirname(MAN),exist_ok=True)
PER=['support','proximity','helmet_state','harness_state','tied_off','height_band','posture','visibility','feet_on']
IMG=['viewpoint','distance','lighting','occlusion','image_flag']
def build(s, W, H):
    R=[]; base=dict(to_name='image',original_width=W,original_height=H,image_rotation=0)
    for b in s.get('boxes',[]):
        R.append(dict(base,id=b['id'],from_name='box',type='rectanglelabels',
            value=dict(x=b['x'],y=b['y'],width=b['w'],height=b['h'],rotation=0,rectanglelabels=[b['label']])))
        for f in PER:
            if f in b and b[f] not in (None,[],''):
                v=b[f] if isinstance(b[f],list) else [b[f]]
                R.append(dict(base,id=b['id'],from_name=f,type='choices',value=dict(x=b['x'],y=b['y'],width=b['w'],height=b['h'],rotation=0,choices=v)))
        for k,(x,y) in (b.get('kp') or {}).items():
            R.append(dict(base,id=f"{b['id']}_{k}",from_name='kp',type='keypointlabels',value=dict(x=x,y=y,width=0.6,keypointlabels=[k])))
    for p in s.get('platforms',[]):
        R.append(dict(base,id=p['id'],from_name='platform',type='polygonlabels',value=dict(points=p['points'],closed=True,polygonlabels=['scaffold_platform'])))
    for b in s.get('boxes',[]):
        if b.get('on'):
            R.append(dict(type='relation',from_id=b['id'],to_id=b['on'],direction='right',labels=[b.get('support')]))
    for f in IMG:
        if s.get(f): R.append(dict(base,id=f'img_{f}',from_name=f,type='choices',value=dict(choices=[s[f]])))
    return R
new=not os.path.exists(MAN)
mf=open(MAN,'a',newline=''); w=csv.writer(mf)
if new: w.writerow(['task_id','global_id','pass','annotation_id','written_at','annotator'])
for fn in os.environ['SPEC_FILES'].split(':'):
    s=json.load(open(fn)); t=Task.objects.get(id=s['task'],project=P)
    if WHO=='claude': assert t.data.get('pass')==1, f"task {t.id} is a pass-2 task; Claude must not label it (blind second pass)"
    pr=t.predictions.first(); W=H=None
    if pr and pr.result: W=pr.result[0].get('original_width'); H=pr.result[0].get('original_height')
    if not W:
        from PIL import Image; from urllib.parse import unquote
        W,H=Image.open(os.path.join(os.path.expanduser('~/GenIA-data'),unquote(t.data['image'].split('?d=')[1]))).size
    res=build(s,W,H)
    old=t.annotations.filter(completed_by=U)
    ids=[a.id for a in old]
    if ids and not s.get('replace'): print('skip, already annotated', t.id); continue
    old.delete()
    a=Annotation.objects.create(task=t,project=P,completed_by=U,result=res,lead_time=0,parent_prediction=pr)
    t.update_is_labeled(); t.save()
    w.writerow([t.id,t.data['global_id'],t.data.get('pass'),a.id,datetime.datetime.now().isoformat(timespec='seconds'),WHO])
    print('ok', t.id, t.data['global_id'], len(res), 'results')
mf.close()
