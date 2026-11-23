"""Write AI-made v2 annotations into the Label Studio project "GenIA relation v2" from spec JSON files.
v2 copy of claude_labeler/write_ann.py (v1, project 1, unchanged). Run through ../claude_labeler/ls_env.sh.
Reads SPEC_FILES (colon-separated). Coordinates are percent of the image. Refuses to write to project 1 (v1, frozen).
Every task written is appended to ~/GenIA-data/labels/relations/v2/{ANNOTATOR}_labeled_tasks.csv.
Spec format: see codex_prompts/CONTEXT_v2.md."""
import json, os, csv, datetime
from projects.models import Project
from tasks.models import Task, Annotation
from users.models import User
P = Project.objects.get(title="GenIA relation v2")
assert P.id != 1, "project 1 is GenIA relation v1 (frozen)"
U = User.objects.get(id=1)
WHO = os.environ.get('ANNOTATOR', 'claude')
assert WHO in ('claude', 'codex', 'human'), WHO
MAN = os.path.expanduser(f'~/GenIA-data/labels/relations/v2/{WHO}_labeled_tasks.csv')
os.makedirs(os.path.dirname(MAN), exist_ok=True)
PER = ['support_surface', 'feet_on', 'feet_visible', 'proximity', 'helmet_state', 'harness_state', 'anchor_visible',
       'height_band', 'posture']
IMG = ['all_people_boxed', 'viewpoint', 'distance', 'lighting', 'occlusion', 'image_flag']
KP = {'left_ankle', 'right_ankle', 'left_ankle_occluded', 'right_ankle_occluded'}


def build(s, W, H):
    R = []; base = dict(to_name='image', original_width=W, original_height=H, image_rotation=0)
    for b in s.get('boxes', []):
        R.append(dict(base, id=b['id'], from_name='box', type='rectanglelabels',
                      value=dict(x=b['x'], y=b['y'], width=b['w'], height=b['h'], rotation=0, rectanglelabels=[b['label']])))
        for f in PER:
            if f in b and b[f] not in (None, [], ''):
                v = b[f] if isinstance(b[f], list) else [b[f]]
                R.append(dict(base, id=b['id'], from_name=f, type='choices',
                              value=dict(x=b['x'], y=b['y'], width=b['w'], height=b['h'], rotation=0, choices=v)))
        for k, (x, y) in (b.get('kp') or {}).items():
            assert k in KP, f"v2 keypoints are ankles only, got {k}"
            R.append(dict(base, id=f"{b['id']}_{k}", from_name='kp', type='keypointlabels',
                          value=dict(x=x, y=y, width=0.6, keypointlabels=[k])))
    for i, g in enumerate(s.get('ignore', [])):
        R.append(dict(base, id=f"ign{i}", from_name='ignore', type='rectanglelabels',
                      value=dict(x=g['x'], y=g['y'], width=g['w'], height=g['h'], rotation=0, rectanglelabels=['ignore_region'])))
    for p in s.get('platforms', []):
        R.append(dict(base, id=p['id'], from_name='platform', type='polygonlabels',
                      value=dict(points=p['points'], closed=True, polygonlabels=['scaffold_platform'])))
    for b in s.get('boxes', []):
        if b.get('on'):
            lab = 'on_platform' if b.get('support_surface') == 'scaffold_platform' else 'on_frame_or_ladder'
            R.append(dict(type='relation', from_id=b['id'], to_id=b['on'], direction='right', labels=[lab]))
    for f in IMG:
        if s.get(f): R.append(dict(base, id=f'img_{f}', from_name=f, type='choices', value=dict(choices=[s[f]])))
    return R


new = not os.path.exists(MAN)
mf = open(MAN, 'a', newline=''); w = csv.writer(mf)
if new: w.writerow(['task_id', 'global_id', 'pass', 'annotation_id', 'written_at', 'annotator', 'guideline_version'])
for fn in os.environ['SPEC_FILES'].split(':'):
    s = json.load(open(fn)); t = Task.objects.get(id=s['task'], project=P)
    if WHO == 'claude': assert t.data.get('pass') == 1, f"task {t.id} is pass 2; Claude must not label it (blind)"
    pr = t.predictions.first(); W = H = None
    if pr and pr.result: W = pr.result[0].get('original_width'); H = pr.result[0].get('original_height')
    if not W:
        from PIL import Image; from urllib.parse import unquote
        W, H = Image.open(os.path.join(os.path.expanduser('~/GenIA-data'), unquote(t.data['image'].split('?d=')[1]))).size
    res = build(s, W, H)
    old = t.annotations.filter(completed_by=U)
    if old.exists() and not s.get('replace'): print('skip, already annotated', t.id); continue
    old.delete()
    a = Annotation.objects.create(task=t, project=P, completed_by=U, result=res, lead_time=0, parent_prediction=pr)
    t.update_is_labeled(); t.save()
    w.writerow([t.id, t.data['global_id'], t.data.get('pass'), a.id, datetime.datetime.now().isoformat(timespec='seconds'),
                WHO, s.get('guideline_version', 'v2.0')])
    print('ok', t.id, t.data['global_id'], len(res), 'results')
mf.close()
