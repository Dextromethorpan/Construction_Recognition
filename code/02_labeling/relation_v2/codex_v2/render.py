"""Render a project-2 ("GenIA relation v2") task for Codex. Wraps ../claude_v2/render_v2.py; output goes to codex_v2/_render/.
usage:
  render.py TASK_ID [--crop x0,y0,x1,y1]   -> pre-filled boxes only (what Label Studio showed before labeling)
  render.py --spec /abs/path/NNNN.json     -> overlay of a v2 spec (boxes with ids + codes, platforms, ankles) -> tNNNN_spec.jpg
Each person line printed below the path shows the person's fields from the spec (or prefill), so you can check them."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'claude_v2'))
import render_v2 as R
R.OUT = os.path.join(HERE, '_render'); os.makedirs(R.OUT, exist_ok=True)
PER = ['support_surface', 'feet_on', 'feet_visible', 'proximity', 'helmet_state', 'harness_state', 'anchor_visible']


def spec_to_pred(s):
    pred = []
    for b in s.get('boxes', []):
        v = dict(x=b['x'], y=b['y'], width=b['w'], height=b['h'])
        pred.append(dict(id=b['id'], from_name='box', type='rectanglelabels', value=dict(v, rectanglelabels=[b['label']])))
        for f in PER:
            if b.get(f): pred.append(dict(id=b['id'], from_name=f, type='choices', value=dict(v, choices=[b[f]])))
        if b.get('on'): pred.append(dict(id=b['id'], from_name='on', type='choices', value=dict(v, choices=['on=' + b['on']])))
        for k, (x, y) in (b.get('kp') or {}).items():
            pred.append(dict(id=f"{b['id']}_{k}", from_name='kp', type='keypointlabels', value=dict(x=x, y=y, keypointlabels=[k])))
    for p in s.get('platforms', []):
        pred.append(dict(id=p['id'], from_name='platform', type='polygonlabels', value=dict(points=p['points'])))
    return pred


if __name__ == '__main__':
    a = sys.argv[1:]
    if a and a[0] == '--spec':
        s = json.load(open(a[1])); tid = int(s['task'])
        R.tasks[tid] = dict(R.tasks[tid], pred=spec_to_pred(s))
        R.render(tid)
        src = os.path.join(R.OUT, f't{tid}.jpg'); dst = os.path.join(R.OUT, f't{tid}_spec.jpg'); os.replace(src, dst)
        print('spec overlay ->', dst, '| all_people_boxed:', s.get('all_people_boxed'), '| image_flag:', s.get('image_flag'))
    else:
        crop = None
        if '--crop' in a: i = a.index('--crop'); crop = [float(v) for v in a[i + 1].split(',')]; a = a[:i] + a[i + 2:]
        for x in a: R.render(int(x), crop)
