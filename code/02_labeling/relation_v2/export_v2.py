"""Snapshot all annotations of project 2 (GenIA relation v2) to labels/relations/v2/export_<timestamp>.json (never overwrites). v2 copy of claude_labeler/export_ann.py."""
import json, os, datetime
from projects.models import Project
P=Project.objects.get(title="GenIA relation v2"); assert P.id!=1
out=[]
for t in P.tasks.order_by('id'):
    anns=[dict(id=a.id,completed_by=a.completed_by_id,created_at=a.created_at.isoformat(),updated_at=a.updated_at.isoformat(),was_cancelled=a.was_cancelled,result=a.result) for a in t.annotations.all()]
    if anns: out.append(dict(id=t.id,data=t.data,annotations=anns))
ts=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
fn=os.path.expanduser(f'~/GenIA-data/labels/relations/v2/export_{ts}.json')
assert not os.path.exists(fn)
json.dump(out,open(fn,'w'))
print(fn,len(out),'tasks with annotations')
