# Dump all tasks of project 2 (GenIA relation v2) with data + predictions, for offline rendering (v2 copy of claude_labeler/dump_tasks.py).
import json, os
from projects.models import Project
p=Project.objects.get(title="GenIA relation v2"); assert p.id!=1
out=[]
for t in p.tasks.order_by('id'):
    preds=[r for pr in t.predictions.all() for r in pr.result]
    out.append({"id":t.id,"data":t.data,"pred":preds,"n_ann":t.annotations.count()})
json.dump(out,open(os.path.expanduser('~/GenIA-data/labeling_queue/relation_v2/claude_tasks_dump_v2.json'),'w'))
print(len(out))
