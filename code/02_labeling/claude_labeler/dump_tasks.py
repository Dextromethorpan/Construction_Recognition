# Dump all tasks (id, data, predictions) to JSON for offline rendering.
import json, os
from projects.models import Project
p=Project.objects.get(id=1)
out=[]
for t in p.tasks.order_by('id'):
    preds=[r for pr in t.predictions.all() for r in pr.result]
    out.append({"id":t.id,"data":t.data,"pred":preds,"n_ann":t.annotations.count()})
json.dump(out,open(os.path.expanduser('~/GenIA-data/labeling_queue/relation_v1/claude_tasks_dump.json'),'w'))
print(len(out))
