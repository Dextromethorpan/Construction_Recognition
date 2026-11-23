# usage: next.sh N -> render next N pass-1 tasks without a Claude v2 annotation (queue order)
cd ~/GenIA-code/02_labeling/relation_v2/claude_v2
~/GenIA-data/.venv/bin/python -c "
import json,csv,os
H=os.path.expanduser('~/GenIA-data'); T=json.load(open(H+'/labeling_queue/relation_v2/claude_tasks_dump_v2.json'))
m=H+'/labels/relations/v2/claude_labeled_tasks.csv'; done=set()
if os.path.exists(m): done={int(r['task_id']) for r in csv.DictReader(open(m))}
todo=[t['id'] for t in T if t['data']['pass']==1 and t['id'] not in done]
print(' '.join(map(str,todo[:$1])))" > _next.txt
~/GenIA-data/.venv/bin/python render_v2.py $(cat _next.txt) | sed "s#$PWD/_render/##"
