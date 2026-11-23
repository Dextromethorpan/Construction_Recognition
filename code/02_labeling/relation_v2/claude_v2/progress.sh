# export project 2, parse with frozen v1 comparison, print summary
cd ~/GenIA-code/02_labeling/relation_v2
E=$(zsh ../claude_labeler/ls_env.sh $PWD/export_v2.py | grep export_ | cut -d' ' -f1)
echo $E
~/GenIA-data/.venv/bin/python parse_v2.py $E ${E%.json}_people.csv --frozen-v1 ~/GenIA-data/labels/relations/v1/frozen_v1/codex_reviewed.json
~/GenIA-data/.venv/bin/python - ${E%.json}_people.csv <<'PY'
import csv,sys,collections as C,json
R=list(csv.DictReader(open(sys.argv[1])))
T=json.load(open(__import__('os').path.expanduser('~/GenIA-data/labeling_queue/relation_v2/claude_tasks_dump_v2.json')))
pri={t['id']:t['data']['priority'] for t in T}
tasks={int(r['task_id']) for r in R}
print('tasks by priority',sorted(C.Counter(pri[t] for t in tasks).items()))
print('people',len(R))
print('support_surface',C.Counter(r['support_surface'] for r in R).most_common())
print('helmet',C.Counter(r['helmet_state'] for r in R).most_common())
print('harness',C.Counter(r['harness_state'] for r in R).most_common())
f=C.Counter(x for r in R for x in r['flags'].split('|') if x); print('flags',f.most_common())
rel=[r for r in R if r['v1_support']]
print('relabel people matched to v1',len(rel),'disagree',sum('disagrees_with_v1' in r['flags'] for r in rel))
print('v1->v2',C.Counter((r['v1_support'],r['support_v1']) for r in rel if 'disagrees_with_v1' in r['flags']).most_common())
PY
