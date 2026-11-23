"""Phase C: seeded random sample of 50 people from pass-1 tasks, stratified by split_v2 (proportional, largest remainder).
Writes labels/relations/v2/human_check_sample.csv with task id + person box id + box (no labels, so the human check stays blind).
usage: human_sample.py PEOPLE_CSV [N=50] [SEED=20261120]"""
import csv, random, sys, os, collections
src = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 50; SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 20261120
R = [r for r in csv.DictReader(open(src)) if r['pass'] == '1' and not r.get('image_flag')]
by = collections.defaultdict(list)
for r in R: by[r['split_v2']].append(r)
quota = {k: N * len(v) / len(R) for k, v in by.items()}
alloc = {k: int(q) for k, q in quota.items()}
for k in sorted(quota, key=lambda k: quota[k] - alloc[k], reverse=True)[:N - sum(alloc.values())]: alloc[k] += 1
rng = random.Random(SEED); out = []
for k in sorted(by):
    pool = sorted(by[k], key=lambda r: (int(r['task_id']), r['person_id']))
    out += rng.sample(pool, alloc[k])
out.sort(key=lambda r: (r['split_v2'], int(r['task_id']), r['person_id']))
dst = os.path.expanduser('~/GenIA-data/labels/relations/v2/human_check_sample.csv')
with open(dst, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['sample_idx', 'task_id', 'global_id', 'split_v2', 'task_kind', 'person_id', 'x', 'y', 'w', 'h', 'seed', 'source'])
    for i, r in enumerate(out): w.writerow([i, r['task_id'], r['global_id'], r['split_v2'], r['task_kind'], r['person_id'], r['x'], r['y'], r['w'], r['h'], SEED, os.path.basename(src)])
print(dst, len(out), dict(alloc), 'population', len(R), {k: len(v) for k, v in by.items()})
print('tasks', len({r['task_id'] for r in out}), 'task_kind', collections.Counter(r['task_kind'] for r in out))
