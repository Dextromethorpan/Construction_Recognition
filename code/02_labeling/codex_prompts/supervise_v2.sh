#!/bin/zsh
# v2 copy of supervise.sh (v1, unchanged). Feed Codex one prompt per project-2 task, sequentially.
# usage: supervise_v2.sh v2_pass2|v2_review [max_tasks] [shard] [n_shards]. Skips tasks already done. STOP file stops it.
C=/Applications/ChatGPT.app/Contents/Resources/codex
P=$HOME/GenIA-code/02_labeling/codex_prompts; V=$HOME/GenIA-data/labels/relations/v2
R=$HOME/GenIA-code/02_labeling/relation_v2/codex_v2/render.py; PY=$HOME/GenIA-data/.venv/bin/python
KIND=$1; MAX=${2:-1000}; SHARD=${3:-0}; NSH=${4:-1}; n=0
mkdir -p $P/runs_v2
for f in $P/$KIND/task_*.md; do
  t=${${f:t}#task_}; t=${t%.md}; ti=$((10#$t))
  if [[ $KIND == v2_pass2 ]]; then grep -q "^$ti," $V/codex_labeled_tasks.csv 2>/dev/null && continue
  else [[ -f $V/codex_reviews/$t.json ]] && continue; fi
  (( ti % NSH == SHARD )) || continue
  [[ -f $P/STOP ]] && { echo "STOP file found"; break; }
  [[ -f $P/THROTTLE && $SHARD -ge 2 ]] && { echo "THROTTLE: worker $SHARD exits (drop to 2 workers)"; break; }
  if [[ $KIND == v2_pass2 ]]; then img=$($PY $R $ti | head -1 | cut -d' ' -f1)
  else $PY $R --spec $V/claude_specs/$t.json > /dev/null; img=$HOME/GenIA-code/02_labeling/relation_v2/codex_v2/_render/t${ti}_spec.jpg; fi
  echo "$(date +%H:%M:%S) start $t"
  for try in 1 2 3 4; do
    $C exec -C $HOME/GenIA-code --add-dir $HOME/GenIA-data -s workspace-write --skip-git-repo-check \
       -i "$img" -o $P/runs_v2/$KIND-$t.out "$(cat $f)

The rendered image with a % grid and person crops is attached (also at $img). Work fully autonomously; do not ask questions. Finish by running write_ann_v2.py (or writing the review file) for this one task only." < /dev/null > $P/runs_v2/$KIND-$t.log 2>&1
    rc=$?
    grep -q "429 Too Many Requests\|usage limit\|rate limit" $P/runs_v2/$KIND-$t.log || break
    touch $P/THROTTLE; echo "$(date +%H:%M:%S) 429 on $t (try $try), backing off $((try*180))s"; [[ -f $P/STOP ]] && break; sleep $((try*180))
    [[ $try == 4 ]] && { echo "$(date +%H:%M:%S) still rate-limited after 4 tries: writing STOP_RATELIMIT"; touch $P/STOP $P/STOP_RATELIMIT; }
  done
  echo "$(date +%H:%M:%S) done  $t exit=$rc"
  n=$((n+1)); [[ $n -ge $MAX ]] && break
done
exit 0
