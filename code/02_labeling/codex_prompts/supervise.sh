#!/bin/zsh
# Feed Codex one prompt per task, sequentially. usage: supervise.sh pass2|review [max_tasks]
# Skips tasks already done. Logs each run to runs/<task>.log and the final message to runs/<task>.out
C=/Applications/ChatGPT.app/Contents/Resources/codex
P=$HOME/GenIA-code/02_labeling/codex_prompts; V=$HOME/GenIA-data/labels/relations/v1
KIND=$1; MAX=${2:-1000}; SHARD=${3:-0}; NSH=${4:-1}; n=0
for f in $P/$KIND/task_*.md; do
  t=${${f:t}#task_}; t=${t%.md}; ti=$((10#$t))
  if [[ $KIND == pass2 ]]; then grep -q "^$ti," $V/codex_labeled_tasks.csv 2>/dev/null && continue
  else [[ -f $V/codex_reviews/$t.json ]] && continue; fi
  (( ti % NSH == SHARD )) || continue
  [[ -f $P/STOP ]] && { echo "STOP file found"; break; }
  img=$($HOME/GenIA-data/.venv/bin/python $HOME/GenIA-code/02_labeling/claude_labeler/render.py $ti | cut -d' ' -f1)
  echo "$(date +%H:%M:%S) start $t"
  $C exec -C $HOME/GenIA-code --add-dir $HOME/GenIA-data -s workspace-write --skip-git-repo-check \
     -i "$img" -o $P/runs/$t.out "$(cat $f)

The rendered image with a % grid is attached (also at $img). Work fully autonomously; do not ask questions. Finish by running write_ann.py (or writing the review file) for this one task only." < /dev/null > $P/runs/$t.log 2>&1
  echo "$(date +%H:%M:%S) done  $t exit=$?"
  n=$((n+1)); [[ $n -ge $MAX ]] && break
done
exit 0
