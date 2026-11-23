#!/bin/zsh
# After the 4 review workers end: retry missed tasks once with 4 shards, then write done marker.
P=$HOME/GenIA-code/02_labeling/codex_prompts; V=$HOME/GenIA-data/labels/relations/v1
while pgrep -f "^zsh .*supervise.sh review" >/dev/null; do sleep 120; done
if [[ $(ls $V/codex_reviews | wc -l) -lt 754 && ! -f $P/STOP ]]; then
  for i in 0 1 2 3 4 5 6 7; do zsh $P/supervise.sh review 1000 $i 8 >> $P/runs/supervisor_$i.log 2>&1 & done; wait
fi
echo "$(date +%H:%M:%S) finished: $(ls $V/codex_reviews | wc -l) reviews" > $P/runs/DONE
