#!/bin/zsh
# Phase B driver: blind pass 2 (4 shards) -> retry -> export + kappa (Claude pass 1 vs Codex pass 2, before review)
# -> review of all Claude pass-1 tasks (8 shards) -> retry -> export + kappa after review -> runs_v2/DONE. STOP file stops it.
P=$HOME/GenIA-code/02_labeling/codex_prompts; V=$HOME/GenIA-data/labels/relations/v2; L=$P/runs_v2; mkdir -p $L
RV=$HOME/GenIA-code/02_labeling/relation_v2; PY=$HOME/GenIA-data/.venv/bin/python
sweep() { for i in $(seq 0 $(($2-1))); do zsh $P/supervise_v2.sh $1 1000 $i $2 >> $L/sup_$1_$i.log 2>&1 & done; wait; }
kap() { cd $RV; E=$(zsh ../claude_labeler/ls_env.sh $RV/export_v2.py | grep export_ | cut -d' ' -f1)
  $PY parse_v2.py $E ${E%.json}_people.csv --frozen-v1 $HOME/GenIA-data/labels/relations/v1/frozen_v1/codex_reviewed.json > /dev/null
  $PY kappa_v2.py ${E%.json}_people.csv --pass2 --out $V/kappa_pass2_$1.csv > $L/kappa_$1.txt 2>&1; echo "$E" >> $L/kappa_$1.txt; }
echo "$(date +%H:%M:%S) pass2 start" >> $L/driver.log
sweep v2_pass2 ${NP2:-2}; [[ -f $P/STOP ]] || sweep v2_pass2 1
echo "$(date +%H:%M:%S) pass2 end: $(grep -c ',2,' $V/codex_labeled_tasks.csv 2>/dev/null) pass-2 tasks" >> $L/driver.log
[[ -f $P/STOP ]] || kap before_review; echo "$(date +%H:%M:%S) kappa before review done" >> $L/driver.log
[[ -f $P/STOP ]] && { echo "$(date +%H:%M:%S) STOP" >> $L/driver.log; exit 0; }
sweep v2_review ${NRV:-3}; [[ -f $P/STOP ]] || sweep v2_review 1
echo "$(date +%H:%M:%S) review end: $(ls $V/codex_reviews | wc -l) reviews" >> $L/driver.log
kap after_review
echo "$(date +%H:%M:%S) finished: $(ls $V/codex_reviews | wc -l) reviews" > $L/DONE
