#!/bin/zsh
# Resume Phase B review only (pass 2 done, pre-review kappa kept): remaining v2_review tasks with 3 workers
# (a 429 writes THROTTLE and worker 3 exits = drop to 2), sweep leftovers with 2 then 1, then final export + kappa.
P=$HOME/GenIA-code/02_labeling/codex_prompts; V=$HOME/GenIA-data/labels/relations/v2; L=$P/runs_v2
RV=$HOME/GenIA-code/02_labeling/relation_v2; PY=$HOME/GenIA-data/.venv/bin/python
sweep() { for i in $(seq 0 $(($2-1))); do zsh $P/supervise_v2.sh $1 1000 $i $2 >> $L/sup_$1_r_$i.log 2>&1 & done; wait; }
echo "$(date +%H:%M:%S) resume review: $(ls $V/codex_reviews | wc -l) done" >> $L/driver.log
sweep v2_review 3; [[ -f $P/STOP ]] || sweep v2_review 2; [[ -f $P/STOP ]] || sweep v2_review 1
echo "$(date +%H:%M:%S) review end: $(ls $V/codex_reviews | wc -l) reviews" >> $L/driver.log
cd $RV; E=$(zsh ../claude_labeler/ls_env.sh $RV/export_v2.py | grep export_ | cut -d' ' -f1)
$PY parse_v2.py $E ${E%.json}_people.csv --frozen-v1 $HOME/GenIA-data/labels/relations/v1/frozen_v1/codex_reviewed.json > /dev/null
$PY kappa_v2.py ${E%.json}_people.csv --pass2 --out $V/kappa_pass2_final.csv > $L/kappa_final.txt 2>&1; echo "$E" >> $L/kappa_final.txt
echo "$(date +%H:%M:%S) finished: $(ls $V/codex_reviews | wc -l) reviews" > $L/DONE_FINAL
