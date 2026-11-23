# usage: batch.sh LAST 'python dict literal {task: ({codes}, kwargs)}'  -> saves, writes, renders next
cd ~/GenIA-data/labels/relations/v1/claude_specs
PYTHONPATH=~/GenIA-code/02_labeling/claude_labeler python3 -c "
from helpers import *
D=$2
T={t:quick(t,c[0],**(c[1] if len(c)>1 else {})) for t,c in D.items()}
save(T)
open('${TMPDIR:-/tmp}/genia_labeler/batch.txt','w').write(':'.join(f'$PWD/{t:04d}.json' for t in D))
" || exit 1
SPEC_FILES=$(cat ${TMPDIR:-/tmp}/genia_labeler/batch.txt) zsh ~/GenIA-code/02_labeling/claude_labeler/ls_env.sh ~/GenIA-code/02_labeling/claude_labeler/write_ann.py | grep -cE "^ok"
zsh ~/GenIA-code/02_labeling/claude_labeler/next_sheet.sh $1 | tail -9
