# usage: next_sheet.sh LAST_DONE_ID  -> renders next 6 todo tasks into scratchpad/render/sheet.jpg and prints boxes
S=${TMPDIR:-/tmp}/genia_labeler; R=$(dirname $0)/_render
cd $S && python3 -c "
t=open('real_todo2.txt').read().split(); i=t.index('$1'); print(' '.join(t[i+1:i+10]))" > next.txt
~/GenIA-data/.venv/bin/python ~/GenIA-code/02_labeling/claude_labeler/render.py $(cat next.txt) | cut -d' ' -f2-4 | paste -d' ' <(tr ' ' '\n' < next.txt) -
cd $R && ~/GenIA-data/.venv/bin/python -c "
from PIL import Image
ids='$(cat $S/next.txt)'.split()
ims=[Image.open(f't{i}.jpg').resize((420,420)) for i in ids]
out=Image.new('RGB',(1260,1260),'white')
for k,im in enumerate(ims): out.paste(im,((k%3)*420,(k//3)*420))
out.save("sheet.jpg",quality=80)"
cd ~/GenIA-data/labels/relations/v1/claude_specs && PYTHONPATH=~/GenIA-code/02_labeling/claude_labeler python3 -c "
from helpers import from_pred
for t in '$(cat $S/next.txt)'.split():
  print(t,[(b['id'],b['label'][:6],round(b['x']),round(b['y']),round(b['w']),round(b['h'])) for b in from_pred(int(t))])"
