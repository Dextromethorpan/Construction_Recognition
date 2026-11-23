"""Apply strict-rule harness decisions. usage: harness_apply.py "idx idx ..."  (indices into harness_recheck_people.csv to set not_visible)
Writes decisions to harness_recheck_decisions.csv; push happens with harness_push.py."""
import csv,sys,os
H=os.path.expanduser('~/GenIA-data/labels/relations/v2')
R=list(csv.DictReader(open(H+'/harness_recheck_people.csv')))
a,n=int(sys.argv[1]),int(sys.argv[2]); V={int(x) for x in sys.argv[3].split()}
fn=H+'/harness_recheck_decisions.csv'; new=not os.path.exists(fn)
w=csv.writer(open(fn,'a'))
if new: w.writerow(['idx','task_id','person_id','decision'])
for i in range(a,min(a+n,len(R))): w.writerow([i,R[i]['task_id'],R[i]['person_id'],'not_visible' if i in V else 'not_worn'])
print(len(V),'->not_visible of',min(n,len(R)-a))
