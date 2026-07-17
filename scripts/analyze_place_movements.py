import csv,json,sys
from pathlib import Path
import numpy as np,pandas as pd
out=Path(sys.argv[1]);p=pd.read_csv(out/'step_logs.csv');data=json.loads((out/'abcd_comparison.json').read_text())
rows=[]
for cfg,d in p.groupby('config'):
 actual_cos=[];object_cos=[]
 for seed,s in d.groupby('seed'):
  s=s.sort_values('place_step');goal=s[['goal_dx','goal_dy']].to_numpy();tcp=s[['actual_tcp_x','actual_tcp_y']].to_numpy();dtcp=np.vstack([[0,0],np.diff(tcp,axis=0)]);dobj=s[['object_step_x','object_step_y']].to_numpy()
  cos=lambda a,b:np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-8)
  actual_cos.extend(cos(goal,dtcp)[1:]);object_cos.extend(cos(goal,dobj)[1:])
 actual_cos=np.asarray(actual_cos);object_cos=np.asarray(object_cos);c=data['configs'][cfg]
 c['actual_tcp_toward_rate']=float(np.mean(actual_cos>0));c['actual_tcp_mean_cosine']=float(np.mean(actual_cos));c['object_toward_rate']=float(np.mean(object_cos>0));c['object_mean_cosine']=float(np.mean(object_cos));c['distance_increase_step_rate']=float(np.mean(d.progress<0))
 rows.append({'config':cfg,'policy_toward_rate':c['policy_toward_rate'],'servo_toward_rate':c['servo_toward_rate'],'final_command_toward_rate':c['final_toward_rate'],'policy_servo_conflict_rate':c['policy_servo_conflict_rate'],'actual_tcp_toward_rate':c['actual_tcp_toward_rate'],'actual_tcp_mean_cosine':c['actual_tcp_mean_cosine'],'object_toward_rate':c['object_toward_rate'],'object_mean_cosine':c['object_mean_cosine'],'distance_increase_step_rate':c['distance_increase_step_rate']})
with (out/'command_direction_analysis.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
flat=[]
for cfg,c in data['configs'].items():flat.append({'config':cfg,**{k:v for k,v in c.items() if k not in ('failures','mode')},'mode':c['mode']})
with (out/'abcd_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=flat[0]);w.writeheader();w.writerows(flat)
(out/'abcd_comparison.json').write_text(json.dumps(data,indent=2))
