#!/usr/bin/env python3
import csv,json,os,sys,copy
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(os.environ.get('PROJECT_ROOT',Path(__file__).resolve().parents[1]));sys.path.insert(0,str(ROOT))
import gymnasium as gym,mujoco
from stable_baselines3 import PPO
import fourc2
OUT=Path(os.environ.get('LAYER_OUT',ROOT/'results/place_ik_realization'));OUT.mkdir(parents=True,exist_ok=True)
MODEL=ROOT/'checkpoints/best_full_flow_v22.zip';ENV='My4C2AllStageSinglePPOV22Cube3cm-v0'
VARIANTS=[('current',.35,.02),('position_only',0.,0.),('no_posture',.35,0.),('no_axis',0.,.02)]

def kp2(e):
 ids=e.arm_actuator_ids;e.model.actuator_gainprm[ids,0]*=2;e.model.actuator_biasprm[ids,1]*=2
def cos(a,b):return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))
def classify(c_cmd,c_fk,c_act,c_obj,ncmd):
 if ncmd<1e-9:return 'unclassified_zero_cmd'
 if c_cmd>0 and c_fk<0:return 'A_cmd_to_fk'
 if c_cmd>0 and c_fk>0 and c_act<0:return 'B_fk_to_actual'
 if c_cmd>0 and c_fk>0 and c_act>0 and c_obj<0:return 'C_actual_to_object'
 if c_cmd>0 and c_fk>0 and c_act>0 and c_obj>0:return 'D_all_toward'
 return 'unclassified'
def state(e,label,place_step,dist):
 names=['stage','is_grasp_latched','grasp_object_offset','latched_object_xy','gripper_command_normalized','release_steps','release_has_opened','place_descent_active','place_handoff_count','tcp_target_pos','tcp_target_quat']
 return {'label':label,'place_step':place_step,'distance':dist,'qpos':e.data.qpos.copy(),'qvel':e.data.qvel.copy(),'ctrl':e.data.ctrl.copy(),'mocap_pos':e.data.mocap_pos.copy(),'mocap_quat':e.data.mocap_quat.copy(),'attrs':{n:copy.deepcopy(getattr(e,n)) for n in names}}
def restore(e,s):
 e.data.qpos[:]=s['qpos'];e.data.qvel[:]=s['qvel'];e.data.ctrl[:]=s['ctrl'];e.data.mocap_pos[:]=s['mocap_pos'];e.data.mocap_quat[:]=s['mocap_quat']
 for n,v in s['attrs'].items():setattr(e,n,copy.deepcopy(v))
 mujoco.mj_forward(e.model,e.data)

def main():
 model=PPO.load(MODEL,device='cpu');rows=[];seed_rows=[];snap_series=[]
 for seed in range(10):
  env=gym.make(ENV,max_tcp_lead=.03);e=env.unwrapped;kp2(e);obs,info=env.reset(seed=seed);done=False;epstep=pstep=0;counts=Counter();first={};local=[]
  while not done:
   action,_=model.predict(obs,deterministic=True);obs,r,t,tr,info=env.step(action);done=t or tr;epstep+=1
   if int(e.diag_stage_before)!=5:continue
   pstep+=1;before=e.diag_tcp_actual_before[:2];objb=e.diag_object_before[:2];target=e.diag_target_after_lead_clip[:2];fk=e.diag_ik['fk_target_tcp'][:2];after=e.diag_tcp_actual_after[:2];obja=e.diag_object_after[:2];goal=np.asarray(info['goal_position'][:2]);g=goal-objb
   cmd=target-before;fkd=fk-before;act=after-before;obj=obja-objb;cc=cos(g,cmd);cf=cos(g,fkd);ca=cos(g,act);co=cos(g,obj);label=classify(cc,cf,ca,co,np.linalg.norm(cmd));counts[label]+=1
   if label not in first:first[label]=pstep
   row={'seed':seed,'episode_step':epstep,'place_step':pstep,'class':label,
    **{f'tcp_before_{k}':e.diag_tcp_actual_before[i] for i,k in enumerate('xyz')},**{f'object_before_{k}':e.diag_object_before[i] for i,k in enumerate('xyz')},
    **{f'tcp_target_{k}':e.diag_target_after_lead_clip[i] for i,k in enumerate('xyz')},**{f'fk_qtarget_tcp_{k}':e.diag_ik['fk_target_tcp'][i] for i,k in enumerate('xyz')},
    **{f'tcp_after_{k}':e.diag_tcp_actual_after[i] for i,k in enumerate('xyz')},**{f'object_after_{k}':e.diag_object_after[i] for i,k in enumerate('xyz')},
    'qpos_before':json.dumps(e.diag_qpos_before.tolist()),'q_target':json.dumps(e.diag_q_target.tolist()),'qpos_after':json.dumps(e.diag_qpos_after.tolist()),
    'ik_converged':e.diag_ik['converged'],'ik_iterations':e.diag_ik['iterations'],'dq_clip_iterations':e.diag_ik['dq_clip_iterations'],
    'cmd_dx':cmd[0],'cmd_dy':cmd[1],'fk_dx':fkd[0],'fk_dy':fkd[1],'actual_dx':act[0],'actual_dy':act[1],'object_dx':obj[0],'object_dy':obj[1],'goal_dx':g[0],'goal_dy':g[1],
    'cos_goal_cmd':cc,'cos_goal_fk':cf,'cos_goal_actual':ca,'cos_goal_object':co,'cos_cmd_fk':cos(cmd,fkd),'cos_fk_actual':cos(fkd,act),'cos_actual_object':cos(act,obj),
    'cmd_norm':np.linalg.norm(cmd),'fk_norm':np.linalg.norm(fkd),'actual_norm':np.linalg.norm(act),'object_norm':np.linalg.norm(obj),'qgap_before':np.linalg.norm(e.diag_q_target-e.diag_qpos_before),'qgap_after':np.linalg.norm(e.diag_q_target-e.diag_qpos_after),'goal_distance':np.linalg.norm(g)}
   rows.append(row);local.append(row)
   if seed==0:snap_series.append(state(e,f'step_{pstep}',pstep,row['goal_distance']))
  total=sum(counts.values());seed_rows.append({'seed':seed,'place_steps':total,**{k:counts[k] for k in ('A_cmd_to_fk','B_fk_to_actual','C_actual_to_object','D_all_toward','unclassified','unclassified_zero_cmd')},**{f'{k}_rate':counts[k]/max(total,1) for k in ('A_cmd_to_fk','B_fk_to_actual','C_actual_to_object','D_all_toward')},'first_reverse_step':min([v for k,v in first.items() if k.startswith(('A_','B_','C_'))],default=None),'first_reverse_layer':next((k for k,v in sorted(first.items(),key=lambda x:x[1]) if k.startswith(('A_','B_','C_'))),None)})
  env.close();print(seed,dict(counts))
 # seed 0 snapshots: entry, minimum-distance turn, final far state.
 distances=np.asarray([s['distance'] for s in snap_series]);away=np.flatnonzero(distances>distances[0]+.05);pre_away=max(1,int(away[0])-1) if len(away) else max(1,len(snap_series)//2);idx=[0,pre_away,len(snap_series)-1];snaps=[]
 for label,i in zip(('place_entry','before_away','far_after'),idx):s=copy.deepcopy(snap_series[i]);s['label']=label;snaps.append(s)
 snaprows=[]
 for s in snaps:
  for variant,aw,pw in VARIANTS:
   for mm in (5,10,12):
    e=gym.make(ENV,max_tcp_lead=.03,ik_axis_weight=aw,ik_posture_weight=pw).unwrapped;kp2(e);e.reset(seed=0);restore(e,s);tcp=e.data.site_xpos[e.pinch_site_id].copy();obj=e.data.site_xpos[e.object_site_id].copy();goal=e.data.site_xpos[e.goal_site_id].copy();direction=(goal-obj)[:2];direction/=np.linalg.norm(direction)+1e-12;target=tcp.copy();target[:2]+=direction*mm/1000
    if aw==0:e.desired_approach_axis=e._pinch_approach_axis(e.data)
    qtarget=e._solve_ik(target);fk=e.diag_ik['fk_target_tcp'].copy();e.data.ctrl[e.arm_actuator_ids]=qtarget
    for sub in range(e.frame_skip):mujoco.mj_step(e.model,e.data);e._update_grasp_latch(update_counter=sub==e.frame_skip-1)
    actual=e.data.site_xpos[e.pinch_site_id].copy();obja=e.data.site_xpos[e.object_site_id].copy();g=(goal-obj)[:2];cmd=(target-tcp)[:2];fkd=(fk-tcp)[:2];act=(actual-tcp)[:2];od=(obja-obj)[:2]
    snaprows.append({'snapshot':s['label'],'source_place_step':s['place_step'],'source_goal_distance':s['distance'],'variant':variant,'command_mm':mm,'axis_weight':aw,'posture_weight':pw,'cmd_dx':cmd[0],'cmd_dy':cmd[1],'fk_dx':fkd[0],'fk_dy':fkd[1],'actual_dx':act[0],'actual_dy':act[1],'object_dx':od[0],'object_dy':od[1],'cos_goal_cmd':cos(g,cmd),'cos_goal_fk':cos(g,fkd),'cos_goal_actual':cos(g,act),'cos_goal_object':cos(g,od),'cos_cmd_fk':cos(cmd,fkd),'cos_fk_actual':cos(fkd,act),'ik_converged':e.diag_ik['converged'],'ik_iterations':e.diag_ik['iterations'],'dq_clip_iterations':e.diag_ik['dq_clip_iterations'],'qgap_before':np.linalg.norm(qtarget-s['attrs']['tcp_target_pos'][:0]) if False else np.linalg.norm(qtarget-e.diag_ik['initial_qpos']),'qgap_after':np.linalg.norm(qtarget-e.data.qpos[e.arm_qpos_ids])})
    e.close()
 with (OUT/'step_layer_decomposition.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 with (OUT/'seed_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=seed_rows[0]);w.writeheader();w.writerows(seed_rows)
 with (OUT/'snapshot_tests.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=snaprows[0]);w.writeheader();w.writerows(snaprows)
 totals=Counter(r['class'] for r in rows);summary={'total_place_steps':len(rows),'counts':dict(totals),'rates':{k:totals[k]/len(rows) for k in totals},'first_reverse_by_seed':{str(r['seed']):{'step':r['first_reverse_step'],'layer':r['first_reverse_layer']} for r in seed_rows},'snapshots':[{'label':s['label'],'place_step':s['place_step'],'goal_distance':s['distance']} for s in snaps]}
 (OUT/'classification_summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
