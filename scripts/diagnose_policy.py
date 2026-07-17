#!/usr/bin/env python3
import csv,json,sys,os
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
ROOT=Path(os.environ.get('PROJECT_ROOT',Path(__file__).resolve().parents[1]));sys.path.insert(0,str(ROOT))
import gymnasium as gym
from stable_baselines3 import PPO
import fourc2

OUT=Path(os.environ.get('DIAG_OUT',ROOT/'results/diagnostics')); OUT.mkdir(parents=True,exist_ok=True)
MODEL=ROOT/'checkpoints/best_full_flow_v22.zip'; ENV='My4C2AllStageSinglePPOV22Cube3cm-v0'
NAMES={0:'Reach',1:'Grasp',2:'Lift',5:'Place'}

def serial(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,(np.bool_,)): return bool(x)
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,)): return float(x)
    return x

def main():
 model=PPO.load(MODEL,device='cpu'); samples=defaultdict(list); events=[]; episodes=[]
 for seed in range(10):
  env=gym.make(ENV).unwrapped;obs,info=env.reset(seed=seed);maxev=None;step=0;entered_place=False
  prev_stage=int(env.stage);prev_target=env.tcp_target_pos.copy();transition_jumps=[];error_series=[]
  done=False
  while not done and step < 900:
   action,_=model.predict(obs,deterministic=True);action=np.asarray(action,np.float32)
   obs,r,term,trunc,info=env.step(action);step+=1;done=term or trunc
   stage=NAMES.get(int(env.stage),str(env.stage));entered_place |= int(env.stage)==5
   actual=env.data.site_xpos[env.pinch_site_id].copy();target=env.tcp_target_pos.copy();err=target-actual;norm=float(np.linalg.norm(err))
   samples[stage].append(norm);error_series.append(norm)
   if int(env.stage)!=prev_stage:
    transition_jumps.append({'step':step,'from':NAMES.get(prev_stage,str(prev_stage)),'to':stage,'target_jump':float(np.linalg.norm(target-prev_target)),'error':norm})
   ik=env.diag_ik
   event={'seed':seed,'step':step,'stage':stage,'tcp_position':actual,'tcp_target_position':target,
    'dx':err[0],'dy':err[1],'dz':err[2],'error_norm':norm,'policy_action':action,
    'scaled_tcp_action':env.diag_scaled_tcp_action,'qpos':env.data.qpos[env.arm_qpos_ids].copy(),
    'q_target':env.diag_q_target,'dq':ik['total_dq'],'last_dq':ik['last_dq'],'ik_damping':ik['damping'],
    'ik_iterations':ik['iterations'],'ik_converged':ik['converged'],'dq_clip':ik['dq_clip'],
    'joint_limit_clip':ik['joint_limit_clip'],'workspace_clip':env.diag_workspace_clip,
    'target_actual_distance':norm,'table_contact':bool(info.get('table_contact_count',0)),
    'contact_count':int(env.data.ncon),'q_target_qpos_gap':float(np.linalg.norm(env.diag_q_target-env.data.qpos[env.arm_qpos_ids]))}
   if maxev is None or norm>maxev['error_norm']: maxev=event
   prev_stage=int(env.stage);prev_target=target.copy()
  events.append({k:serial(v) for k,v in maxev.items()})
  episodes.append({'seed':seed,'entered_place':entered_place,'place_success':bool(info.get('place_success',False)),
    'final_stage':NAMES.get(int(env.stage),str(env.stage)),'steps':step,'transition_jumps':transition_jumps,
    'error_first':error_series[0],'error_last':error_series[-1],'error_slope':float(np.polyfit(np.arange(len(error_series)),error_series,1)[0])})
  env.close();print(seed,step,maxev['error_norm'],maxev['stage'])
 rows=[];summary={}
 for stage in ('Reach','Grasp','Lift','Place'):
  a=np.asarray(samples[stage]);d={'stage':stage,'mean':float(a.mean()),'median':float(np.median(a)),'p95':float(np.percentile(a,95)),'max':float(a.max()),'samples':len(a)}
  rows.append(d);summary[stage]=d
 with (OUT/'stage_tcp_error.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 (OUT/'max_error_events.json').write_text(json.dumps(events,indent=2))
 (OUT/'stage_tcp_error_summary.json').write_text(json.dumps({'stages':summary,'episodes':episodes,
   'entered_place_count':sum(e['entered_place'] for e in episodes),'place_success_count':sum(e['place_success'] for e in episodes)},indent=2))
if __name__=='__main__':main()
