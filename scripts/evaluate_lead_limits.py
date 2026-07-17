#!/usr/bin/env python3
import csv,json,os,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(os.environ.get('PROJECT_ROOT',Path(__file__).resolve().parents[1]));sys.path.insert(0,str(ROOT))
import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import PPO
import fourc2
OUT=Path(os.environ.get('LEAD_OUT',ROOT/'results/lead_limit'));OUT.mkdir(parents=True,exist_ok=True)
MODEL=ROOT/'checkpoints/best_full_flow_v22.zip';ENV='My4C2AllStageSinglePPOV22Cube3cm-v0'
CONFIGS=[('baseline',None),('lead_20mm',.020),('lead_25mm',.025),('lead_30mm',.030)]
NAMES={0:'reach',1:'grasp',2:'lift',5:'place'}

def stats(a):
 a=np.asarray(a,float);return {'mean':float(a.mean()),'p95':float(np.percentile(a,95)),'max':float(a.max()),'samples':len(a)}
def main():
 model=PPO.load(MODEL,device='cpu');comparisons=[];allout={}
 for name,limit in CONFIGS:
  folder=OUT/name;folder.mkdir(exist_ok=True);episodes=[];stage_errors=defaultdict(list)
  for seed in range(10):
   video=seed==0;env=gym.make(ENV,render_mode='rgb_array' if video else None,max_tcp_lead=limit);raw=env.unwrapped
   if video:raw.mujoco_renderer.width=960;raw.mujoco_renderer.height=720
   obs,info=env.reset(seed=seed);frames=[env.render()] if video else [];done=False;steps=0
   ever={k:False for k in ('reach','grasp','lift')};entered_place=False;place_success=False;dqclips=0;jclips=0
   while not done:
    action,_=model.predict(obs,deterministic=True);obs,r,term,trunc,info=env.step(action);done=term or trunc;steps+=1
    stage=NAMES.get(int(raw.stage),str(raw.stage));entered_place|=stage=='place';place_success|=bool(info.get('place_success',False))
    for k in ever:ever[k]|=bool(info.get(f'{k}_success',False))
    err=float(np.linalg.norm(raw.tcp_target_pos-raw.data.site_xpos[raw.pinch_site_id]));stage_errors[stage].append(err)
    dqclips+=int(raw.diag_ik['dq_clip']);jclips+=int(raw.diag_ik['joint_limit_clip'])
    if video:frames.append(env.render())
   if video:imageio.mimsave(folder/f'{name}_seed_0.mp4',frames,fps=20,macro_block_size=1)
   episodes.append({'seed':seed,'full_success':bool(info.get('is_success',False)),'reach_ever':ever['reach'],'grasp_ever':ever['grasp'],
    'lift_ever':ever['lift'],'entered_place':entered_place,'place_success':place_success,'steps':steps,'dq_clip_count':dqclips,
    'joint_limit_clip_count':jclips,'lead_clip_count':raw.lead_clip_count,'max_raw_lead':raw.max_raw_tcp_lead,
    'max_clipped_lead':raw.max_clipped_tcp_lead})
   env.close();print(name,seed,steps,entered_place,place_success,raw.max_raw_tcp_lead,raw.max_clipped_tcp_lead)
  stage={k:stats(stage_errors[k]) for k in ('reach','grasp','lift','place')}
  def count(k):return int(sum(e[k] for e in episodes))
  summary={'config':name,'max_tcp_lead':limit,'episodes':10,'full_success':count('full_success'),'reach_ever':count('reach_ever'),
   'grasp_ever':count('grasp_ever'),'lift_ever':count('lift_ever'),'entered_place':count('entered_place'),'place_success':count('place_success'),
   'mean_episode_length':float(np.mean([e['steps'] for e in episodes])),'dq_clip_count':count('dq_clip_count'),
   'joint_limit_clip_count':count('joint_limit_clip_count'),'lead_clip_count':count('lead_clip_count'),
   'max_raw_lead':max(e['max_raw_lead'] for e in episodes),'max_clipped_lead':max(e['max_clipped_lead'] for e in episodes),'stage_tcp_error':stage}
  (folder/'evaluation.json').write_text(json.dumps({'summary':summary,'episodes':episodes},indent=2))
  with (folder/'episodes.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=episodes[0]);w.writeheader();w.writerows(episodes)
  flat={k:v for k,v in summary.items() if k!='stage_tcp_error'}
  for st,d in stage.items():
   for metric in ('mean','p95','max'):flat[f'{st}_{metric}']=d[metric]
  comparisons.append(flat);allout[name]={'summary':summary,'episodes':episodes}
 with (OUT/'comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=comparisons[0]);w.writeheader();w.writerows(comparisons)
 (OUT/'comparison.json').write_text(json.dumps(allout,indent=2))
if __name__=='__main__':main()
