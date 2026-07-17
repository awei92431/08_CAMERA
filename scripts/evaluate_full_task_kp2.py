#!/usr/bin/env python3
import csv,json,os,sys
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
ROOT=Path(os.environ.get('PROJECT_ROOT',Path(__file__).resolve().parents[1]));sys.path.insert(0,str(ROOT))
import gymnasium as gym
from stable_baselines3 import PPO
import fourc2
OUT=Path(os.environ.get('FULL_OUT',ROOT/'results/full_task_kp2'));OUT.mkdir(parents=True,exist_ok=True)
MODEL=ROOT/'checkpoints/best_full_flow_v22.zip';ENV='My4C2AllStageSinglePPOV22Cube3cm-v0';NAMES={0:'reach',1:'grasp',2:'lift',5:'place'}

def scale_kp(raw):
 ids=raw.arm_actuator_ids;raw.model.actuator_gainprm[ids,0]*=2.;raw.model.actuator_biasprm[ids,1]*=2.
def classify(e):
 if e['full_success']:return 'success'
 if not e['entered_place']:return 'pre_place_failure'
 if not e['ever_place_xy_ready']:return '未到达释放区域'
 if not e['ever_place_low_ready']:return '未下降到释放高度'
 if not e['ever_place_open_ready'] and not e['gripper_opened']:return '未触发释放条件'
 if e['gripper_opened'] and not e['object_released']:return '夹爪已打开但物体未正确落下'
 if e['object_released'] and e['final_goal_xy']>=e['place_xy_threshold']:return '物体落下但位置超出成功阈值'
 if e['object_released'] and not e['post_release_stable']:return '物体落下后不稳定'
 return 'episode超时'
def stats(a):
 a=np.asarray(a,float);return {'mean':float(a.mean()),'p95':float(np.percentile(a,95)),'max':float(a.max()),'samples':len(a)}

def main():
 model=PPO.load(MODEL,device='cpu');episodes=[];stage_errors=defaultdict(list)
 for seed in range(10):
  env=gym.make(ENV,max_tcp_lead=.03);raw=env.unwrapped;scale_kp(raw);obs,info=env.reset(seed=seed);done=False;steps=0
  ever={k:False for k in ('reach','grasp','lift')};entered=False;xyready=lowready=openready=opened=released=False;release_speeds=[];post_z=[]
  min_grasp_xy=np.inf;max_grasp_drift=0.;max_qvel=max_force=0.;sat=force_samples=dqclips=dqiters=jclips=robot_table=0;last_place=None
  while not done:
   action,_=model.predict(obs,deterministic=True);obs,r,t,tr,info=env.step(action);done=t or tr;steps+=1;stage=NAMES.get(int(raw.stage),str(raw.stage))
   for k in ever:ever[k]|=bool(info.get(f'{k}_success',False))
   err=float(info.get('tcp_target_error',0));stage_errors[stage].append(err)
   if stage=='grasp':min_grasp_xy=min(min_grasp_xy,float(info['grasp_xy_error']));max_grasp_drift=max(max_grasp_drift,float(info['object_horizontal_drift']))
   if stage=='place':
    entered=True;last_place=dict(info);xyready|=bool(info['place_xy_ready']);lowready|=bool(info['place_low_ready']);openready|=bool(info['place_open_ready']);opened|=bool(info['place_has_opened'] or info['place_opened']);released|=bool(opened and not info['is_grasp_latched'])
    if released:release_speeds.append(float(info['object_speed']));post_z.append(float(info['object_position'][2]))
   max_qvel=max(max_qvel,float(np.max(np.abs(raw.data.qvel[raw.arm_qvel_ids]))));forces=raw.data.actuator_force[raw.arm_actuator_ids];max_force=max(max_force,float(np.max(np.abs(forces))))
   ranges=raw.model.actuator_forcerange[raw.arm_actuator_ids];sat+=int(np.sum(np.abs(forces)>=.99*np.max(np.abs(ranges),axis=1)));force_samples+=6
   dqclips+=int(raw.diag_ik['dq_clip_iterations']);dqiters+=int(raw.diag_ik['iterations']);jclips+=int(raw.diag_ik['joint_limit_clip']);robot_table+=int(info['table_contact_count']>0)
  p=last_place or info;final_speed=float(p.get('object_speed',0));stable=bool(released and len(release_speeds)>=5 and max(release_speeds[-5:])<.04)
  bounce=bool(post_z and max(post_z)-min(post_z)>.015);dropped=bool(released and (p['object_position'][2]<raw.table_top_z+raw.object_half_size-.01));flung=bool(max(release_speeds or [0])>.5)
  e={'seed':seed,'full_success':bool(info.get('is_success',False)),'reach_success':ever['reach'],'grasp_success':ever['grasp'],'lift_success':ever['lift'],'entered_place':entered,'place_success':bool(info.get('place_success',False)),
   'steps':steps,'final_goal_xy':float(p.get('object_to_goal_xy_distance',np.nan)),'final_height_error':abs(float(p.get('object_lift',0))-raw.release_success_lift),
   'ever_place_xy_ready':xyready,'ever_place_low_ready':lowready,'ever_place_open_ready':openready,'gripper_opened':opened,'object_released':released,'post_release_stable':stable,
   'final_object_speed':final_speed,'grasp_min_xy_error':None if not np.isfinite(min_grasp_xy) else min_grasp_xy,'grasp_max_horizontal_drift':max_grasp_drift,
   'max_joint_velocity':max_qvel,'max_actuator_force':max_force,'actuator_saturation_rate':sat/force_samples,'dq_clip_count':dqclips,'dq_clip_rate':dqclips/dqiters,
   'joint_limit_clip_count':jclips,'lead_clip_count':raw.lead_clip_count,'robot_table_contact_steps':robot_table,'object_dropped_below_table':dropped,'object_fling':flung,'post_release_bounce':bounce,
   'place_xy_threshold':raw.place_handoff_xy_threshold,'release_height_threshold':raw.release_success_lift}
  e['failure_category']=classify(e);episodes.append(e);env.close();print(seed,e['full_success'],e['failure_category'],steps,e['final_goal_xy'],e['final_height_error'])
 stage={k:stats(stage_errors[k]) for k in ('reach','grasp','lift','place')}
 count=lambda k:int(sum(bool(e[k]) for e in episodes))
 summary={'configuration':{'max_tcp_lead':.03,'kp_scale':2.0,'checkpoint':str(MODEL),'seeds':list(range(10))},'episodes':10,'full_success':count('full_success'),'reach_success':count('reach_success'),'grasp_success':count('grasp_success'),'lift_success':count('lift_success'),'entered_place':count('entered_place'),'place_success':count('place_success'),
  'mean_steps':float(np.mean([e['steps'] for e in episodes])),'stage_tcp_error':stage,'mean_final_goal_xy':float(np.nanmean([e['final_goal_xy'] for e in episodes])),'mean_final_height_error':float(np.nanmean([e['final_height_error'] for e in episodes])),
  'max_joint_velocity':max(e['max_joint_velocity'] for e in episodes),'max_actuator_force':max(e['max_actuator_force'] for e in episodes),'actuator_saturation_rate':sum(e['actuator_saturation_rate'] for e in episodes)/10,
  'dq_clip_count':sum(e['dq_clip_count'] for e in episodes),'joint_limit_clip_count':sum(e['joint_limit_clip_count'] for e in episodes),'lead_clip_count':sum(e['lead_clip_count'] for e in episodes),'robot_table_contact_steps':sum(e['robot_table_contact_steps'] for e in episodes)}
 summary['dq_clip_rate']=float(sum(e['dq_clip_count'] for e in episodes)/sum((e['dq_clip_count']/e['dq_clip_rate']) if e['dq_clip_rate']>0 else 0 for e in episodes))
 failure={'counts':dict(Counter(e['failure_category'] for e in episodes)),'episodes':[{'seed':e['seed'],'category':e['failure_category'],'final_goal_xy':e['final_goal_xy'],'final_height_error':e['final_height_error'],'xy_ready':e['ever_place_xy_ready'],'low_ready':e['ever_place_low_ready'],'open_ready':e['ever_place_open_ready'],'opened':e['gripper_opened'],'released':e['object_released'],'stable':e['post_release_stable']} for e in episodes]}
 with (OUT/'episodes.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=episodes[0]);w.writeheader();w.writerows(episodes)
 (OUT/'summary.json').write_text(json.dumps({'summary':summary,'episodes':episodes},indent=2));(OUT/'failure_analysis.json').write_text(json.dumps(failure,indent=2))
if __name__=='__main__':main()
