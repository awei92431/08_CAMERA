#!/usr/bin/env python3
import csv,json,os,sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(os.environ.get('PROJECT_ROOT',Path(__file__).resolve().parents[1]));sys.path.insert(0,str(ROOT))
import gymnasium as gym
from stable_baselines3 import PPO
import fourc2
OUT=Path(os.environ.get('PLACE_OUT',ROOT/'results/place_control_diagnosis'));OUT.mkdir(parents=True,exist_ok=True)
MODEL=ROOT/'checkpoints/best_full_flow_v22.zip';ENV='My4C2AllStageSinglePPOV22Cube3cm-v0'
CONFIGS=[('A_current','combined'),('B_servo_only','servo_only'),('C_policy_only','policy_only'),('D_oracle','oracle')]

def scale_kp(raw):
 ids=raw.arm_actuator_ids;raw.model.actuator_gainprm[ids,0]*=2.;raw.model.actuator_biasprm[ids,1]*=2.
def cosine(a,b):return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))
def classify(e):
 if e['place_success']:return 'success'
 if not e['entered_place']:return 'pre_place_failure'
 if not e['reached_xy_release_region']:return '未到达释放区域'
 if not e['reached_low_height']:return '未下降到释放高度'
 if not e['place_open_ready']:return '未触发释放条件'
 if not e['gripper_opened']:return '释放条件满足但夹爪未打开'
 if not e['object_released']:return '夹爪已打开但物体未正确落下'
 if not e['post_release_stable']:return '物体落下后不稳定'
 return 'episode超时'

def main():
 model=PPO.load(MODEL,device='cpu');logs=[];eps=[];configs={}
 for cfg,mode in CONFIGS:
  for seed in range(10):
   env=gym.make(ENV,max_tcp_lead=.03,place_xy_control_mode=mode,place_oracle_xy_gain=.30);raw=env.unwrapped;scale_kp(raw);obs,info=env.reset(seed=seed);done=False;step=place_step=0;prev_dist=None;initial_dist=None
   entered=xyok=lowok=ready=opened=released=False;release_stable=[];min_dist=np.inf;positive=negative=policy_pos=servo_pos=final_pos=conflict=policy_n=servo_n=final_n=0;first_away=None;place_errors=[];robot_table=drop=fling=bounce=False;prev_obj=None
   while not done:
    action,_=model.predict(obs,deterministic=True);obs,r,t,tr,info=env.step(action);done=t or tr;step+=1
    if int(raw.diag_stage_before)!=5:continue
    entered=True;place_step+=1;obj=np.asarray(info['object_position'][:2]);goal=np.asarray(info['goal_position'][:2]);desired=goal-obj;dist=float(info['object_to_goal_xy_distance']);min_dist=min(min_dist,dist)
    if initial_dist is None:initial_dist=dist
    progress=0. if prev_dist is None else prev_dist-dist;prev_dist=dist;obj_step=np.zeros(2) if prev_obj is None else obj-prev_obj;prev_obj=obj.copy()
    policy=np.asarray(raw.diag_place_policy_xy);servo=np.asarray(raw.diag_place_servo_xy);final=np.asarray(raw.diag_place_final_delta_xy)
    cp=cosine(desired,policy);cs=cosine(desired,servo);cf=cosine(desired,final);policy_n+=int(np.linalg.norm(policy)>1e-9);servo_n+=int(np.linalg.norm(servo)>1e-9);final_n+=int(np.linalg.norm(final)>1e-9);policy_pos+=int(cp>0);servo_pos+=int(cs>0);final_pos+=int(cf>0);positive+=int(cf>0);negative+=int(cf<0);conflict+=int(np.dot(policy,servo)<0)
    if first_away is None and dist>initial_dist+.05:first_away=place_step
    xy=bool(info['place_xy_ready']);low=bool(info['place_low_ready']);delay=bool(info['release_steps']>=raw.release_min_open_steps);openready=bool(info['place_open_ready']);opening_pred=bool(xy and low and delay);op=bool(info['place_has_opened'] or info['place_opened']);rel=bool(op and not info['is_grasp_latched']);xyok|=xy;lowok|=low;ready|=openready;opened|=op;released|=rel
    if rel:release_stable.append(float(info['object_xy_speed'])<.04)
    robot_table|=bool(info['table_contact_count']>0);drop|=bool(info['object_position'][2]<raw.table_top_z+raw.object_half_size-.01);fling|=bool(info['object_speed']>.5);place_errors.append(float(info['tcp_target_error']))
    logs.append({'config':cfg,'seed':seed,'episode_step':step,'place_step':place_step,'object_x':obj[0],'object_y':obj[1],'goal_x':goal[0],'goal_y':goal[1],'goal_dx':desired[0],'goal_dy':desired[1],'goal_distance':dist,'progress':progress,
     'policy_action_x':action[0],'policy_action_y':action[1],'policy_scaled_dx':policy[0],'policy_scaled_dy':policy[1],'servo_dx':servo[0],'servo_dy':servo[1],'final_command_dx':final[0],'final_command_dy':final[1],
     'policy_cosine':cp,'servo_cosine':cs,'final_cosine':cf,'policy_servo_opposed':bool(np.dot(policy,servo)<0),'raw_target_x':raw.diag_raw_target[0],'raw_target_y':raw.diag_raw_target[1],
     'safe_target_x':raw.diag_safe_target[0],'safe_target_y':raw.diag_safe_target[1],'pre_smooth_x':raw.diag_target_before_smoothing[0],'pre_smooth_y':raw.diag_target_before_smoothing[1],
     'post_smooth_x':raw.diag_target_after_smoothing[0],'post_smooth_y':raw.diag_target_after_smoothing[1],'pre_lead_x':raw.diag_target_before_lead_clip[0],'pre_lead_y':raw.diag_target_before_lead_clip[1],
     'post_lead_x':raw.diag_target_after_lead_clip[0],'post_lead_y':raw.diag_target_after_lead_clip[1],'actual_tcp_x':info['pinch_position'][0],'actual_tcp_y':info['pinch_position'][1],
     'tcp_target_x':info['tcp_target_position'][0],'tcp_target_y':info['tcp_target_position'][1],'target_actual_dx':info['tcp_target_position'][0]-info['pinch_position'][0],'target_actual_dy':info['tcp_target_position'][1]-info['pinch_position'][1],
     'object_step_x':obj_step[0],'object_step_y':obj_step[1],'object_xy_step':info['object_xy_step'],'tcp_error':info['tcp_target_error'],
     'goal_xy_ok':xy,'height_ok':low,'release_delay_ok':delay,'release_steps':info['release_steps'],'release_min_steps':raw.release_min_open_steps,'opening_predicate':opening_pred,'place_open_ready':openready,
     'gripper_opened':op,'grasp_latched':info['is_grasp_latched'],'object_speed_ok':info['object_xy_speed']<.04,'table_ok':info['table_contact_count']==0,'boundary_ok':info['object_table_boundary_penalty']==0,'place_success':info['place_success']})
   if place_errors:a=np.asarray(place_errors);tcp={'mean':float(a.mean()),'p95':float(np.percentile(a,95)),'max':float(a.max())}
   else:tcp={'mean':None,'p95':None,'max':None}
   e={'config':cfg,'seed':seed,'full_success':bool(info['is_success']),'entered_place':entered,'place_success':bool(info['place_success']),'reached_xy_release_region':xyok,'reached_low_height':lowok,'place_open_ready':ready,'gripper_opened':opened,'object_released':released,'post_release_stable':bool(release_stable and all(release_stable[-5:])),
    'min_goal_xy':None if not np.isfinite(min_dist) else min_dist,'final_goal_xy':float(info['object_to_goal_xy_distance']),'place_steps':place_step,'final_cosine_positive_rate':positive/max(final_n,1),'final_cosine_negative_rate':negative/max(final_n,1),
    'policy_toward_rate':policy_pos/max(policy_n,1),'servo_toward_rate':servo_pos/max(servo_n,1),'final_toward_rate':final_pos/max(final_n,1),'policy_servo_conflict_rate':conflict/max(place_step,1),'first_away_step':first_away,
    'distance_increased':bool(initial_dist is not None and info['object_to_goal_xy_distance']>initial_dist),'tcp_error_mean':tcp['mean'],'tcp_error_p95':tcp['p95'],'tcp_error_max':tcp['max'],'robot_table_contact':robot_table,'object_drop':drop,'object_fling':fling,'object_bounce':bounce}
   e['failure_type']=classify(e);eps.append(e);env.close();print(cfg,seed,e['place_success'],e['failure_type'],e['min_goal_xy'],e['final_goal_xy'])
  ce=[e for e in eps if e['config']==cfg];configs[cfg]={'mode':mode,'episodes':10,'full_success':sum(e['full_success'] for e in ce),'entered_place':sum(e['entered_place'] for e in ce),'reached_xy_release_region':sum(e['reached_xy_release_region'] for e in ce),'reached_low_height':sum(e['reached_low_height'] for e in ce),'place_open_ready':sum(e['place_open_ready'] for e in ce),'gripper_opened':sum(e['gripper_opened'] for e in ce),'place_success':sum(e['place_success'] for e in ce),
   'mean_min_goal_xy':float(np.mean([e['min_goal_xy'] for e in ce])),'mean_final_goal_xy':float(np.mean([e['final_goal_xy'] for e in ce])),'mean_place_steps':float(np.mean([e['place_steps'] for e in ce])),'policy_toward_rate':float(np.mean([e['policy_toward_rate'] for e in ce])),'servo_toward_rate':float(np.mean([e['servo_toward_rate'] for e in ce])),'final_toward_rate':float(np.mean([e['final_toward_rate'] for e in ce])),'policy_servo_conflict_rate':float(np.mean([e['policy_servo_conflict_rate'] for e in ce])),'tcp_error_mean':float(np.mean([e['tcp_error_mean'] for e in ce])),'tcp_error_p95':float(np.mean([e['tcp_error_p95'] for e in ce])),'tcp_error_max':max(e['tcp_error_max'] for e in ce),'failures':dict(Counter(e['failure_type'] for e in ce))}
 fields=logs[0].keys()
 with (OUT/'step_logs.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(logs)
 with (OUT/'episode_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=eps[0]);w.writeheader();w.writerows(eps)
 dirrows=[{'config':k,**{x:v for x,v in d.items() if x in ('policy_toward_rate','servo_toward_rate','final_toward_rate','policy_servo_conflict_rate')}} for k,d in configs.items()]
 with (OUT/'command_direction_analysis.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=dirrows[0]);w.writeheader();w.writerows(dirrows)
 relfields=['config','seed','reached_xy_release_region','reached_low_height','place_open_ready','gripper_opened','object_released','post_release_stable','failure_type'];relrows=[{k:e[k] for k in relfields} for e in eps]
 with (OUT/'release_condition_analysis.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=relfields);w.writeheader();w.writerows(relrows)
 rows=[{k:v for k,v in d.items() if k!='failures'} for d in configs.values()]
 with (OUT/'abcd_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 (OUT/'abcd_comparison.json').write_text(json.dumps({'configs':configs,'episodes':eps},indent=2))
if __name__=='__main__':main()
