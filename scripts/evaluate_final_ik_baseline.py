#!/usr/bin/env python3
import csv, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(os.environ.get("PROJECT_ROOT",Path(__file__).resolve().parents[1]));sys.path.insert(0,str(ROOT))
import gymnasium as gym
from stable_baselines3 import PPO
import fourc2
OUT=Path(os.environ.get("FINAL_IK_OUT",ROOT/"results/final_ik_baseline"));OUT.mkdir(parents=True,exist_ok=True)
ENV="My4C2AllStageSinglePPOV22Cube3cm-v0"; MODEL=ROOT/"checkpoints/best_full_flow_v22.zip"

def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def failure(r):
    if r["full_success"]: return "success"
    if not r["reach_success"]: return "reach"
    if not r["grasp_success"]: return "grasp"
    if not r["lift_success"]: return "lift"
    if not r["entered_place"]: return "place_not_entered"
    if not r["reached_xy_release_region"]: return "place_xy"
    if not r["reached_low_height"]: return "place_height"
    if not r["place_open_ready"]: return "place_open_ready"
    if not r["gripper_opened"]: return "gripper_open"
    return "place_terminal"
def main():
    model=PPO.load(MODEL,device="cpu"); rows=[]; tcp=defaultdict(list)
    for seed in range(60):
      env=gym.make(ENV);e=env.unwrapped;obs,info=env.reset(seed=seed);done=False;steps=0
      reach=grasp=lift=entered=xyready=lowready=openready=opened=False; minxy=np.inf; rev=nplace=dqclip=ikiter=0
      minmargin=np.inf;minmarginj=np.full(6,np.inf);maxqvel=maxforce=0.;table=0;maxpenetration=0.;dropped=flung=False
      initial_object=e.data.site_xpos[e.object_site_id].copy();initial_goal=e.data.site_xpos[e.goal_site_id].copy()
      while not done:
        action,_=model.predict(obs,deterministic=True);obs,_,t,tr,info=env.step(action);done=t or tr;steps+=1
        reach|=bool(info.get("reach_success"));grasp|=bool(info.get("grasp_success"));lift|=bool(info.get("lift_success"))
        q=e.data.qpos[e.arm_qpos_ids];m=np.minimum(q-e.arm_ctrl_low,e.arm_ctrl_high-q);minmargin=min(minmargin,float(m.min()));minmarginj=np.minimum(minmarginj,m)
        maxqvel=max(maxqvel,float(np.max(np.abs(e.data.qvel[e.arm_qvel_ids]))));maxforce=max(maxforce,float(np.max(np.abs(e.data.actuator_force[e.arm_actuator_ids]))));table+=int(info.get("table_contact_count",0)>0);maxpenetration=max(maxpenetration,float(info.get("pad_object_penetration",0)))
        dropped|=bool(info["object_position"][2]<e.table_top_z+e.object_half_size-.01);flung|=bool(info.get("object_speed",0)>.5)
        dqclip+=e.diag_ik["dq_clip_iterations"];ikiter+=e.diag_ik["iterations"]
        if int(e.diag_stage_before)==5:
          entered=True;xy=float(info["object_to_goal_xy_distance"]);minxy=min(minxy,xy);xyready|=bool(info["place_xy_ready"]);lowready|=bool(info["place_low_ready"]);openready|=bool(info["place_open_ready"]);opened|=bool(info["place_opened"] or info["place_has_opened"]);tcp["all"].append(float(info["tcp_target_error"]))
          cmd=e.diag_target_after_lead_clip[:2]-e.diag_tcp_actual_before[:2];fkd=e.diag_ik["fk_target_tcp"][:2]-e.diag_tcp_actual_before[:2];g=np.asarray(info["goal_position"][:2])-e.diag_object_before[:2]
          if np.linalg.norm(cmd)>1e-9:nplace+=1;rev+=int(cos(g,cmd)>0 and cos(g,fkd)<0)
      r={"seed":seed,"full_success":bool(info.get("is_success")),"place_success":bool(info.get("place_success")),"reach_success":reach,"grasp_success":grasp,"lift_success":lift,"entered_place":entered,"reached_xy_release_region":xyready,"reached_low_height":lowready,"place_open_ready":openready,"gripper_opened":opened,"final_goal_xy":float(info.get("object_to_goal_xy_distance",np.nan)),"min_goal_xy":None if not np.isfinite(minxy) else minxy,"episode_steps":steps,"dq_clip_rate":dqclip/max(ikiter,1),"command_fk_reverse_rate":rev/max(nplace,1),"min_joint_limit_margin":minmargin,**{f"joint_{i}_min_margin":minmarginj[i] for i in range(6)},"max_joint_velocity":maxqvel,"max_actuator_force":maxforce,"robot_table_contact_steps":table,"max_pad_object_penetration":maxpenetration,"object_dropped":dropped,"object_fling":flung,"initial_object_x":initial_object[0],"initial_object_y":initial_object[1],"initial_goal_x":initial_goal[0],"initial_goal_y":initial_goal[1]};r["failure_stage"]=failure(r);rows.append(r);env.close();print(seed,r["full_success"],r["failure_stage"],r["final_goal_xy"])
    with (OUT/"episodes.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    a=lambda k:[r[k] for r in rows];p=np.asarray(tcp["all"])
    summary={"configuration":{"episodes":60,"seeds":list(range(60)),"checkpoint":str(MODEL),"max_tcp_lead":.03,"arm_kp_scale":2.,"posture_mode":"off","approach_axis_weight":.35},"success":{"full":sum(a("full_success")),"place":sum(a("place_success")),"reach":sum(a("reach_success")),"grasp":sum(a("grasp_success")),"lift":sum(a("lift_success")),"entered_place":sum(a("entered_place")),"xy_release":sum(a("reached_xy_release_region")),"low_height":sum(a("reached_low_height")),"open_ready":sum(a("place_open_ready")),"opened":sum(a("gripper_opened"))},"failure_counts":dict(Counter(a("failure_stage"))),"metrics":{"final_goal_xy_mean":float(np.mean(a("final_goal_xy"))),"final_goal_xy_p95":float(np.percentile(a("final_goal_xy"),95)),"final_goal_xy_max":float(np.max(a("final_goal_xy"))),"min_goal_xy_mean":float(np.mean(a("min_goal_xy"))),"place_tcp_error_mean":float(p.mean()),"place_tcp_error_p95":float(np.percentile(p,95)),"place_tcp_error_max":float(p.max()),"dq_clip_rate_mean":float(np.mean(a("dq_clip_rate"))),"command_fk_reverse_rate_mean":float(np.mean(a("command_fk_reverse_rate"))),"joint_limit_margin_min":float(np.min(a("min_joint_limit_margin"))),"joint_limit_margin_by_joint_min":[float(min(a(f"joint_{i}_min_margin"))) for i in range(6)],"max_joint_velocity":float(np.max(a("max_joint_velocity"))),"max_actuator_force":float(np.max(a("max_actuator_force"))),"robot_table_contact_steps":int(np.sum(a("robot_table_contact_steps"))),"max_pad_object_penetration":float(np.max(a("max_pad_object_penetration"))),"object_drop_episodes":int(np.sum(a("object_dropped"))),"object_fling_episodes":int(np.sum(a("object_fling"))),"episode_steps_mean":float(np.mean(a("episode_steps"))),"episode_steps_max":int(np.max(a("episode_steps")))}}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
