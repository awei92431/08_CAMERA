#!/usr/bin/env python3
import copy, csv, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
import gymnasium as gym
import mujoco
from stable_baselines3 import PPO
import fourc2

OUT = Path(os.environ.get("NULLSPACE_OUT", ROOT / "results/nullspace_posture"))
OUT.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ROOT / "checkpoints/best_full_flow_v22.zip"
ENV_ID = "My4C2AllStageSinglePPOV22Cube3cm-v0"
MODES = ("raw", "off", "nullspace")
STAGE_PLACE = 5

def kp2(e):
    ids=e.arm_actuator_ids; e.model.actuator_gainprm[ids,0]*=2.; e.model.actuator_biasprm[ids,1]*=2.
def cosine(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def dump_csv(path, rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
def state(e,label,pstep,dist):
    names=("stage","is_grasp_latched","grasp_object_offset","latched_object_xy","gripper_command_normalized","release_steps","release_has_opened","place_descent_active","place_handoff_count","tcp_target_pos","tcp_target_quat")
    return {"label":label,"place_step":pstep,"distance":dist,"qpos":e.data.qpos.copy(),"qvel":e.data.qvel.copy(),"ctrl":e.data.ctrl.copy(),"mocap_pos":e.data.mocap_pos.copy(),"mocap_quat":e.data.mocap_quat.copy(),"attrs":{n:copy.deepcopy(getattr(e,n)) for n in names}}
def restore(e,s):
    e.data.qpos[:]=s["qpos"]; e.data.qvel[:]=s["qvel"]; e.data.ctrl[:]=s["ctrl"]
    e.data.mocap_pos[:]=s["mocap_pos"]; e.data.mocap_quat[:]=s["mocap_quat"]
    for n,v in s["attrs"].items(): setattr(e,n,copy.deepcopy(v))
    mujoco.mj_forward(e.model,e.data)
def make_env(mode):
    env=gym.make(ENV_ID,max_tcp_lead=.03,ik_axis_weight=.35,ik_posture_weight=.02,ik_posture_mode=mode)
    kp2(env.unwrapped); return env

def collect_snapshots(model):
    env=make_env("raw"); e=env.unwrapped; obs,_=env.reset(seed=0); series=[]; done=False; pstep=0
    while not done:
        action,_=model.predict(obs,deterministic=True); obs,_,t,tr,info=env.step(action); done=t or tr
        if int(e.diag_stage_before)==STAGE_PLACE:
            pstep+=1; series.append(state(e,f"step_{pstep}",pstep,float(info["object_to_goal_xy_distance"])))
    env.close(); d=np.array([s["distance"] for s in series]); away=np.flatnonzero(d>d[0]+.05)
    before=max(1,int(away[0])-1) if len(away) else max(1,len(series)//2)
    picks=(0,before,len(series)-1); labels=("place_entry","before_away","far_after")
    out=[]
    for label,i in zip(labels,picks): s=copy.deepcopy(series[i]); s["label"]=label; out.append(s)
    return out

def run_snapshots(model):
    rows=[]; projections=[]
    for s in collect_snapshots(model):
      for mode in MODES:
       for mm in (5,10,12):
        env=make_env(mode); e=env.unwrapped; e.reset(seed=0); restore(e,s)
        tcp=e.data.site_xpos[e.pinch_site_id].copy(); obj=e.data.site_xpos[e.object_site_id].copy(); goal=e.data.site_xpos[e.goal_site_id].copy()
        direction=(goal-obj)[:2]; direction/=np.linalg.norm(direction)+1e-12; target=tcp.copy(); target[:2]+=direction*mm/1000.
        qtarget=e._solve_ik(target); ik=e.diag_ik; fk=ik["fk_target_tcp"].copy(); e.data.ctrl[e.arm_actuator_ids]=qtarget
        for sub in range(e.frame_skip): mujoco.mj_step(e.model,e.data); e._update_grasp_latch(update_counter=sub==e.frame_skip-1)
        actual=e.data.site_xpos[e.pinch_site_id].copy(); obja=e.data.site_xpos[e.object_site_id].copy()
        g=(goal-obj)[:2]; cmd=(target-tcp)[:2]; fkd=(fk-tcp)[:2]; act=(actual-tcp)[:2]; od=(obja-obj)[:2]
        row={"snapshot":s["label"],"source_place_step":s["place_step"],"source_goal_distance":s["distance"],"mode":mode,"command_mm":mm,
          "cmd_dx":cmd[0],"cmd_dy":cmd[1],"fk_dx":fkd[0],"fk_dy":fkd[1],"actual_dx":act[0],"actual_dy":act[1],"object_dx":od[0],"object_dy":od[1],
          "cos_goal_command":cosine(g,cmd),"cos_goal_fk":cosine(g,fkd),"cos_goal_actual":cosine(g,act),"cos_goal_object":cosine(g,od),
          "position_residual":float(np.linalg.norm(ik["position_error"])),"approach_axis_residual":ik["approach_axis_error_norm"],
          "posture_raw_norm":ik["posture_raw_increment_norm"],"posture_projected_norm":ik["posture_projected_increment_norm"],"j_posture_norm":ik["j_posture_norm"],
          "j_posture_max":ik["max_j_posture_norm"],"j_posture_rms":ik["rms_j_posture_norm"],"ik_iterations":ik["iterations"],"dq_clip_iterations":ik["dq_clip_iterations"],"min_joint_limit_margin":ik["min_joint_limit_margin"]}
        rows.append(row); projections.append({k:row[k] for k in ("snapshot","mode","command_mm","posture_raw_norm","posture_projected_norm","j_posture_norm","j_posture_max","j_posture_rms","position_residual","approach_axis_residual")}); env.close()
    dump_csv(OUT/"snapshot_comparison.csv",rows); dump_csv(OUT/"posture_projection_metrics.csv",projections)
    c=[r for r in rows if r["mode"]=="nullspace"]
    passed=sum(r["cos_goal_fk"]>0 and r["cos_goal_actual"]>0 and r["cos_goal_object"]>0 for r in c)
    return rows,passed

def run_episodes(model):
    rows=[]; place_tcp=defaultdict(list)
    for mode in MODES:
      for seed in range(10):
        env=make_env(mode); e=env.unwrapped; obs,info=env.reset(seed=seed); done=False; steps=0
        reach=grasp=lift=entered=xyready=lowready=openready=opened=False; minxy=np.inf; place_rev=place_n=dqclip=ikiter=0
        minmargin=np.inf; minmargin_j=np.full(6,np.inf); maxqvel=maxforce=maxjump=0.; prevq=e.data.qpos[e.arm_qpos_ids].copy(); table=0; dropped=flung=False
        while not done:
          action,_=model.predict(obs,deterministic=True); obs,_,t,tr,info=env.step(action); done=t or tr; steps+=1
          reach|=bool(info.get("reach_success")); grasp|=bool(info.get("grasp_success")); lift|=bool(info.get("lift_success"))
          q=e.data.qpos[e.arm_qpos_ids]; margin=np.minimum(q-e.arm_ctrl_low,e.arm_ctrl_high-q); minmargin=min(minmargin,float(margin.min())); minmargin_j=np.minimum(minmargin_j,margin)
          maxqvel=max(maxqvel,float(np.max(np.abs(e.data.qvel[e.arm_qvel_ids])))); maxforce=max(maxforce,float(np.max(np.abs(e.data.actuator_force[e.arm_actuator_ids])))); maxjump=max(maxjump,float(np.linalg.norm(q-prevq))); prevq=q.copy(); table+=int(info.get("table_contact_count",0)>0)
          dqclip+=e.diag_ik["dq_clip_iterations"]; ikiter+=e.diag_ik["iterations"]
          if int(e.diag_stage_before)==STAGE_PLACE:
            entered=True; xy=float(info["object_to_goal_xy_distance"]); minxy=min(minxy,xy); xyready|=bool(info["place_xy_ready"]); lowready|=bool(info["place_low_ready"]); openready|=bool(info["place_open_ready"]); opened|=bool(info["place_opened"] or info["place_has_opened"])
            before=e.diag_tcp_actual_before[:2]; target=e.diag_target_after_lead_clip[:2]; fk=e.diag_ik["fk_target_tcp"][:2]; goal=np.asarray(info["goal_position"][:2]); obj=e.diag_object_before[:2]; cmd=target-before; fkd=fk-before; g=goal-obj
            if np.linalg.norm(cmd)>1e-9: place_n+=1; place_rev+=int(cosine(g,cmd)>0 and cosine(g,fkd)<0)
            place_tcp[mode].append(float(info["tcp_target_error"]))
          dropped|=bool(info.get("object_position",[0,0,1])[2] < e.table_top_z+e.object_half_size-.01); flung|=bool(info.get("object_speed",0)>.5)
        finalxy=float(info.get("object_to_goal_xy_distance",np.nan)); rows.append({"mode":mode,"seed":seed,"full_success":bool(info.get("is_success")),"place_success":bool(info.get("place_success")),"reach_success":reach,"grasp_success":grasp,"lift_success":lift,"entered_place":entered,"reached_xy_release_region":xyready,"reached_low_height":lowready,"place_open_ready":openready,"gripper_opened":opened,"place_command_fk_reverse_rate":place_rev/max(place_n,1),"min_goal_xy":None if not np.isfinite(minxy) else minxy,"final_goal_xy":finalxy,"episode_steps":steps,"final_q_home_norm":float(np.linalg.norm(e.data.qpos[e.arm_qpos_ids]-e.home_arm_qpos)),"min_joint_limit_margin":minmargin,**{f"joint_{i}_min_margin":minmargin_j[i] for i in range(6)},"max_joint_step_norm":maxjump,"posture_jump":maxjump>.25,"dq_clip_rate":dqclip/max(ikiter,1),"max_joint_velocity":maxqvel,"max_actuator_force":maxforce,"robot_table_contact_steps":table,"object_dropped":dropped,"object_fling":flung}); env.close(); print(mode,seed,rows[-1]["full_success"],finalxy)
    dump_csv(OUT/"episode_comparison.csv",rows); return rows,place_tcp

def summarize(snap,passed,episodes=None,place_tcp=None):
    summary={"configuration":{"max_tcp_lead":.03,"kp_scale":2.,"checkpoint":str(MODEL_PATH),"seeds":list(range(10)),"axis_weight":.35,"posture_weight":.02},"snapshot_nullspace_direction_pass":passed,"full_evaluation_ran":episodes is not None,"snapshot":{}}
    for mode in MODES:
      rr=[r for r in snap if r["mode"]==mode]; summary["snapshot"][mode]={"direction_pass":sum(r["cos_goal_fk"]>0 and r["cos_goal_actual"]>0 and r["cos_goal_object"]>0 for r in rr),"mean_cos_fk":float(np.mean([r["cos_goal_fk"] for r in rr])),"max_j_posture_norm":max(r["j_posture_max"] for r in rr),"mean_position_residual":float(np.mean([r["position_residual"] for r in rr]))}
    if episodes is not None:
      summary["episodes"]={}
      for mode in MODES:
       rr=[r for r in episodes if r["mode"]==mode]; tcp=np.asarray(place_tcp[mode]); summary["episodes"][mode]={"episodes":10,"full_success":sum(r["full_success"] for r in rr),"place_success":sum(r["place_success"] for r in rr),"reach_success":sum(r["reach_success"] for r in rr),"grasp_success":sum(r["grasp_success"] for r in rr),"lift_success":sum(r["lift_success"] for r in rr),"entered_place":sum(r["entered_place"] for r in rr),"reached_xy_release_region":sum(r["reached_xy_release_region"] for r in rr),"reached_low_height":sum(r["reached_low_height"] for r in rr),"place_open_ready":sum(r["place_open_ready"] for r in rr),"gripper_opened":sum(r["gripper_opened"] for r in rr),"mean_command_fk_reverse_rate":float(np.mean([r["place_command_fk_reverse_rate"] for r in rr])),"mean_min_goal_xy":float(np.mean([r["min_goal_xy"] for r in rr if r["min_goal_xy"] is not None])),"mean_final_goal_xy":float(np.mean([r["final_goal_xy"] for r in rr])),"place_tcp_error_mean":float(tcp.mean()),"place_tcp_error_p95":float(np.percentile(tcp,95)),"place_tcp_error_max":float(tcp.max()),"mean_episode_steps":float(np.mean([r["episode_steps"] for r in rr])),"mean_q_home_norm":float(np.mean([r["final_q_home_norm"] for r in rr])),"min_joint_limit_margin":float(min(r["min_joint_limit_margin"] for r in rr)),"posture_jump_episodes":sum(r["posture_jump"] for r in rr),"mean_dq_clip_rate":float(np.mean([r["dq_clip_rate"] for r in rr])),"max_joint_velocity":max(r["max_joint_velocity"] for r in rr),"max_actuator_force":max(r["max_actuator_force"] for r in rr),"robot_table_contact_steps":sum(r["robot_table_contact_steps"] for r in rr),"object_drop_episodes":sum(r["object_dropped"] for r in rr),"object_fling_episodes":sum(r["object_fling"] for r in rr)}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)); return summary

def main():
    model=PPO.load(MODEL_PATH,device="cpu"); snap,passed=run_snapshots(model); episodes=place_tcp=None
    if passed>=8: episodes,place_tcp=run_episodes(model)
    summary=summarize(snap,passed,episodes,place_tcp); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
