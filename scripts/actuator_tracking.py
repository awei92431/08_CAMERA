#!/usr/bin/env python3
import csv,json,os,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(os.environ.get('PROJECT_ROOT',Path(__file__).resolve().parents[1]));sys.path.insert(0,str(ROOT))
import gymnasium as gym,mujoco
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
import fourc2
OUT=Path(os.environ.get('ACT_OUT',ROOT/'results/actuator_tracking'));OUT.mkdir(parents=True,exist_ok=True);PLOTS=OUT/'plots';PLOTS.mkdir(exist_ok=True)
ENV='My4C2AllStageSinglePPOV22Cube3cm-v0';HOLD=200;STEADY=20
CONFIGS=[('kp_1_0x',1.0),('kp_1_5x',1.5),('kp_2_0x',2.0)]

def table_contacts(env):
 n=0
 for c in env.data.contact:
  if c.geom1==env.table_geom_id:
   n+=int(c.geom2!=env.object_geom_id)
  elif c.geom2==env.table_geom_id:
   n+=int(c.geom1!=env.object_geom_id)
 return n

def z_contributions(env,qtarget):
 actual_q=env.data.qpos[env.arm_qpos_ids].copy();base_z=float(env.data.site_xpos[env.pinch_site_id,2]);out=[]
 env._copy_data_for_ik()
 for i,qid in enumerate(env.arm_qpos_ids):
  env.ik_data.qpos[env.arm_qpos_ids]=actual_q;env.ik_data.qpos[qid]=qtarget[i];mujoco.mj_forward(env.model,env.ik_data)
  out.append(float(env.ik_data.site_xpos[env.pinch_site_id,2]-base_z))
 return out

def scale_kp(env,scale):
 ids=env.arm_actuator_ids;env.model.actuator_gainprm[ids,0]*=scale;env.model.actuator_biasprm[ids,1]*=scale

def run_config(name,scale):
 joint_rows=[];force_rows=[];cases=[];responses={};saturation_samples=total_joint_samples=0;dq_clip_iterations=ik_iterations=0
 for axis in range(3):
  for sign in (1,-1):
   for mm in (10,30,50):
    env=gym.make(ENV,max_tcp_lead=.03,ik_axis_weight=.35,ik_posture_weight=.02).unwrapped;scale_kp(env,scale);env.reset(seed=123)
    start=env.data.site_xpos[env.pinch_site_id].copy();target=start.copy();target[axis]+=sign*mm/1000;case=f"{'xyz'[axis]}_{'p' if sign>0 else 'm'}{mm}";series=[]
    for step in range(HOLD):
     qtarget=env._solve_ik(target);fk=env.diag_ik['fk_target_tcp'].copy();dq_clip_iterations+=env.diag_ik['dq_clip_iterations'];ik_iterations+=env.diag_ik['iterations'];env.data.ctrl[env.arm_actuator_ids]=qtarget
     max_table=0
     for _ in range(env.frame_skip):mujoco.mj_step(env.model,env.data);max_table=max(max_table,table_contacts(env))
     actual=env.data.site_xpos[env.pinch_site_id].copy();qpos=env.data.qpos[env.arm_qpos_ids].copy();qvel=env.data.qvel[env.arm_qvel_ids].copy();forces=env.data.actuator_force[env.arm_actuator_ids].copy();zcon=z_contributions(env,qtarget)
     rec={'step':step+1,'actual':actual,'target':target,'fk':fk,'position_error':float(np.linalg.norm(target-actual)),
      'actuator_error':float(np.linalg.norm(fk-actual)),'qgap':float(np.linalg.norm(qtarget-qpos)),'max_joint_velocity':float(np.max(np.abs(qvel))),
      'max_table_contacts':max_table,'ncon':int(env.data.ncon)};series.append(rec)
     if step>=HOLD-STEADY:
      for i,jname in enumerate(fourc2.envs.allstage.ARM_JOINT_NAMES):
       aid=int(env.arm_actuator_ids[i]);did=int(env.arm_qvel_ids[i]);fr=env.model.actuator_forcerange[aid].copy();sat=bool(abs(forces[i])>=.99*max(abs(fr[0]),abs(fr[1])))
       row={'config':name,'case':case,'step':step+1,'joint':jname,'q_target':qtarget[i],'qpos':qpos[i],'q_residual':qtarget[i]-qpos[i],'qvel':qvel[i],
        'actuator_ctrl':env.data.ctrl[aid],'actuator_force':forces[i],'qfrc_actuator':env.data.qfrc_actuator[did],'qfrc_bias':env.data.qfrc_bias[did],
        'kp':env.model.actuator_gainprm[aid,0],'force_low':fr[0],'force_high':fr[1],'force_saturated':sat,'tcp_z_contribution':zcon[i]}
       joint_rows.append(row);force_rows.append(row.copy());saturation_samples+=int(sat);total_joint_samples+=1
    resp=np.asarray([r['actual'] for r in series]);cmd=sign*mm/1000;proj=sign*(resp[:,axis]-start[axis]);threshold=.9*abs(cmd);cross=np.flatnonzero(proj>=threshold);rise=None if not len(cross) else float((cross[0]+1)*env.model.opt.timestep*env.frame_skip)
    ss=series[-STEADY:];err=np.asarray([r['position_error'] for r in ss]);signed=np.mean(target-resp[-STEADY:],axis=0);overshoot=max(0.,float(proj.max()-abs(cmd)));turns=int(np.sum(np.diff(np.sign(np.diff(proj)))!=0));osc=turns>10
    cases.append({'case':case,'axis':'xyz'[axis],'direction':sign,'amplitude_mm':mm,'steady_3d_error':float(err.mean()),'error_x':float(signed[0]),'error_y':float(signed[1]),'error_z':float(signed[2]),
     'actuator_tracking_error':float(np.mean([r['actuator_error'] for r in ss])),'q_target_qpos_norm':float(np.mean([r['qgap'] for r in ss])),
     'rise_time_s':rise,'overshoot':overshoot,'oscillation':osc,'turning_points':turns,'max_joint_velocity':max(r['max_joint_velocity'] for r in series),
     'max_actuator_force':float(max(abs(r['actuator_force']) for r in force_rows[-STEADY*6:])),'saturation_rate':float(sum(r['force_saturated'] for r in force_rows[-STEADY*6:])/(STEADY*6)),
     'max_table_contacts':max(r['max_table_contacts'] for r in series),'max_contacts':max(r['ncon'] for r in series)})
    if axis==2 and mm==50:
     responses[case]=(np.arange(1,HOLD+1)*env.model.opt.timestep*env.frame_skip,target[2],resp[:,2])
    env.close()
 vals=lambda k:np.asarray([c[k] for c in cases],float)
 summary={'config':name,'kp_scale':scale,'tests':18,'mean_steady_3d_error':float(vals('steady_3d_error').mean()),'p95_steady_3d_error':float(np.percentile(vals('steady_3d_error'),95)),'max_steady_3d_error':float(vals('steady_3d_error').max()),
  'mean_signed_error_xyz':[float(vals(f'error_{k}').mean()) for k in 'xyz'],'mean_actuator_tracking_error':float(vals('actuator_tracking_error').mean()),
  'mean_q_target_qpos_norm':float(vals('q_target_qpos_norm').mean()),'mean_rise_time_s':float(np.mean([c['rise_time_s'] for c in cases if c['rise_time_s'] is not None])),
  'max_overshoot':float(vals('overshoot').max()),'oscillating_tests':int(sum(c['oscillation'] for c in cases)),'max_joint_velocity':float(vals('max_joint_velocity').max()),
  'max_actuator_force':float(max(abs(r['actuator_force']) for r in force_rows)),'torque_saturation_rate':float(saturation_samples/total_joint_samples),
  'dq_clip_rate':float(dq_clip_iterations/ik_iterations),
  'tests_with_table_contact':int(sum(c['max_table_contacts']>0 for c in cases)),'max_contact_count':int(vals('max_contacts').max())}
 for case,(t,targ,z) in responses.items():
  plt.figure(figsize=(7,4));plt.plot(t,np.full_like(t,targ),label='target z');plt.plot(t,z,label='actual z');plt.xlabel('time (s)');plt.ylabel('TCP Z (m)');plt.grid(alpha=.3);plt.legend();plt.tight_layout();plt.savefig(PLOTS/f'{name}_{case}.png',dpi=140);plt.close()
 return summary,cases,joint_rows,force_rows

def main():
 allout={};comparisons=[];allj=[];allf=[]
 for idx,(name,scale) in enumerate(CONFIGS):
  summary,cases,jrows,frows=run_config(name,scale);allout[name]={'summary':summary,'cases':cases};comparisons.append(summary);allj+=jrows;allf+=frows;print(name,json.dumps(summary))
  if idx==0 and summary['torque_saturation_rate']>.01:
   allout['scan_stopped_reason']='baseline torque saturation >1%';break
  if idx>0 and (summary['oscillating_tests']>0 or summary['tests_with_table_contact']>0):
   allout['scan_stopped_reason']=f'{name} instability/contact';break
 with (OUT/'joint_residuals.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=allj[0]);w.writeheader();w.writerows(allj)
 with (OUT/'force_diagnostics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=allf[0]);w.writeheader();w.writerows(allf)
 with (OUT/'kp_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=comparisons[0]);w.writeheader();w.writerows(comparisons)
 (OUT/'kp_comparison.json').write_text(json.dumps(allout,indent=2))
if __name__=='__main__':main()
