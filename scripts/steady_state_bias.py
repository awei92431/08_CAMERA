#!/usr/bin/env python3
import csv,json,os,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(os.environ.get('PROJECT_ROOT',Path(__file__).resolve().parents[1]));sys.path.insert(0,str(ROOT))
import gymnasium as gym,mujoco
import fourc2
OUT=Path(os.environ.get('BIAS_OUT',ROOT/'results/steady_state_bias'));OUT.mkdir(parents=True,exist_ok=True)
ENV='My4C2AllStageSinglePPOV22Cube3cm-v0';HOLD=200;STEADY=20
CONFIGS=[('A_current',.35,.02),('B_no_posture',.35,0.0),('C_half_axis',.175,.02)]

def main():
 all_results={};comparison=[]
 for name,axis_w,posture_w in CONFIGS:
  folder=OUT/name;folder.mkdir(exist_ok=True);rows=[];cases=[];total_clip_iters=total_iters=0
  for axis in range(3):
   for sign in (1,-1):
    for mm in (10,30,50):
     env=gym.make(ENV,max_tcp_lead=.030,ik_axis_weight=axis_w,ik_posture_weight=posture_w).unwrapped;env.reset(seed=123)
     start=env.data.site_xpos[env.pinch_site_id].copy();target=start.copy();target[axis]+=sign*mm/1000;series=[]
     for step in range(HOLD):
      qtarget=env._solve_ik(target);ik=env.diag_ik;env.data.ctrl[env.arm_actuator_ids]=qtarget
      for _ in range(env.frame_skip):mujoco.mj_step(env.model,env.data)
      actual=env.data.site_xpos[env.pinch_site_id].copy();fk=ik['fk_target_tcp'].copy();ikerr=target-fk;acterr=fk-actual;qgap=qtarget-env.data.qpos[env.arm_qpos_ids]
      actual_axis=env._pinch_approach_axis(env.data);axiserr=np.cross(actual_axis,env.desired_approach_axis)
      rec={'config':name,'case':f"{'xyz'[axis]}_{'p' if sign>0 else 'm'}{mm}",'step':step+1,'time_s':(step+1)*env.model.opt.timestep*env.frame_skip,
       **{f'target_{k}':target[i] for i,k in enumerate('xyz')},**{f'fk_qtarget_{k}':fk[i] for i,k in enumerate('xyz')},**{f'actual_{k}':actual[i] for i,k in enumerate('xyz')},
       **{f'ik_error_{k}':ikerr[i] for i,k in enumerate('xyz')},**{f'actuator_error_{k}':acterr[i] for i,k in enumerate('xyz')},
       'position_error_norm':float(np.linalg.norm(target-actual)),'ik_error_norm':float(np.linalg.norm(ikerr)),'actuator_error_norm':float(np.linalg.norm(acterr)),
       'q_target_qpos_norm':float(np.linalg.norm(qgap)),'approach_axis_error_norm':float(np.linalg.norm(axiserr)),
       'posture_error_norm':ik['posture_error_norm'],'weighted_posture_error_norm':ik['weighted_posture_error_norm'],
       'ik_converged':ik['converged'],'ik_iterations':ik['iterations'],'dq_clip_iterations':ik['dq_clip_iterations']}
      rows.append(rec);series.append(rec);total_clip_iters+=ik['dq_clip_iterations'];total_iters+=ik['iterations']
     ss=series[-STEADY:]
     mean=lambda k:float(np.mean([r[k] for r in ss]))
     cases.append({'case':series[-1]['case'],'axis':'xyz'[axis],'direction':sign,'amplitude_mm':mm,
      'steady_3d_error':mean('position_error_norm'),'steady_ik_error':mean('ik_error_norm'),'steady_actuator_error':mean('actuator_error_norm'),
      'steady_error_x':float(np.mean([r['target_x']-r['actual_x'] for r in ss])),'steady_error_y':float(np.mean([r['target_y']-r['actual_y'] for r in ss])),
      'steady_error_z':float(np.mean([r['target_z']-r['actual_z'] for r in ss])),'approach_axis_error':mean('approach_axis_error_norm'),
      'q_target_qpos_norm':mean('q_target_qpos_norm'),'posture_error_norm':mean('posture_error_norm'),
      'ik_converged':all(r['ik_converged'] for r in ss),'mean_ik_iterations':mean('ik_iterations')})
     env.close()
  vals=lambda k:np.asarray([c[k] for c in cases])
  summary={'config':name,'axis_weight':axis_w,'posture_weight':posture_w,'tests':18,
   'mean_steady_3d_error':float(vals('steady_3d_error').mean()),'p95_steady_3d_error':float(np.percentile(vals('steady_3d_error'),95)),
   'max_steady_3d_error':float(vals('steady_3d_error').max()),'mean_axis_error_xyz':[float(vals(f'steady_error_{k}').mean()) for k in 'xyz'],
   'mean_ik_solve_error':float(vals('steady_ik_error').mean()),'mean_actuator_tracking_error':float(vals('steady_actuator_error').mean()),
   'mean_approach_axis_error':float(vals('approach_axis_error').mean()),'mean_q_target_qpos_norm':float(vals('q_target_qpos_norm').mean()),
   'mean_posture_error_norm':float(vals('posture_error_norm').mean()),'dq_clip_rate':float(total_clip_iters/total_iters),
   'accuracy_within_1mm':int(np.sum(vals('steady_3d_error')<=.001)),'accuracy_within_3mm':int(np.sum(vals('steady_3d_error')<=.003)),
   'accuracy_within_5mm':int(np.sum(vals('steady_3d_error')<=.005)),'steady_converged_tests':int(sum(c['ik_converged'] for c in cases))}
  with (folder/'steps.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
  (folder/'summary.json').write_text(json.dumps({'summary':summary,'cases':cases},indent=2));all_results[name]={'summary':summary,'cases':cases};comparison.append(summary)
 with (OUT/'comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=comparison[0]);w.writeheader();w.writerows(comparison)
 (OUT/'comparison.json').write_text(json.dumps(all_results,indent=2))
if __name__=='__main__':main()
