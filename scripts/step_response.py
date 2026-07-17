#!/usr/bin/env python3
import csv,json,sys,os
from pathlib import Path
import numpy as np
ROOT=Path(os.environ.get('PROJECT_ROOT',Path(__file__).resolve().parents[1]));sys.path.insert(0,str(ROOT))
import gymnasium as gym, mujoco
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
import fourc2
OUT=Path(os.environ.get('STEP_OUT',ROOT/'results/step_response'));PLOTS=OUT/'plots';OUT.mkdir(parents=True,exist_ok=True);PLOTS.mkdir(exist_ok=True)
ENV='My4C2AllStageSinglePPOV22Cube3cm-v0';DT=0.02;HOLD=200

def main():
 rows=[];summ=[]
 for axis in range(3):
  for sign in (1,-1):
   for mm in (10,30,50):
    env=gym.make(ENV).unwrapped;env.reset(seed=123);start=env.data.site_xpos[env.pinch_site_id].copy();target=start.copy();target[axis]+=sign*mm/1000
    vals=[];dqclip=False;jclip=False;ikfail=False
    for step in range(HOLD):
     qtarget=env._solve_ik(target);ik=env.diag_ik;dqclip|=ik['dq_clip'];jclip|=ik['joint_limit_clip'];ikfail|=not ik['converged']
     env.data.ctrl[env.arm_actuator_ids]=qtarget
     for _ in range(env.frame_skip):mujoco.mj_step(env.model,env.data)
     actual=env.data.site_xpos[env.pinch_site_id].copy();err=target-actual
     rec={'case':f"{'xyz'[axis]}_{'p' if sign>0 else 'm'}{mm}",'step':step+1,'time_s':(step+1)*DT,
      **{f'target_{k}':target[i] for i,k in enumerate('xyz')},**{f'actual_{k}':actual[i] for i,k in enumerate('xyz')},
      'error_norm':np.linalg.norm(err),'q_target':json.dumps(qtarget.tolist()),'actuator_ctrl':json.dumps(env.data.ctrl[env.arm_actuator_ids].tolist()),
      'joint_limit_clip':ik['joint_limit_clip'],'dq_clip':ik['dq_clip'],'ik_failed':not ik['converged']};rows.append(rec);vals.append(actual.copy())
    vals=np.asarray(vals);command=sign*mm/1000;response=sign*(vals[:,axis]-start[axis]);final=response[-20:].mean();err=np.linalg.norm(target-vals,axis=1)
    reached=np.flatnonzero(response>=.9*abs(command));rise=None if not len(reached) else float((reached[0]+1)*DT)
    overshoot=max(0.,float(response.max()-abs(command)));osc=int(np.sum(np.diff(np.sign(np.diff(response)))!=0))>10
    summ.append({'case':rows[-1]['case'],'axis':'xyz'[axis],'direction':sign,'amplitude_mm':mm,'target_tcp':target.tolist(),'final_tcp':vals[-1].tolist(),
     'reached':bool(abs(final-abs(command))<=.001),'rise_time_s':rise,'steady_state_error':float(err[-20:].mean()),'max_error':float(err.max()),
     'overshoot':overshoot,'oscillation':osc,'dq_clip':dqclip,'joint_limit_clip':jclip,'ik_failed':ikfail})
    t=np.arange(1,HOLD+1)*DT;plt.figure(figsize=(7,4));plt.plot(t,np.full(HOLD,target[axis]),label='target');plt.plot(t,vals[:,axis],label='actual');plt.xlabel('time (s)');plt.ylabel(f"TCP {'xyz'[axis]} (m)");plt.legend();plt.grid(alpha=.3);plt.tight_layout();plt.savefig(PLOTS/f"{rows[-1]['case']}.png",dpi=140);plt.close();env.close()
 with (OUT/'step_response.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 (OUT/'summary.json').write_text(json.dumps(summ,indent=2))
if __name__=='__main__':main()
