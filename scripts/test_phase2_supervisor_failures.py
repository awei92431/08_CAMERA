"""Fault injection and dependency-boundary tests for Phase 2 supervisor."""

import inspect
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from fourc2.object_estimate import ObjectEstimate  # noqa: E402
from fourc2.task_supervisor import (  # noqa: E402
    GripperState, RobotState, SupervisorInputUnavailable,
    TaskSupervisor, TaskSupervisorConfig,
)

OUTPUT = ROOT / "outputs" / "phase2_task_supervisor"


def config():
    return TaskSupervisorConfig(.07,0.,.018,.025,.95,.04,.035,.035,.03,.035,
        .08,.012,.051,6,4,24,5,.70,max_input_age=.5)


def robot(t, z=.385):
    return RobotState([.5,0,z],np.eye(3),np.zeros(3),1.,t,True,"test_fk")


def grip(t, opening=1., effort=0., confidence=0., fault=False, valid=True):
    return GripperState(0. if opening<.9 else 1.,opening,0.,"stopped",effort,
                        fault,confidence,t,valid,"test_gripper")


def expect(name, fn, source, reason, out):
    try: fn()
    except SupervisorInputUnavailable as exc:
        assert exc.source==source and exc.reason==reason,(exc.source,exc.reason)
        out[name]={"source":exc.source,"reason":exc.reason,"fail_closed":True}
    else: raise AssertionError(name)


def main():
    out={}; obj=ObjectEstimate([.5,0,.315],0.,True,1.,"rgbd_visual","e1")
    goal=np.array([.6,0,.315]); s=TaskSupervisor(config())
    expect("missing_gripper",lambda:s.update(0.,obj,robot(0.),None,goal),
           "gripper_state","missing",out)
    expect("stale_gripper",lambda:s.update(1.,obj.with_position(obj.position,1.,"rgbd_visual","e2"),robot(1.),grip(0.),goal),
           "gripper_state","stale",out)
    expect("gripper_fault",lambda:s.update(0.,obj,robot(0.),grip(0.,fault=True),goal),
           "gripper_state","fault",out)
    expect("missing_object",lambda:s.update(0.,None,robot(0.),grip(0.),goal),
           "object_estimate","missing",out)

    empty=TaskSupervisor(config()); empty.update(0.,obj,robot(0.),grip(0.),goal)
    empty.update(.02,obj.with_position(obj.position,.02,"rgbd_visual","e2"),robot(.02,.315),grip(.02),goal)
    for i in range(40):
        t=.04+.02*i; eo=obj.with_position(obj.position,t,"rgbd_visual",f"x{i}")
        empty.update(t,eo,robot(t,.315),grip(t,opening=0.,effort=0.,confidence=0.),goal)
    assert not empty.grasp_confirmed
    out["empty_close"]={"grasp_confirmed":False,"false_positive":False}

    source=inspect.getsource(TaskSupervisor)
    forbidden=[token for token in ("mujoco","site_xpos","object_qpos",
                                    "contact.dist","penetration","is_grasp_latched",
                                    "reward","is_success")
               if token in source]
    assert not forbidden,forbidden
    out["dependency_boundary"]={"forbidden_tokens_found":forbidden,
        "reads_only_declared_inputs":True,"latch_isolated":True}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    path=OUTPUT/"failure_injection_results.json"
    path.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2));print("PASS",path)


if __name__=="__main__":main()
