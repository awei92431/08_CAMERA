# Sim-to-Real migration plan

This plan minimizes architectural change while removing privileged runtime
state in dependency order. It does not propose threshold changes or PPO
training as the first response.

## Design rule: one source per runtime quantity

Introduce interface-level data records, not a second environment:

- `RobotState`: timestamped joint position/velocity, TCP pose, safety status;
- `ObjectEstimate`: pose, timestamp, covariance/confidence, source and validity;
- `GripperState`: commanded/actual opening, current/effort, contact/hold state;
- `TaskState`: stage, timers, failure reason and recovery status.

Simulation and hardware should provide backend implementations of these same
interfaces. PPO observation, servo, FSM and safety must consume the same
authoritative records. Reward/evaluator may separately read simulator truth.

## Phase 1: remove object truth from Place servo

Goal: eliminate the clearest dual source before changing the FSM.

1. Route the existing RGB-D/TCP-propagated `ObjectEstimate` to Place XY servo.
2. Use it for `goal_delta_xy`, descent entry, low-enough/release geometry and
   object-based Reach/Grasp safety clamps.
3. Preserve the present gains and thresholds for the first comparison; change
   only the data source.
4. Add estimate freshness/confidence checks. Invalid or stale input must stop or
   enter recovery, never fall back silently to MuJoCo truth or `[0,0,0]`.
5. Run paired simulation tests verifying that policy observation, servo and FSM
   report the same object estimate ID/timestamp each step.

Exit criteria:

- no control-path `site_xpos[object_site_id]` read in Place servo/descent/release;
- no truth fallback;
- 100-seed regression and injected dropout/staleness tests recorded.

## Phase 2: replace FSM contact and grasp truth

Goal: make stage/gripper decisions depend on deployable signals.

1. Separate stage transitions from reward computation into a task supervisor.
2. Replace ALIGN/DESCEND geometry with `ObjectEstimate + TCP FK`.
3. Define grasp confirmation from actual gripper opening, current/effort,
   commanded close completion and temporal consistency; add tactile input only
   if hardware provides it.
4. Remove penetration depth and direct object-qpos latch from runtime semantics.
5. Trigger the existing TCP-object tracker from the new grasp confirmation.
6. Define slip/loss detection and bounded retry/abort timeouts.
7. Decide how to replace PPO `obs[35:39]`: measured estimates, explicit zeros
   only with retraining evidence, or a policy fine-tune using deployable fields.

Exit criteria:

- FSM has no MuJoCo contact, penetration, object qpos or simulated latch input;
- simulation backend emulates real sensor messages rather than exposing truth;
- task supervisor can run without reward computation.

## Phase 3: connect real D435i calibration

Goal: turn the nominal camera model into metric, synchronized real perception.

1. Read factory/stream-specific Color and Depth intrinsics and distortion.
2. Validate depth scale, invalid-depth behavior and filtering at working range.
3. Replace nominal Depth→Color extrinsics with calibrated values or verified
   device calibration; retain geometric z-buffer alignment.
4. Measure wrist camera mounting transform from CAD and perform hand-eye
   calibration against the UR5e flange/TCP chain.
5. Timestamp RGB, depth and robot joints; interpolate robot pose to camera time.
6. Store calibration version, serial number, units and frame convention with
   every test log.

Exit criteria:

- known-target base-frame error characterized across workspace and arm poses;
- transform direction/axis/unit tests pass on recorded real data;
- calibration age/version is checked at startup.

## Phase 4: connect the real UR5e joint/TCP interface

Goal: replace MuJoCo state and actuator calls without changing PPO semantics.

1. Receive timestamped joint state and robot safety/status feedback.
2. Compute TCP FK/Jacobian using the calibrated kinematic model and active TCP.
3. Feed PPO TCP targets through the existing DLS IK.
4. Convert joint targets to a supported bounded real-time UR interface
   (`servoj`/trajectory/RTDE architecture as appropriate), with rate and
   acceleration limiting outside PPO.
5. Verify no direct qpos write and no mocap/weld dependency exists in hardware.

Exit criteria:

- low-speed free-space TCP tracking passes position/orientation and watchdog
  bounds;
- command loss, stale state and protective stop cause a safe halt.

## Phase 5: connect the real gripper SDK and feedback

Goal: replace command-derived `gripper_state` and simulated contact.

1. Implement open/close/hold commands through the vendor SDK.
2. Read actual opening/finger position, motion status, current/effort and faults.
3. Calibrate empty close, object-width close and stall signatures.
4. Publish `GripperState` for PPO adapter, FSM and safety supervisor.
5. Validate grasp confirmation and release completion without camera truth.

Exit criteria:

- open/closed/holding/fault states are repeatable with timeouts;
- FSM uses measured feedback only and never infers completion from command alone.

## Phase 6: production safety and failure recovery

Goal: make failures bounded before object interaction.

Add independently enforced:

- joint/TCP speed and acceleration limits;
- workspace and calibrated table/fixture keep-out volumes;
- force/current/collision and UR protective-stop handling;
- state, camera and command freshness watchdogs;
- per-stage and per-motion timeouts;
- visual invalid/stale behavior, re-observation and last-frame policy;
- grasp retry limits, slip/loss response and safe retreat;
- hardware E-stop and operator reset procedure;
- structured logging of every safety decision and data-source ID.

Exit criteria:

- fault-injection tests demonstrate safe stop/recovery for every missing or stale
  input and communication failure;
- no learned action can bypass the independent safety layer.

## Phase 7: progressive physical validation

Run in this order, with explicit review gates:

1. powered-off frame and calibration checks;
2. low-speed no-load joint and TCP motion above an empty table;
3. camera observation cycle with a fixed calibration target;
4. fake/lightweight object with gripper disabled;
5. gripper open/close away from the table;
6. low-force dummy-object contact and grasp;
7. lift only a few millimetres with catch/support protection;
8. short transport and controlled release;
9. full low-speed grasp/place over a bounded workspace;
10. expand workspace/speed only after logged acceptance tests.

At every gate compare estimated pose, TCP tracking, gripper state, supervisor
state and safety events. Do not use task success alone as the acceptance metric.

## Minimum next change

Change **Place servo first**, because it is a direct post-PPO motion command that
currently bypasses the visual estimate with object truth. Then change the FSM as
the immediately following phase; a visual Place servo with a privileged release
FSM is still not deployable, but fixing the servo first establishes the single
authoritative `ObjectEstimate` interface that the FSM should consume.
