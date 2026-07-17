from gymnasium.envs.registration import register

from fourc2.envs.allstage import (
    STAGE_FULL,
    STAGE_GRASP,
    STAGE_LIFT,
    STAGE_PLACE,
    STAGE_REACH,
    STAGE_REACH_GRASP,
)


PIPELINE_VERSION = "cube3cm-single-ppo-v2.2-xy6mm"


ENV_IDS = [
    "My4C2ReachStage-v0",
    "My4C2GraspStage-v0",
    "My4C2LiftStage-v0",
    "My4C2PlaceStage-v0",
    "My4C2GraspLiftStage-v0",
    "My4C2TransportPlaceStage-v0",
    "My4C2ReachGraspStage-v0",
    "My4C2AllStage-v0",
    "My4C2ReachStageCube3cm-v0",
    "My4C2GraspStageCube3cm-v0",
    "My4C2LiftStageCube3cm-v0",
    "My4C2PlaceStageCube3cm-v0",
    "My4C2GraspLiftStageCube3cm-v0",
    "My4C2TransportPlaceStageCube3cm-v0",
    "My4C2ReachGraspStageCube3cm-v0",
    "My4C2AllStageCube3cm-v0",
    "My4C2CurriculumCube3cm-v0",
    "My4C2FinalCurriculumCube3cm-v0",
    "My4C2LiftSingleV22Cube3cm-v0",
    "My4C2GraspLiftTransitionV22Cube3cm-v0",
    "My4C2ReachGraspLiftEvalV22Cube3cm-v0",
    "My4C2PlaceSingleV22Cube3cm-v0",
    "My4C2LiftPlaceTransitionV22Cube3cm-v0",
    "My4C2AllStageSinglePPOV22Cube3cm-v0",
]


CUBE3CM_KWARGS = {
    "model_xml_path": "scene_cube3cm.xml",
    "object_half_size": 0.015,
    "grasp_height_offset": 0.018,
    "lift_height": 0.05,
    "close_reward_distance": 0.025,
    "contact_reward_distance": 0.035,
    "rough_close_reward_distance": 0.055,
    "rough_grasp_xy_close_threshold": 0.035,
    "rough_grasp_z_close_threshold": 0.035,
    "grasp_descend_xy_threshold": 0.018,
    "grasp_xy_close_threshold": 0.012,
    "grasp_z_close_threshold": 0.022,
    "latch_distance_threshold": 0.035,
    "max_pregrasp_object_xy_drift": 0.015,
    # A 30 mm cube with 10-15 mm accepted XY error can be held by only one
    # edge of each finger.  Require the pinch center within 6 mm for the
    # multi-step stable latch and within 4 mm for strict quality.
    "stable_grasp_xy_threshold": 0.006,
    "latch_grasp_xy_threshold": 0.004,
    "latch_grasp_z_threshold": 0.022,
    "place_handoff_xy_threshold": 0.024,
    "place_descent_xy_threshold": 0.028,
    "place_xy_servo_gain": 0.55,
    "place_xy_servo_max_delta": 0.012,
    "release_open_xy_threshold": 0.024,
    "release_success_lift": 0.016,
    "release_min_open_steps": 8,
    "release_descent_action_scale": 0.30,
    "place_success_bonus": 3000.0,
}


def register_allstage(id, stage, max_episode_steps, **kwargs):
    register(
        id=id,
        entry_point="fourc2.envs:My4C2AllStageEnv",
        max_episode_steps=max_episode_steps,
        kwargs={"training_stage": stage, **kwargs},
    )


register_allstage(
    "My4C2ReachStage-v0",
    STAGE_REACH,
    250,
)
register_allstage(
    "My4C2GraspStage-v0",
    STAGE_GRASP,
    350,
)
register_allstage(
    "My4C2LiftStage-v0",
    STAGE_LIFT,
    350,
)
register_allstage(
    "My4C2PlaceStage-v0",
    STAGE_PLACE,
    450,
)
register_allstage(
    "My4C2GraspLiftStage-v0",
    STAGE_LIFT,
    350,
)
register_allstage(
    "My4C2TransportPlaceStage-v0",
    STAGE_PLACE,
    450,
)
register_allstage(
    "My4C2ReachGraspStage-v0",
    STAGE_REACH_GRASP,
    500,
)
register_allstage(
    "My4C2AllStage-v0",
    STAGE_FULL,
    900,
    include_stage_observation=True,
)

register_allstage(
    "My4C2ReachStageCube3cm-v0",
    STAGE_REACH,
    250,
    **CUBE3CM_KWARGS,
)
register_allstage(
    "My4C2GraspStageCube3cm-v0",
    STAGE_GRASP,
    350,
    **CUBE3CM_KWARGS,
)
register_allstage(
    "My4C2LiftStageCube3cm-v0",
    STAGE_LIFT,
    350,
    **CUBE3CM_KWARGS,
)
register_allstage(
    "My4C2PlaceStageCube3cm-v0",
    STAGE_PLACE,
    450,
    **CUBE3CM_KWARGS,
)
register_allstage(
    "My4C2GraspLiftStageCube3cm-v0",
    STAGE_LIFT,
    350,
    **CUBE3CM_KWARGS,
)
register_allstage(
    "My4C2TransportPlaceStageCube3cm-v0",
    STAGE_PLACE,
    450,
    **CUBE3CM_KWARGS,
)
register_allstage(
    "My4C2ReachGraspStageCube3cm-v0",
    STAGE_REACH_GRASP,
    500,
    **CUBE3CM_KWARGS,
)
register_allstage(
    "My4C2AllStageCube3cm-v0",
    STAGE_FULL,
    900,
    include_stage_observation=True,
    **CUBE3CM_KWARGS,
)

register_allstage(
    "My4C2CurriculumCube3cm-v0",
    STAGE_FULL,
    900,
    reset_stage_probabilities=(
        (STAGE_REACH, 0.30),
        (STAGE_GRASP, 0.40),
        (STAGE_LIFT, 0.20),
        (STAGE_PLACE, 0.10),
    ),
    include_stage_observation=True,
    **CUBE3CM_KWARGS,
)
register_allstage(
    "My4C2FinalCurriculumCube3cm-v0",
    STAGE_FULL,
    900,
    reset_stage_probabilities=(
        (STAGE_REACH, 0.10),
        (STAGE_GRASP, 0.35),
        (STAGE_LIFT, 0.35),
        (STAGE_PLACE, 0.20),
    ),
    include_stage_observation=True,
    **CUBE3CM_KWARGS,
)

# v2.2 keeps one 39-D PPO through the complete curriculum.  "Single" envs
# teach one newly introduced stage from a realistic scripted predecessor;
# "Transition" envs start one stage earlier and execute the real reward and
# controller handoff.  Eval envs start at Reach and terminate at the requested
# chain endpoint, so handoff checkpoints cannot be selected by the new stage
# alone.
register_allstage(
    "My4C2LiftSingleV22Cube3cm-v0",
    STAGE_LIFT,
    350,
    include_stage_observation=False,
    **CUBE3CM_KWARGS,
)
register_allstage(
    "My4C2GraspLiftTransitionV22Cube3cm-v0",
    STAGE_FULL,
    650,
    reset_stage_probabilities=((STAGE_GRASP, 1.0),),
    sequential_training=True,
    success_stage=STAGE_LIFT,
    include_stage_observation=False,
    **CUBE3CM_KWARGS,
)
register_allstage(
    "My4C2ReachGraspLiftEvalV22Cube3cm-v0",
    STAGE_FULL,
    750,
    reset_stage_probabilities=((STAGE_REACH, 1.0),),
    sequential_training=True,
    success_stage=STAGE_LIFT,
    include_stage_observation=False,
    **CUBE3CM_KWARGS,
)
register_allstage(
    "My4C2PlaceSingleV22Cube3cm-v0",
    STAGE_PLACE,
    450,
    include_stage_observation=False,
    place_reset_variation=True,
    **CUBE3CM_KWARGS,
)
register_allstage(
    "My4C2LiftPlaceTransitionV22Cube3cm-v0",
    STAGE_FULL,
    700,
    reset_stage_probabilities=((STAGE_LIFT, 1.0),),
    sequential_training=True,
    success_stage=STAGE_PLACE,
    include_stage_observation=False,
    **CUBE3CM_KWARGS,
)
register_allstage(
    "My4C2AllStageSinglePPOV22Cube3cm-v0",
    STAGE_FULL,
    900,
    reset_stage_probabilities=((STAGE_REACH, 1.0),),
    sequential_training=True,
    success_stage=STAGE_PLACE,
    include_stage_observation=False,
    **CUBE3CM_KWARGS,
)
