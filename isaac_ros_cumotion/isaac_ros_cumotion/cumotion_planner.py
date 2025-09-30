# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# See original header in your file.

from copy import deepcopy
from os import path
import threading
import time
from typing import Optional
import numpy as np
import torch
import rclpy

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, Cylinder, Mesh, Sphere
from curobo.geom.types import VoxelGrid as CuVoxelGrid
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState as CuJointState
from curobo.util.trajectory import InterpolateType
from curobo.wrap.reacher.motion_gen import (
    MotionGen, MotionGenConfig, MotionGenPlanConfig, MotionGenStatus
)

# === MPC ===
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.rollout.rollout_base import Goal

from geometry_msgs.msg import Point, Vector3, PoseStamped
from isaac_ros_cumotion.update_kinematics import get_robot_config, UpdateLinkSpheresServer
from isaac_ros_cumotion_python_utils.utils import (
    get_grid_center, get_grid_min_corner, get_grid_size, is_grid_valid,
    load_grid_corners_from_workspace_file
)

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import CollisionObject, MoveItErrorCodes, RobotTrajectory, RobotState, DisplayTrajectory

from nvblox_msgs.srv import EsdfAndGradients
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker

from rclpy.duration import Duration
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, ParameterValue


class CumotionActionServer(Node):
    def __init__(self):
        super().__init__('cumotion_action_server')
        self.tensor_args = TensorDeviceType()

        # ---------------- Parameters ----------------
        self.declare_parameter('robot', 'ur5e.yml')
        self.declare_parameter('urdf_path', rclpy.Parameter.Type.STRING)
        self.declare_parameter('yml_file_path', rclpy.Parameter.Type.STRING)
        self.declare_parameter('time_dilation_factor', 0.5)
        self.declare_parameter('max_attempts', 10)
        self.declare_parameter('num_graph_seeds', 6)
        self.declare_parameter('num_trajopt_seeds', 6)
        self.declare_parameter('include_trajopt_retract_seed', True)
        self.declare_parameter('num_trajopt_time_steps', 6)
        self.declare_parameter('trajopt_finetune_iters', 400)
        self.declare_parameter('interpolation_dt', 0.025)
        self.declare_parameter('collision_cache_mesh', 20)
        self.declare_parameter('collision_cache_cuboid', 20)
        self.declare_parameter('voxel_size', 0.05)
        self.declare_parameter('read_esdf_world', False)
        self.declare_parameter('publish_curobo_world_as_voxels', False)
        self.declare_parameter('add_ground_plane', False)
        self.declare_parameter('publish_voxel_size', 0.05)
        self.declare_parameter('max_publish_voxels', 500000)
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('tool_frame', rclpy.Parameter.Type.STRING)

        # Workspace / ESDF
        self.declare_parameter('workspace_file_path', '')
        self.declare_parameter('grid_center_m', [0.0, 0.0, 0.0])
        self.declare_parameter('grid_size_m', [2.0, 2.0, 2.0])
        self.declare_parameter('update_esdf_on_request', True)
        self.declare_parameter('use_aabb_on_request', True)
        self.declare_parameter('esdf_service_name', '/nvblox_node/get_esdf_and_gradient')

        # Debug / scaling overrides
        self.declare_parameter('override_moveit_scaling_factors', False)
        self.declare_parameter('update_link_sphere_server', 'planner_attach_object')
        self.declare_parameter(
            'excluded_joint_names',
            ParameterValue(type=ParameterType.PARAMETER_STRING_ARRAY, string_array_value=[]),
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY)
        )

        # MPPI rollout viz
        self.declare_parameter('viz_enable_mppi_rollouts', True)
        self.declare_parameter('viz_rollouts_topic', '/mpc_rollouts')
        self._viz_enable_mppi_rollouts = self.get_parameter('viz_enable_mppi_rollouts').get_parameter_value().bool_value
        self._viz_rollouts_topic = self.get_parameter('viz_rollouts_topic').get_parameter_value().string_value
        self._viz_rollouts_pub = self.create_publisher(Marker, self._viz_rollouts_topic, 10)



        # === MPC params ===
        self.declare_parameter('use_mpc', True)
        self.declare_parameter('mpc_autorun', True)
        self.declare_parameter('mpc_step_dt', 0.05) # 0.03 for pose control
        self.declare_parameter('mpc_cmd_topic', '/ur_arm_controller/joint_trajectory')  # change if your controller differs
        self.declare_parameter('mpc_world_update_period', 0.15)
        # Optional command smoothing to reduce jerkiness
        self.declare_parameter('mpc_cmd_smoothing_alpha', 0.35)
        self.declare_parameter('mpc_cmd_max_step', 0.08)
        # Progress watchdog: skip ahead if stalled
        self.declare_parameter('mpc_stall_ticks', 15)          # ticks without advancing before forcing jump
        self.declare_parameter('mpc_stall_jump_points', 3)     # points to jump ahead when stalled
        self._mg_path = None     # dict with EE xyz [N,3], s [N], q_mpc [N,DoF], names, etc.
        self._ema_pose_err = 0.0 # for adaptive look-ahead
        self._endgame = False
        self._cmd_alpha = float(self.get_parameter('mpc_cmd_smoothing_alpha').get_parameter_value().double_value)
        self._cmd_max_step = float(self.get_parameter('mpc_cmd_max_step').get_parameter_value().double_value)
        self._last_cmd_pos = None
        self._stall_ticks_thresh = int(self.get_parameter('mpc_stall_ticks').get_parameter_value().integer_value)
        self._stall_jump_pts = int(self.get_parameter('mpc_stall_jump_points').get_parameter_value().integer_value)
        if self._stall_ticks_thresh < 1:
            self._stall_ticks_thresh = 15
        if self._stall_jump_pts < 1:
            self._stall_jump_pts = 2
        self._stall_counter = 0

        # Look-ahead parameters
        self.declare_parameter('mpc_lookahead_m', 0.35)           # nominal 20 cm
        self.declare_parameter('mpc_min_lookahead_voxels', 4)     # >= 3 * voxel_size
        self._mpc_lookahead_m = self.get_parameter('mpc_lookahead_m').get_parameter_value().double_value
        self._mpc_min_lookahead_voxels = self.get_parameter('mpc_min_lookahead_voxels').get_parameter_value().integer_value
        self._goal = None
        # Direct single-pose goal (bypass MotionGen path streaming)
        self._direct_goal_pose = None

        # Forward-only progress tracking ---
        self._la_last_idx = 0        # last index we accepted (monotonic)
        self._s_progress = 0.0       # last arclength we accepted (monotonic)
        self._backtrack_pts = 5      # allow tiny look-back window to avoid getting stuck
        self._finish_margin_m = 0.02 # when this close to final EE, snap to the end
        self._last_goal_idx = -1     # last goal index sent to MPC
        self._s_tgt = 0.0            # latest target arclength used for lookahead
        self._last_goal_s = -1.0     # last arclength sent to MPC
        self._min_goal_step_s = 1e-3 # initialize; will update after __voxel_size is read

        self.__voxel_pub = self.create_publisher(Marker, '/curobo/voxels', 10)
        self.planner_busy = False
        self.lock = threading.Lock()
        self._esdf_lock = threading.Lock()
        self._esdf_future: Optional[Future] = None
        self._esdf_request_in_progress = False
        self._esdf_last_request_ts = 0.0
        self._esdf_last_success_ts = 0.0
        self._esdf_update_success = False
        self._esdf_timer = None
        self._esdf_timer_group = None
        self._pending_esdf_grid: Optional[CuVoxelGrid] = None

        self.__robot_file = self.get_parameter('robot').get_parameter_value().string_value

        try:
            self.__urdf_path = self.get_parameter('urdf_path').get_parameter_value().string_value or None
        except rclpy.exceptions.ParameterUninitializedException:
            self.__urdf_path = None

        try:
            self.__yml_path = self.get_parameter('yml_file_path').get_parameter_value().string_value or None
        except rclpy.exceptions.ParameterUninitializedException:
            self.__yml_path = None

        if self.__yml_path:
            self.__robot_file = self.__yml_path

        try:
            self.__tool_frame = self.get_parameter('tool_frame').get_parameter_value().string_value or None
        except rclpy.exceptions.ParameterUninitializedException:
            self.__tool_frame = None

        self.__joint_states_topic = self.get_parameter('joint_states_topic').get_parameter_value().string_value
        self.__add_ground_plane = self.get_parameter('add_ground_plane').get_parameter_value().bool_value
        self.__override_moveit_scaling_factors = (
            self.get_parameter('override_moveit_scaling_factors').get_parameter_value().bool_value
        )

        excluded_param = self.get_parameter('excluded_joint_names').get_parameter_value().string_array_value
        self._excluded_joint_names = set(excluded_param) if excluded_param else set()

        # Motion generation parameters
        self.__max_attempts = self.get_parameter('max_attempts').get_parameter_value().integer_value
        self.__num_graph_seeds = self.get_parameter('num_graph_seeds').get_parameter_value().integer_value
        self.__num_trajopt_seeds = self.get_parameter('num_trajopt_seeds').get_parameter_value().integer_value
        self.__num_trajopt_time_steps = self.get_parameter('num_trajopt_time_steps').get_parameter_value().integer_value
        self.__trajopt_finetune_iters = self.get_parameter('trajopt_finetune_iters').get_parameter_value().integer_value
        self.__interpolation_dt = self.get_parameter('interpolation_dt').get_parameter_value().double_value

        include_trajopt_retract_seed = (
            self.get_parameter('include_trajopt_retract_seed').get_parameter_value().bool_value
        )
        if include_trajopt_retract_seed:
            self.__num_trajopt_noisy_seeds = 1
            self.__trajopt_seed_ratio = {'linear': 1.0, 'bias': 0.0}
        else:
            self.__num_trajopt_noisy_seeds = 2
            self.__trajopt_seed_ratio = {'linear': 0.5, 'bias': 0.5}

        collision_cache_cuboid = self.get_parameter('collision_cache_cuboid').get_parameter_value().integer_value
        collision_cache_mesh = self.get_parameter('collision_cache_mesh').get_parameter_value().integer_value
        self.__collision_cache = {'obb': collision_cache_cuboid, 'mesh': collision_cache_mesh}

        # ESDF service
        self.__read_esdf_grid = self.get_parameter('read_esdf_world').get_parameter_value().bool_value
        self.__publish_curobo_world_as_voxels = (
            self.get_parameter('publish_curobo_world_as_voxels').get_parameter_value().bool_value
        )
        self.__grid_center_m = self.get_parameter('grid_center_m').get_parameter_value().double_array_value
        self.__max_publish_voxels = self.get_parameter('max_publish_voxels').get_parameter_value().integer_value
        self.__workspace_file_path = self.get_parameter('workspace_file_path').get_parameter_value().string_value
        self.__grid_size_m = self.get_parameter('grid_size_m').get_parameter_value().double_array_value
        self.__update_esdf_on_request = self.get_parameter('update_esdf_on_request').get_parameter_value().bool_value
        self.__use_aabb_on_request = self.get_parameter('use_aabb_on_request').get_parameter_value().bool_value
        self.__publish_voxel_size = self.get_parameter('publish_voxel_size').get_parameter_value().double_value
        self.__voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value
        # now that voxel size is known, set a sensible min arclength step for goal updates
        self._min_goal_step_s = max(1e-3, 0.5 * self.__voxel_size)
        self._update_link_sphere_server = self.get_parameter('update_link_sphere_server').get_parameter_value().string_value
        self.__esdf_client = None
        self.__esdf_req = None

        # Setup the grid position and dimension
        if path.exists(self.__workspace_file_path):
            self.get_logger().info(f'Loading grid center and dims from workspace file: {self.__workspace_file_path}.')
            min_corner, max_corner = load_grid_corners_from_workspace_file(self.__workspace_file_path)
            self.__grid_size_m = get_grid_size(min_corner, max_corner, self.__voxel_size)
            self.__grid_center_m = get_grid_center(min_corner, self.__grid_size_m)
            self.get_logger().info(f'Loaded grid dims: {self.__grid_size_m}, voxel size: {self.__voxel_size}')
        else:
            self.get_logger().info('Loading grid position and dims from grid_center_m and grid_size_m parameters.')

        if is_grid_valid(self.__grid_size_m, self.__voxel_size):
            self.get_logger().fatal('Number of voxels should be at least 1 in every dimension.')
            raise SystemExit

        if self.__read_esdf_grid:
            esdf_service_name = self.get_parameter('esdf_service_name').get_parameter_value().string_value
            esdf_service_cb_group = MutuallyExclusiveCallbackGroup()
            self.__esdf_client = self.create_client(EsdfAndGradients, esdf_service_name,
                                                    callback_group=esdf_service_cb_group)
            while not self.__esdf_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Service({esdf_service_name}) not available, waiting again...')
            self.__esdf_req = EsdfAndGradients.Request()

        # Load MG + warmup
        self.load_motion_gen()
        self.warmup()
        self.__query_count = 0
        self.__tensor_args = self.motion_gen.tensor_args

        # === MPC state ===
        self._use_mpc = self.get_parameter('use_mpc').get_parameter_value().bool_value
        self._mpc_autorun = self.get_parameter('mpc_autorun').get_parameter_value().bool_value
        self._mpc_step_dt = self.get_parameter('mpc_step_dt').get_parameter_value().double_value
        self._mpc_cmd_topic = self.get_parameter('mpc_cmd_topic').get_parameter_value().string_value
        self._mpc_world_update_period = self.get_parameter('mpc_world_update_period').get_parameter_value().double_value

        if self.__read_esdf_grid:
            period = max(self._mpc_world_update_period, 0.05)
            self._esdf_timer_group = ReentrantCallbackGroup()
            self._esdf_timer = self.create_timer(period, self._esdf_timer_callback, callback_group=self._esdf_timer_group)
            self._queue_esdf_request(force=True)

        self._mpc_active = False
        self._update_goal = False
        self._motion_gen_result = None
        self._last_world_update_ts = 0.0
        self.goal_buffer = None

        # ROS I/O
        self.subscription = self.create_subscription(
            JointState, self.__joint_states_topic, self.js_callback, 10
        )
        self.__js_buffer = None

        # Fast-topic pose goal (optional): publish PoseStamped to switch target without action
        self._pose_goal_sub = self.create_subscription(
            PoseStamped, 'cumotion/goal_pose', self.pose_goal_callback, 10
        )

        # Viz timer
        self.timer = self.create_timer(self._mpc_step_dt/2, self.viz_timer)

        # MPC publisher & timer
        self._mpc_cmd_pub = self.create_publisher(JointTrajectory, self._mpc_cmd_topic, 10)
        # Publisher to the standard MoveIt display topic used by RViz
        self._display_traj_pub = self.create_publisher(DisplayTrajectory, '/display_planned_path', 10)
        if self._use_mpc:
            self.load_mpc()
            self._mpc_timer = self.create_timer(self._mpc_step_dt, self.mpc_tick)

        self.__update_link_spheres_server = UpdateLinkSpheresServer(
            server_node=self,
            action_name=self._update_link_sphere_server,
            robot_kinematics=self.motion_gen.kinematics,
            robot_base_frame=self.__robot_base_frame
        )
        self._action_server = ActionServer(self, MoveGroup, 'cumotion/move_group', self.execute_callback)

    # ------------------- Callbacks / Helpers -------------------

    def _precompute_mg_path(self, traj: RobotTrajectory):
        """Precompute EE path and cumulative arclength for MG trajectory, and
        cache joint positions reordered to MPC joint order."""
        jt = traj.joint_trajectory
        assert len(jt.points) > 0
        moveit_jn = list(jt.joint_names)
        mpc_jn    = list(self.mpc.rollout_fn.joint_names)
        idx       = [moveit_jn.index(j) for j in mpc_jn]

        # Joint matrix [N, DoF] in MPC order
        q = np.array([p.positions for p in jt.points], dtype=np.float32)[:, idx]
        q_t = torch.from_numpy(q).to(device=self.mpc.tensor_args.device)
        q_t = q_t.contiguous()  # <-- add this

        # FK -> EE positions [N,3]
        js = CuJointState.from_position(position=q_t, joint_names=mpc_jn)
        ee = (self.motion_gen.compute_kinematics(js)
            .ee_pose.position
            .detach().cpu().numpy())

        # cumulative arclength s
        diffs = ee[1:] - ee[:-1]
        seg   = np.linalg.norm(diffs, axis=1)
        s     = np.concatenate([[0.0], np.cumsum(seg)])

        self._mg_path = {
            'q_mpc': q_t,       # torch [N,DoF]
            'ee': ee,           # np  [N,3]
            's': s,             # np  [N]
            'mpc_names': mpc_jn # list[str]
        }
        self._la_last_idx = 0
        self._s_progress = 0.0
        self._endgame = False # reset on new path
        self._last_cmd_pos = None

    def _publish_mppi_rollouts(self):
        if not self._viz_enable_mppi_rollouts or self._viz_rollouts_pub.get_subscription_count() < 1:
            return
        try:
            r = self.mpc.solver.get_rollouts()  # requires store_rollouts=True
        except Exception:
            return
        if r is None:
            return

         # Tensor directly from solver
        t = r
        # Accept [H,3] or [B,H,3]
        if t.ndim == 2 and t.shape[-1] == 3:
            ee = t.unsqueeze(0)
        elif t.ndim == 3 and t.shape[-1] == 3:
            ee = t
        elif t.ndim >= 2 and t.shape[-1] == len(self.mpc.rollout_fn.joint_names):
            # Looks like joint rollouts -> FK fallback
            if t.ndim == 2:
                t = t.unsqueeze(0)
            B, H, DoF = t.shape[:3]
            js = CuJointState.from_position(
                position=t.reshape(-1, DoF),
                joint_names=list(self.mpc.rollout_fn.joint_names),
            )
            ee_pose = self.motion_gen.compute_kinematics(js).ee_pose
            ee = ee_pose.position.contiguous().view(B, H, 3)
        
        # --- Build and publish the Marker ---

        B, H, _ = ee.shape
        stride_h = max(1, H // 30)   # ~30 points per rollout
        stride_b = 1                 # thin batches if needed

        m = Marker()
        m.header.frame_id = self.__robot_base_frame  # ensure this matches the frame of 'ee'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'mpc_rollouts'
        m.id = 0
        m.type = Marker.POINTS
        m.action = Marker.ADD
        m.scale.x = 0.01  # 1 cm
        m.scale.y = 0.01
        m.color.g = 1.0
        m.color.a = 0.9
        m.lifetime = Duration(seconds=0.4).to_msg()  # a touch longer; tweak to taste

        ee_cpu = ee.detach().cpu().numpy()
        for b in range(0, B, stride_b):
            for t in range(0, H, stride_h):
                x, y, z = ee_cpu[b, t, :]
                m.points.append(Point(x=float(x), y=float(y), z=float(z)))

        self._viz_rollouts_pub.publish(m)

    def js_callback(self, msg):
        self.__js_buffer = {'joint_names': msg.name, 'position': msg.position, 'velocity': msg.velocity}

    def pose_goal_callback(self, msg: PoseStamped):
        """Accept a PoseStamped target and feed it directly to MPC as a final goal.
        This bypasses global MotionGen planning and simply commands the controller
        to move towards the desired pose (even if unreachable).
        """
        try:
            if self.__js_buffer is None:
                self.get_logger().error('pose_goal: no JointState received yet; ignoring goal')
                return

            # Optionally refresh ESDF
            if self.__read_esdf_grid:
                try:
                    self.update_voxel_grid(force=True)
                except Exception as e:
                    self.get_logger().warn(f'pose_goal: ESDF update failed: {e}')

            # Build goal pose
            p = msg.pose.position
            q = msg.pose.orientation
            goal_pose = Pose.from_list([p.x, p.y, p.z, q.w, q.x, q.y, q.z], tensor_args=self.tensor_args)

            # Store as a direct goal for MPC and activate tracking
            with self.lock:
                self._direct_goal_pose = goal_pose
                self._motion_gen_result = None  # disable MG path tracking
                self._mg_path = None
                self.goal_buffer = None  # ensure fresh goal buffer on next tick
                self._update_goal = True
                self._last_goal_idx = -1
                self._last_goal_s = -1.0
                self._s_tgt = 0.0
                self._endgame = False
                self._mpc_active = bool(self._mpc_autorun) and self._use_mpc

            self.get_logger().info('pose_goal: set direct MPC goal (autorun=%s)' % self._mpc_active)
        except Exception as e:
            self.get_logger().error(f'pose_goal: exception: {e}')

    def load_motion_gen(self):
        tensor_args = self.tensor_args
        world_file = WorldConfig.from_dict({
            'voxel': {
                'world_voxel': {
                    'dims': self.__grid_size_m,
                    'pose': [0, 0, 0, 1, 0, 0, 0],
                    'voxel_size': self.__voxel_size,
                    'feature_dtype': torch.bfloat16,
                },
            },
        })
        self._world_file_for_mpc = world_file

        robot_config = get_robot_config(
            robot_file=self.__robot_file,
            urdf_file_path=self.__urdf_path,
            logger=self.get_logger()
        )
        robot_dict = robot_config['robot_cfg']
        self._robot_cfg_dict = robot_dict

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_dict,
            world_file,
            tensor_args,
            num_graph_seeds=self.__num_graph_seeds,
            num_trajopt_seeds=self.__num_trajopt_seeds,
            num_trajopt_noisy_seeds=1 if self.get_parameter('include_trajopt_retract_seed').get_parameter_value().bool_value else 2,
            trajopt_tsteps=self.__num_trajopt_time_steps,
            trajopt_seed_ratio={'linear': 1.0, 'bias': 0.0} if self.get_parameter('include_trajopt_retract_seed').get_parameter_value().bool_value else {'linear': 0.5, 'bias': 0.5},
            interpolation_dt=self.__interpolation_dt,
            collision_cache=self.__collision_cache,
            collision_checker_type=CollisionCheckerType.VOXEL,
            ee_link_name=self.__tool_frame,
            finetune_trajopt_iters=self.__trajopt_finetune_iters,
            num_ik_seeds=32,
            num_batch_ik_seeds=32,
            num_batch_trajopt_seeds=1,
            position_threshold=0.005,
            rotation_threshold=0.05,
            cspace_threshold=0.05,
            world_coll_checker=None,
            base_cfg_file='base_cfg.yml',
            particle_ik_file='particle_ik.yml',
            gradient_ik_file='gradient_ik.yml',
            graph_file='graph.yml',
            particle_trajopt_file='particle_trajopt.yml',
            gradient_trajopt_file='gradient_trajopt.yml',
            finetune_trajopt_file=None,
            interpolation_steps=1000,
            interpolation_type=InterpolateType.LINEAR_CUDA,
            use_cuda_graph=True,
            self_collision_check=True,
            self_collision_opt=True,
            evaluate_interpolated_trajectory=True,
            minimize_jerk=True,
            filter_robot_command=True,
            optimize_dt=True,
            project_pose_to_goal_frame=True,
        )

        self.motion_gen = MotionGen(motion_gen_config)
        self.__robot_base_frame = self.motion_gen.kinematics.base_link
        self.__world_collision = self.motion_gen.world_coll_checker
        if not self.__add_ground_plane:
            self.motion_gen.clear_world_cache()
        self.__cumotion_grid_shape = self.__world_collision.get_voxel_grid('world_voxel').get_grid_shape()[0]

    def load_mpc(self):
        mpc_config = MpcSolverConfig.load_from_robot_config(
            self._robot_cfg_dict,
            self._world_file_for_mpc,
            step_dt=self._mpc_step_dt,
            use_mppi=True,
            use_lbfgs=False,
            use_es=False,
            store_rollouts=True,
            collision_checker_type=CollisionCheckerType.VOXEL,
            use_cuda_graph=True,
            use_cuda_graph_metrics=True,
            use_cuda_graph_full_step=False,
            self_collision_check=True,
            collision_activation_distance=0.05,
            compute_metrics=True
        )

        self.mpc = MpcSolver(mpc_config)
        self.get_logger().info('MPC initialized (MPPI).')

    def warmup(self):
        self.get_logger().info('warming up cuMotion, wait until ready')
        self.motion_gen.warmup(enable_graph=True)
        self.get_logger().info('cuMotion is ready for planning queries!')

    def viz_timer(self):
        with self.lock:
            if self.__js_buffer is None:
                return
            js = np.copy(self.__js_buffer['position'])
            j_names = deepcopy(self.__js_buffer['joint_names'])
        self.__update_link_spheres_server.publish_all_active_spheres(
            robot_joint_states=js,
            robot_joint_names=j_names,
            tensor_args=self.__tensor_args,
            rgb=[0.0, 1.0, 1.0, 1.0]
        )
        self._publish_mppi_rollouts()


    def _pick_lookahead_index(self, current_ee_xyz: np.ndarray) -> int:
        """Return an index >= last accepted index, aiming at s_progress + lookahead."""
        if self._mg_path is None:
            return -1

        ee = self._mg_path['ee']
        s  = self._mg_path['s']
        N  = len(s)

        # --- nearest within a forward-biased window ---
        lb = max(0, self._la_last_idx - self._backtrack_pts)
        ub = N  # you can clamp to a forward window if you like (e.g. self._la_last_idx+400)

        # nearest search only in [lb, ub)
        d = np.linalg.norm(ee[lb:ub] - current_ee_xyz[None, :], axis=1)
        i_near = lb + int(np.argmin(d))

        # never move progress backward
        s_cur = max(float(s[i_near]), float(self._s_progress))

        # compute look-ahead distance (respect voxel size & adaptive reduction)
        min_la = max(self._mpc_lookahead_m,
                    self._mpc_min_lookahead_voxels * float(self.__voxel_size))
        la = max(min_la * (0.6 if self._ema_pose_err > 0.12 else 1.0), 0.05)
        # Clamp lookahead by remaining distance to goal for end smoothing
        end_dist = float(np.linalg.norm(current_ee_xyz - ee[-1]))
        la = min(la, max(0.5 * self._finish_margin_m, 0.6 * end_dist))

        s_tgt = min(s_cur + la, float(s[-1]))
        j = int(np.searchsorted(s, s_tgt, side='left'))
        # store for downstream interpolation/gating
        self._s_tgt = s_tgt

        # enforce forward motion
        j = max(j, i_near, self._la_last_idx)
        j = min(j, N - 1)

        # update monotonic progress (a touch of hysteresis)
        self._s_progress = max(self._s_progress, float(s[i_near]))
        self._la_last_idx = max(self._la_last_idx, j - 1)

        # snap to the very end if we're close to final
        if np.linalg.norm(current_ee_xyz - ee[-1]) <= self._finish_margin_m:
            return N - 1
        return j

    def mpc_tick(self):

        if not self._mpc_active or self.__js_buffer is None:
            return

        # Warn if controller topic has no subscribers (wrong topic/controller name)
        subs = self._mpc_cmd_pub.get_subscription_count()
        if subs < 1:
            # Only warn occasionally to avoid log spam
            if int(time.time() * 10) % 50 == 0:
                self.get_logger().warn(
                    f'MPC cmd topic "{self._mpc_cmd_topic}" has {subs} subscribers; '
                    f'controller may be on a different topic.'
                )

        now = time.time()
        if self.__read_esdf_grid:
            self._apply_pending_esdf_grid()

        # Current measured state from joint_states callback
        state = CuJointState.from_position(
            position=self.tensor_args.to_device(self.__js_buffer['position']).unsqueeze(0),
            joint_names=self.__js_buffer['joint_names'],
        )
        if self.__js_buffer['velocity'] and len(self.__js_buffer['velocity']) == len(self.__js_buffer['position']):
            state.velocity = self.tensor_args.to_device(self.__js_buffer['velocity']).unsqueeze(0)
        current_state = self.mpc.get_active_js(state)
        #self.get_logger().info(f'Current state: {current_state}')

        # Compute/update goal for MPC
        if self._motion_gen_result is not None:
            if self._mg_path is None:
                try:
                    self._precompute_mg_path(self._motion_gen_result)
                except Exception as e:
                    self.get_logger().warn(f'Failed to precompute MG path: {e}')
                    return

            # Current EE position
            ee_cur = self.motion_gen.compute_kinematics(current_state).ee_pose.position \
                        .detach().cpu().numpy().reshape(3)

            j_idx = self._pick_lookahead_index(ee_cur)

            # endgame freeze: once we hit the final waypoint, keep it there
            N = len(self._mg_path['s'])
            if j_idx == N - 1:
                self._endgame = True
                self.get_logger().info('MPC in endgame mode (final waypoint reached).')

            if self._endgame:
                j_idx = N - 1

            # If we haven't advanced index for a while, force a jump forward
            if j_idx <= self._last_goal_idx:
                self._stall_counter += 1
            else:
                self._stall_counter = 0
            if self._stall_counter >= self._stall_ticks_thresh and not self._endgame:
                forced_idx = min(N - 1, self._last_goal_idx + self._stall_jump_pts)
                if forced_idx > j_idx:
                    j_idx = forced_idx
                    # keep target arclength consistent with forced index
                    try:
                        self._s_tgt = float(self._mg_path['s'][j_idx])
                    except Exception:
                        pass
                self._stall_counter = 0

            # Build goal_js / goal_pose from MG path at j_idx using arclength interpolation
            q_path = self._mg_path['q_mpc']              # [N, DoF]
            s_path = self._mg_path['s']                  # [N]
            if j_idx <= 0:
                j0, j1, alpha = 0, 0, 1.0
            else:
                j1 = j_idx
                j0 = max(0, j1 - 1)
                ds = float(max(s_path[j1] - s_path[j0], 1e-6))
                alpha = float(np.clip((self._s_tgt - float(s_path[j0])) / ds, 0.0, 1.0))
            qj = (q_path[j0:j0+1, :] * (1.0 - alpha) + q_path[j1:j1+1, :] * alpha).contiguous()
            goal_js = CuJointState.from_position(position=qj, joint_names=self._mg_path['mpc_names'])
            goal_pose = self.motion_gen.compute_kinematics(goal_js).ee_pose.clone()

            retract = goal_js.position
            if retract.ndim == 1:
                retract = retract.unsqueeze(0)
            retract = retract.detach().clone().to(self.mpc.tensor_args.device)

            goal = Goal(
                current_state=current_state,
                goal_state=goal_js,
                goal_pose=goal_pose,
                retract_state=retract
            )
            
            # Always refresh goal for smooth streaming, but avoid re-allocating the buffer
            need_setup = (self.goal_buffer is None) or self._update_goal
            try:
                if not need_setup:
                    # Update goal buffer in-place if supported (attr or dict-like)
                    if hasattr(self.goal_buffer, 'goal_state'):
                        self.goal_buffer.goal_state = goal_js
                        if hasattr(self.goal_buffer, 'goal_pose'):
                            self.goal_buffer.goal_pose = goal_pose
                        if hasattr(self.goal_buffer, 'retract_state'):
                            self.goal_buffer.retract_state = retract
                    elif isinstance(self.goal_buffer, dict):
                        if 'goal_state' in self.goal_buffer:
                            self.goal_buffer['goal_state'] = goal_js
                        if 'goal_pose' in self.goal_buffer:
                            self.goal_buffer['goal_pose'] = goal_pose
                        if 'retract_state' in self.goal_buffer:
                            self.goal_buffer['retract_state'] = retract
                    else:
                        need_setup = True
            except Exception:
                need_setup = True

            # Ensure buffer is valid and update it atomically
            with self.lock:
                if need_setup or (self.goal_buffer is None):
                    # Fallback: recreate buffer on first use or if in-place update unsupported
                    self.goal_buffer = self.mpc.setup_solve_single(goal, 1)

                # pick joint vs pose mode
                self.mpc.enable_pose_cost(enable=True)
                self.mpc.enable_cspace_cost(enable=True)
                if self.goal_buffer is not None:
                    self.mpc.update_goal(self.goal_buffer)
                else:
                    self.get_logger().warn('MPC goal_buffer is None; skipping update this tick')

                self._goal = goal
                self._update_goal = False
                self._last_goal_idx = j_idx
                self._last_goal_s = float(self._s_tgt)
            # Report progress along the original plan (index within precomputed MG path)
            #try:
            #    total_steps = len(self._mg_path['s'])
            #except Exception:
            #    total_steps = None
            #if total_steps is not None and total_steps > 0:
            #    self.get_logger().info(f'Plan step: {j_idx + 1}/{total_steps} (index {j_idx})')
            #else:
            #    self.get_logger().info(f'Plan step index: {j_idx}')


            # === MPC step ===
            mpc_result = self.mpc.step(current_state, max_attempts=2)

            # Update EMA pose error for adaptive look-ahead (simple, robust)
            pe = float(mpc_result.metrics.pose_error.item())
            self._ema_pose_err = 0.8 * self._ema_pose_err + 0.2 * pe

            pose_error = pe
            rotation_error = mpc_result.metrics.rotation_error.item()
            #self.get_logger().info(f'MPC pose error: {pose_error}')
            #self.get_logger().info(f'MPC rotation error: {rotation_error}')

            cmd_state_full = mpc_result.js_action
            #self.get_logger().info(f'MPC command joint state: {cmd_state_full}')

            # Filter out any invalid joint states comparing with current_state
            current_joint_name_set = set(current_state.joint_names)
            valid_positions = []
            valid_velocities = []
            valid_names = []
            has_velocity = getattr(cmd_state_full, 'velocity', None) is not None
            for i, name in enumerate(cmd_state_full.joint_names):
                if name in current_joint_name_set and name not in self._excluded_joint_names:
                    valid_names.append(name)
                    valid_positions.append(cmd_state_full.position[0, i])
                    if has_velocity:
                        valid_velocities.append(cmd_state_full.velocity[0, i])

            if valid_positions:
                cmd_state_filtered = CuJointState.from_position(
                    position=torch.stack(valid_positions, dim=0).unsqueeze(0),
                    joint_names=valid_names
                )
                if has_velocity and valid_velocities:
                    cmd_state_filtered.velocity = torch.stack(valid_velocities, dim=0).unsqueeze(0)
            else:
                cmd_state_filtered = None
                self._last_cmd_pos = None
                if self._excluded_joint_names:
                    self.get_logger().warn('MPC command dropped after applying excluded_joint_names filter')

            #self.get_logger().info(f'MPC command filtered joint state: {cmd_state_filtered}')
            #self.get_logger().info(f'MPC result: {mpc_result}')

            # Publish command state to joint controller
            if cmd_state_filtered is not None:
                # Check if position tensor has NaN values
                if torch.isnan(cmd_state_filtered.position).any():
                    self.get_logger().warn('MPC command joint state contains NaN values')
                    return

                # Create joint trajectory message with position + velocity targets
                joint_trajectory_msg = JointTrajectory()
                joint_trajectory_msg.joint_names = list(cmd_state_filtered.joint_names)
                pt = JointTrajectoryPoint()
                # ensure CPU numpy
                pos_np = cmd_state_filtered.position[0].detach().cpu().numpy()
                # Apply rate limiting and exponential smoothing
                if self._last_cmd_pos is not None and len(self._last_cmd_pos) == len(pos_np):
                    delta = pos_np - self._last_cmd_pos
                    max_step = self._cmd_max_step
                    if max_step > 0.0:
                        delta = np.clip(delta, -max_step, max_step)
                    pos_np = (1.0 - self._cmd_alpha) * self._last_cmd_pos + self._cmd_alpha * (self._last_cmd_pos + delta)
                self._last_cmd_pos = pos_np
                pt.positions = pos_np.tolist()
                vel_tensor = getattr(cmd_state_filtered, 'velocity', None)
                if vel_tensor is not None:
                    pt.velocities = vel_tensor[0].detach().cpu().numpy().tolist()
                pt.time_from_start = Duration(seconds=self._mpc_step_dt).to_msg()
                joint_trajectory_msg.points.append(pt)
                self._mpc_cmd_pub.publish(joint_trajectory_msg)
                # Compute total execution time
                total_execution_time = time.time() - now
                self.get_logger().info(f'Total execution time: {total_execution_time:.4f} seconds')

        elif self._direct_goal_pose is not None:
            # Pose-only goal: push the single desired EE pose to MPC and step
            try:
                # Build goal using current state and the stored target pose
                retract = current_state.position
                if retract.ndim == 1:
                    retract = retract.unsqueeze(0)
                retract = retract.detach().clone().to(self.mpc.tensor_args.device)

                goal = Goal(
                    current_state=current_state,
                    goal_pose=self._direct_goal_pose,
                    retract_state=retract,
                )

                # Refresh or update goal buffer
                need_setup = (self.goal_buffer is None) or self._update_goal
                if need_setup:
                    with self.lock:
                        self.goal_buffer = self.mpc.setup_solve_single(goal, 1)
                        # Pose-only tracking
                        self.mpc.enable_pose_cost(enable=True)
                        self.mpc.enable_cspace_cost(enable=False)
                        if self.goal_buffer is not None:
                            self.mpc.update_goal(self.goal_buffer)
                        self._goal = goal
                        self._update_goal = False
                else:
                    # Update in-place when possible
                    try:
                        if hasattr(self.goal_buffer, 'goal_pose'):
                            self.goal_buffer.goal_pose = self._direct_goal_pose
                        elif isinstance(self.goal_buffer, dict) and 'goal_pose' in self.goal_buffer:
                            self.goal_buffer['goal_pose'] = self._direct_goal_pose
                    except Exception:
                        pass
                    with self.lock:
                        self.mpc.enable_pose_cost(enable=True)
                        self.mpc.enable_cspace_cost(enable=False)
                        if self.goal_buffer is not None:
                            self.mpc.update_goal(self.goal_buffer)
                        self._goal = goal
                        self._update_goal = False

                # === MPC step ===
                mpc_result = self.mpc.step(current_state, max_attempts=2)

                # Update EMA pose error (for diagnostics/consistency)
                pe = float(mpc_result.metrics.pose_error.item())
                self._ema_pose_err = 0.8 * self._ema_pose_err + 0.2 * pe

                cmd_state_full = mpc_result.js_action

                # Filter out any invalid joint states comparing with current_state
                current_joint_name_set = set(current_state.joint_names)
                valid_positions = []
                valid_velocities = []
                valid_names = []
                has_velocity = getattr(cmd_state_full, 'velocity', None) is not None
                for i, name in enumerate(cmd_state_full.joint_names):
                    if name in current_joint_name_set and name not in self._excluded_joint_names:
                        valid_names.append(name)
                        valid_positions.append(cmd_state_full.position[0, i])
                        if has_velocity:
                            valid_velocities.append(cmd_state_full.velocity[0, i])

                if valid_positions:
                    cmd_state_filtered = CuJointState.from_position(
                        position=torch.stack(valid_positions, dim=0).unsqueeze(0),
                        joint_names=valid_names
                    )
                    if has_velocity and valid_velocities:
                        cmd_state_filtered.velocity = torch.stack(valid_velocities, dim=0).unsqueeze(0)
                else:
                    cmd_state_filtered = None
                    self._last_cmd_pos = None
                    if self._excluded_joint_names:
                        self.get_logger().warn('MPC command dropped after applying excluded_joint_names filter')

                # Publish command state to joint controller
                if cmd_state_filtered is not None:
                    if torch.isnan(cmd_state_filtered.position).any():
                        self.get_logger().warn('MPC command joint state contains NaN values')
                        return
                    jt = JointTrajectory()
                    jt.joint_names = list(cmd_state_filtered.joint_names)
                    pt = JointTrajectoryPoint()
                    pos_np = cmd_state_filtered.position[0].detach().cpu().numpy()
                    # Apply rate limiting and exponential smoothing
                    if self._last_cmd_pos is not None and len(self._last_cmd_pos) == len(pos_np):
                        delta = pos_np - self._last_cmd_pos
                        max_step = self._cmd_max_step
                        if max_step > 0.0:
                            delta = np.clip(delta, -max_step, max_step)
                        pos_np = (1.0 - self._cmd_alpha) * self._last_cmd_pos + self._cmd_alpha * (self._last_cmd_pos + delta)
                    self._last_cmd_pos = pos_np
                    pt.positions = pos_np.tolist()
                    vel_tensor = getattr(cmd_state_filtered, 'velocity', None)
                    if vel_tensor is not None:
                        pt.velocities = vel_tensor[0].detach().cpu().numpy().tolist()
                    pt.time_from_start = Duration(seconds=self._mpc_step_dt).to_msg()
                    jt.points.append(pt)
                    self._mpc_cmd_pub.publish(jt)
            except Exception as e:
                self.get_logger().warn(f'Direct-goal MPC tick failed: {e}')


        # ------------------- ESDF / World -------------------

    def _esdf_timer_callback(self):
        self._queue_esdf_request()

    def _queue_esdf_request(self, force: bool = False):
        if not self.__read_esdf_grid or self.__esdf_client is None:
            return

        now = time.time()
        with self._esdf_lock:
            if self._esdf_request_in_progress:
                return
            if not force and (now - self._esdf_last_request_ts) < self._mpc_world_update_period:
                return
            self._esdf_request_in_progress = True
            self._esdf_last_request_ts = now
            self._esdf_update_success = False

        min_corner = get_grid_min_corner(self.__grid_center_m, self.__grid_size_m)
        aabb_min = Point(x=min_corner[0], y=min_corner[1], z=min_corner[2])
        aabb_size = Vector3(x=self.__grid_size_m[0], y=self.__grid_size_m[1], z=self.__grid_size_m[2])
        self.get_logger().info('Dispatching ESDF service request')
        try:
            future = self.send_request(aabb_min, aabb_size)
        except Exception as e:
            with self._esdf_lock:
                self._esdf_request_in_progress = False
                self._esdf_future = None
            self.get_logger().warn(f'Failed to dispatch ESDF request: {e}')
            return

        with self._esdf_lock:
            self._esdf_future = future
        future.add_done_callback(self._on_esdf_response)

    def _on_esdf_response(self, future: Future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().warn(f'ESDF request exception: {e}')
            success = False
        else:
            success = self._handle_esdf_response(response)

        with self._esdf_lock:
            self._esdf_request_in_progress = False
            self._esdf_future = None
            if success:
                self._esdf_last_success_ts = time.time()
                self._esdf_update_success = True
            else:
                self._esdf_update_success = False

        if success:
            self._last_world_update_ts = time.time()

    def update_voxel_grid(self, force: bool = False):
        if not self.__read_esdf_grid:
            return False
        self._queue_esdf_request(force=force)

        if force:
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if self._apply_pending_esdf_grid():
                    return True
                time.sleep(0.002)
            self.get_logger().warn('ESDF update timed out waiting for response')
            return False

        applied = self._apply_pending_esdf_grid()
        if applied:
            return True
        with self._esdf_lock:
            return self._esdf_update_success

    def _apply_pending_esdf_grid(self) -> bool:
        with self._esdf_lock:
            grid = self._pending_esdf_grid
            if grid is None:
                return False
            self._pending_esdf_grid = None

        with self.lock:
            self.__world_collision.update_voxel_data(grid)
            if hasattr(self, 'mpc') and self.mpc is not None:
                try:
                    self.mpc.world_collision.update_voxel_data(grid)
                except Exception as e:
                    self.get_logger().warn(f'Failed to update MPC voxel grid: {e}')

        self.get_logger().info('Updated ESDF grid')
        return True

    def _handle_esdf_response(self, response) -> bool:
        if not response.success:
            self.get_logger().info('ESDF request failed, try again after few seconds.')
            return False

        esdf_grid = self.get_esdf_voxel_grid(response)
        if torch.max(esdf_grid.feature_tensor) <= (-1000.0 + 0.5 * self.__voxel_size + 1e-5):
            self.get_logger().error('ESDF data is empty, try again after few seconds.')
            return False

        with self._esdf_lock:
            self._pending_esdf_grid = esdf_grid

        return True

    def send_request(self, aabb_min_m, aabb_size_m):
        self.__esdf_req.visualize_esdf = True
        self.__esdf_req.update_esdf = self.__update_esdf_on_request
        self.__esdf_req.use_aabb = self.__use_aabb_on_request
        self.__esdf_req.frame_id = self.__robot_base_frame
        self.__esdf_req.aabb_min_m = aabb_min_m
        self.__esdf_req.aabb_size_m = aabb_size_m
        self.get_logger().info(f'ESDF  req = {self.__esdf_req.aabb_min_m}, {self.__esdf_req.aabb_size_m}')
        return self.__esdf_client.call_async(self.__esdf_req)

    def get_esdf_voxel_grid(self, esdf_data):
        esdf_voxel_size = esdf_data.voxel_size_m
        if abs(esdf_voxel_size - self.__voxel_size) > 1e-4:
            self.get_logger().fatal(
                f'Voxel size mismatch: {esdf_voxel_size} vs. requested {self.__voxel_size}')
            raise SystemExit

        esdf_array = esdf_data.esdf_and_gradients
        array_shape = [esdf_array.layout.dim[0].size,
                       esdf_array.layout.dim[1].size,
                       esdf_array.layout.dim[2].size]
        array_data = np.array(esdf_array.data, dtype=np.float32)
        if array_data.shape[0] <= 0:
            self.get_logger().fatal('ESDF array shape is zero')
            raise SystemExit
        array_data = torch.as_tensor(array_data)

        if array_shape != self.__cumotion_grid_shape:
            self.get_logger().fatal(
                f'ESDF shape mismatch vs cuMotion grid: {array_shape} vs {self.__cumotion_grid_shape}')
            raise SystemExit

        grid_origin = [esdf_data.origin_m.x, esdf_data.origin_m.y, esdf_data.origin_m.z]
        grid_center_m = get_grid_center(grid_origin, self.__grid_size_m)

        array_data = array_data.view(array_shape[0], array_shape[1], array_shape[2]).contiguous()
        array_data = array_data.reshape(-1, 1)

        array_data[array_data < -999.9] = 1000.0  # unobserved -> far
        array_data = -1.0 * array_data            # sign flip
        array_data += 0.5 * self.__voxel_size     # surface offset

        esdf_grid = CuVoxelGrid(
            name='world_voxel',
            dims=self.__grid_size_m,
            pose=grid_center_m + [1, 0.0, 0.0, 0.0],
            voxel_size=self.__voxel_size,
            feature_dtype=torch.float32,
            feature_tensor=array_data,
        )
        return esdf_grid

    def get_cumotion_collision_object(self, mv_object: CollisionObject):
        objs, supported_objects = [], True
        world_pose = Pose.from_list([
            mv_object.pose.position.x, mv_object.pose.position.y, mv_object.pose.position.z,
            mv_object.pose.orientation.w, mv_object.pose.orientation.x,
            mv_object.pose.orientation.y, mv_object.pose.orientation.z
        ])
        if len(mv_object.primitives) > 0:
            for k, prim in enumerate(mv_object.primitives):
                pose = mv_object.primitive_poses[k]
                primitive_pose = [pose.position.x, pose.position.y, pose.position.z,
                                  pose.orientation.w, pose.orientation.x,
                                  pose.orientation.y, pose.orientation.z]
                object_pose = world_pose.multiply(Pose.from_list(primitive_pose)).tolist()
                if prim.type == SolidPrimitive.BOX:
                    objs.append(Cuboid(name=f'{mv_object.id}_{k}_cuboid', pose=object_pose, dims=prim.dimensions))
                elif prim.type == SolidPrimitive.SPHERE:
                    r = prim.dimensions[prim.SPHERE_RADIUS]
                    objs.append(Sphere(name=f'{mv_object.id}_{k}_sphere', pose=object_pose, radius=r))
                elif prim.type == SolidPrimitive.CYLINDER:
                    h = prim.dimensions[prim.CYLINDER_HEIGHT]
                    r = prim.dimensions[prim.CYLINDER_RADIUS]
                    objs.append(Cylinder(name=f'{mv_object.id}_{k}_cylinder', pose=object_pose, height=h, radius=r))
                elif prim.type == SolidPrimitive.CONE:
                    self.get_logger().error('Cone primitive is not supported'); supported_objects = False
                else:
                    self.get_logger().error('Unknown primitive type'); supported_objects = False
        if len(mv_object.meshes) > 0:
            for k, mesh in enumerate(mv_object.meshes):
                pose = mv_object.mesh_poses[k]
                mesh_pose = [pose.position.x, pose.position.y, pose.position.z,
                             pose.orientation.w, pose.orientation.x,
                             pose.orientation.y, pose.orientation.z]
                object_pose = world_pose.multiply(Pose.from_list(mesh_pose)).tolist()
                verts = [[v.x, v.y, v.z] for v in mesh.vertices]
                tris = [[t.vertex_indices[0], t.vertex_indices[1], t.vertex_indices[2]]
                        for t in mesh.triangles]
                objs.append(Mesh(name=f'{mv_object.id}_{len(objs)}_mesh',
                                 pose=object_pose, vertices=verts, faces=tris))
        return objs, supported_objects

    def get_joint_trajectory(self, js: CuJointState, dt: float):
        traj = RobotTrajectory()
        cmd_traj = JointTrajectory()
        q_traj = js.position.cpu().contiguous().view(-1, js.position.shape[-1]).numpy()

        vel = None
        if getattr(js, 'velocity', None) is not None:
            vel = js.velocity.cpu().contiguous().view(-1, js.position.shape[-1]).numpy()

        acc = None
        if getattr(js, 'acceleration', None) is not None:
            acc = js.acceleration.cpu().contiguous().view(-1, js.position.shape[-1]).numpy()

        for i in range(len(q_traj)):
            traj_pt = JointTrajectoryPoint()
            traj_pt.positions = q_traj[i].tolist()
            if vel is not None and i < len(vel):
                traj_pt.velocities = vel[i].tolist()
            if acc is not None and i < len(acc):
                traj_pt.accelerations = acc[i].tolist()
            traj_pt.time_from_start = Duration(seconds=i * dt).to_msg()
            cmd_traj.points.append(traj_pt)

        cmd_traj.joint_names = js.joint_names
        cmd_traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_trajectory = cmd_traj
        return traj

    def update_world_objects(self, moveit_objects):
        world_update_status = True
        if len(moveit_objects) > 0:
            cuboid_list, sphere_list, cylinder_list, mesh_list = [], [], [], []
            for obj in moveit_objects:
                cumotion_objects, world_update_status = self.get_cumotion_collision_object(obj)
                for co in cumotion_objects:
                    if   isinstance(co, Cuboid):   cuboid_list.append(co)
                    elif isinstance(co, Cylinder): cylinder_list.append(co)
                    elif isinstance(co, Sphere):   sphere_list.append(co)
                    elif isinstance(co, Mesh):     mesh_list.append(co)

            world_model = WorldConfig(cuboid=cuboid_list, cylinder=cylinder_list,
                                      sphere=sphere_list, mesh=mesh_list).get_collision_check_world()
            self.motion_gen.update_world(world_model)
            if hasattr(self, 'mpc') and self.mpc is not None:
                try:
                    self.mpc.update_world(world_model)
                except Exception as e:
                    self.get_logger().warn(f'Failed to update MPC world (meshes/primitives): {e}')
        if self.__read_esdf_grid:
            world_update_status = self.update_voxel_grid(force=True)
        if self.__publish_curobo_world_as_voxels and self.__voxel_pub.get_subscription_count() > 0:
            voxels = self.__world_collision.get_occupancy_in_bounding_box(
                Cuboid(name='test', pose=[0.0, 0.0, 0.0, 1, 0, 0, 0], dims=self.__grid_size_m),
                voxel_size=self.__publish_voxel_size,
            )
            xyzr_tensor = voxels.xyzr_tensor.clone()
            xyzr_tensor[..., 3] = voxels.feature_tensor
            self.publish_voxels(xyzr_tensor)
        return world_update_status

    # ------------------- Action -------------------

    def execute_callback(self, goal_handle):
        start_time = time.time()

        # TODO: Implement stopping when new goal is received
        if self.planner_busy:
            self.get_logger().error('Planner is busy')
            goal_handle.abort()
            result = MoveGroup.Result()
            result.error_code.val = MoveItErrorCodes.FAILURE
            return result

        self.get_logger().info('Executing goal...')

        # Scaling factors
        min_scaling = min(
            goal_handle.request.request.max_velocity_scaling_factor,
            goal_handle.request.request.max_acceleration_scaling_factor
        )
        time_dilation_factor = min(1.0, min_scaling)
        if time_dilation_factor <= 0.0 or self.__override_moveit_scaling_factors:
            time_dilation_factor = self.get_parameter('time_dilation_factor').get_parameter_value().double_value
        self.get_logger().info(f'Planning with time_dilation_factor: {time_dilation_factor}')

        plan_req = goal_handle.request.request

        # World/ESDF update for global planning
        scene = goal_handle.request.planning_options.planning_scene_diff
        world_objects = scene.world.collision_objects
        if not self.update_world_objects(world_objects):
            result = MoveGroup.Result()
            result.error_code.val = MoveItErrorCodes.COLLISION_CHECKING_UNAVAILABLE
            self.get_logger().error('World update failed.')
            return result

        # Start state for global planning
        if len(plan_req.start_state.joint_state.position) > 0:
            self.get_logger().info('Calculating start state from request')
            start_state = self.motion_gen.get_active_js(
                CuJointState.from_position(
                    position=self.tensor_args.to_device(plan_req.start_state.joint_state.position).unsqueeze(0),
                    joint_names=plan_req.start_state.joint_state.name,
                )
            )
        else:
            self.get_logger().info('Start state empty; reading current /joint_states')
            if self.__js_buffer is None:
                self.get_logger().error('No JointState received from ' + self.__joint_states_topic)
                result = MoveGroup.Result()
                result.error_code.val = MoveItErrorCodes.FAILURE
                return result
            state = CuJointState.from_position(
                position=self.tensor_args.to_device(self.__js_buffer['position']).unsqueeze(0),
                joint_names=self.__js_buffer['joint_names'],
            )
            if self.__js_buffer['velocity'] and len(self.__js_buffer['velocity']) == len(self.__js_buffer['position']):
                state.velocity = self.tensor_args.to_device(self.__js_buffer['velocity']).unsqueeze(0)
            start_state = self.motion_gen.get_active_js(state)

        # Goal (joint or pose)
        # JOINT GOAL
        if len(plan_req.goal_constraints[0].joint_constraints) > 0:
            self.get_logger().info('Goal from joint target')
            goal_config = [jc.position for jc in plan_req.goal_constraints[0].joint_constraints]
            goal_jnames = [jc.joint_name for jc in plan_req.goal_constraints[0].joint_constraints]
            goal_state = self.motion_gen.get_active_js(
                CuJointState.from_position(
                    position=self.tensor_args.to_device(goal_config).contiguous().view(1, -1),
                    joint_names=goal_jnames,
                )
            )
            goal_pose = self.motion_gen.compute_kinematics(goal_state).ee_pose.clone()
        # POSE GOAL
        elif (len(plan_req.goal_constraints[0].position_constraints) > 0
              and len(plan_req.goal_constraints[0].orientation_constraints) > 0):
            self.get_logger().info('Goal from pose')
            position = plan_req.goal_constraints[0].position_constraints[0].constraint_region.primitive_poses[0].position
            orientation = plan_req.goal_constraints[0].orientation_constraints[0].orientation
            pose_list = [position.x, position.y, position.z, orientation.w, orientation.x, orientation.y, orientation.z]
            goal_pose = Pose.from_list(pose_list, tensor_args=self.tensor_args)
            position_link_name = plan_req.goal_constraints[0].position_constraints[0].link_name
            orientation_link_name = plan_req.goal_constraints[0].orientation_constraints[0].link_name
            plan_link_name = self.motion_gen.kinematics.ee_link
            if position_link_name != orientation_link_name:
                result = MoveGroup.Result()
                result.error_code.val = MoveItErrorCodes.INVALID_LINK_NAME
                self.get_logger().error('Position and orientation link names do not match')
                return result
            if position_link_name != plan_link_name:
                result = MoveGroup.Result()
                result.error_code.val = MoveItErrorCodes.INVALID_LINK_NAME
                self.get_logger().error('Pose link does not match planning EE link; relaunch with tool_frame set accordingly')
                return result
        else:
            result = MoveGroup.Result()
            result.error_code.val = MoveItErrorCodes.PLANNING_FAILED
            self.get_logger().error('Unsupported goal constraints')
            return result

        with self.lock:
            self.planner_busy = True

        # Generate global plan trajectory using MotionGen
        self.motion_gen.reset(reset_seed=False)
        motion_gen_result = self.motion_gen.plan_single(
            start_state,
            goal_pose,
            MotionGenPlanConfig(
                max_attempts=self.__max_attempts,
                enable_graph_attempt=3,
                time_dilation_factor=time_dilation_factor,
                ik_fail_return=5,
            ),
        )

        with self.lock:
            self.planner_busy = False

        result = MoveGroup.Result()
        if motion_gen_result.success.item():
            result.error_code.val = MoveItErrorCodes.SUCCESS

            # Build a trajectory (for visualization/logging)
            traj = self.get_joint_trajectory(
                motion_gen_result.optimized_plan,
                float(motion_gen_result.optimized_dt.item())
            )

            # Echo MoveIt's own start_state when provided (prevents plan-only segfaults)
            if len(plan_req.start_state.joint_state.name) > 0:
                result.trajectory_start = plan_req.start_state

            result.planned_trajectory = traj
            result.planning_time = float(motion_gen_result.total_time)

            goal_handle.succeed()

            # Prepare MPC reference and start tracking (MoveIt execution should be disabled in your launch)
            if self._use_mpc:
                with self.lock:
                    self._motion_gen_result = traj
                    self._mpc_active = bool(self._mpc_autorun)
                    # Force recreation of goal buffer for new global trajectory
                    self.goal_buffer = None
                    self._update_goal = True
                    # Reset streaming trackers
                    self._last_goal_idx = -1
                    self._last_goal_s = -1.0
                    self._s_tgt = 0.0
                    self._precompute_mg_path(traj)
                self.get_logger().info(
                    f'MPC reference loaded: optimized plan={traj}; autorun={self._mpc_active}'
                )
        elif not motion_gen_result.valid_query:
            self.get_logger().error(f'Invalid planning query: {motion_gen_result.status}')
            if motion_gen_result.status == MotionGenStatus.INVALID_START_STATE_JOINT_LIMITS:
                result.error_code.val = MoveItErrorCodes.START_STATE_INVALID
            elif motion_gen_result.status in [
                MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION,
                MotionGenStatus.INVALID_START_STATE_SELF_COLLISION,
            ]:
                result.error_code.val = MoveItErrorCodes.START_STATE_IN_COLLISION
            else:
                result.error_code.val = MoveItErrorCodes.PLANNING_FAILED
        else:
            self.get_logger().error(f'Planning failed: {motion_gen_result.status}')
            if motion_gen_result.status == MotionGenStatus.IK_FAIL:
                result.error_code.val = MoveItErrorCodes.NO_IK_SOLUTION
            else:
                result.error_code.val = MoveItErrorCodes.PLANNING_FAILED

        self.get_logger().info(
            f'returned planning result (query, success, failure_status): '
            f'{self.__query_count} {motion_gen_result.success.item()} {motion_gen_result.status}'
        )
        self.__query_count += 1
        self.get_logger().info(f'Total execution time for execute_callback: {time.time() - start_time:.4f} seconds')
        return result

    def publish_voxels(self, voxels):
        vox_size = self.__publish_voxel_size
        marker = Marker()
        marker.header.frame_id = self.__robot_base_frame
        marker.id = 0
        marker.type = 6  # cube list
        marker.ns = 'curobo_world'
        marker.action = 0
        marker.pose.orientation.w = 1.0
        marker.lifetime = Duration(seconds=0.0).to_msg()
        marker.frame_locked = False
        marker.scale.x = vox_size; marker.scale.y = vox_size; marker.scale.z = vox_size
        marker.points = []

        voxels = voxels[voxels[:, 3] > 0.0]
        vox = voxels.contiguous().view(-1, 4).cpu().numpy()
        n = min(len(vox), self.__max_publish_voxels)
        marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0; marker.color.a = 1.0
        for i in range(n):
            pt = Point(x=float(vox[i, 0]), y=float(vox[i, 1]), z=float(vox[i, 2]))
            marker.points.append(pt)
        marker.header.stamp = self.get_clock().now().to_msg()
        self.__voxel_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = CumotionActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down.\n')
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
