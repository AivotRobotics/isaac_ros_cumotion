# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# See original header in your file.

from copy import deepcopy
from os import path
import threading
import time
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
from curobo.util.logger import setup_curobo_logger
from curobo.util.trajectory import InterpolateType
from curobo.wrap.reacher.motion_gen import (
    MotionGen, MotionGenConfig, MotionGenPlanConfig, MotionGenStatus
)

# === MPC ===
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.rollout.rollout_base import Goal

from geometry_msgs.msg import Point, Vector3
from isaac_ros_cumotion.update_kinematics import get_robot_config, UpdateLinkSpheresServer
from isaac_ros_cumotion_python_utils.utils import (
    get_grid_center, get_grid_min_corner, get_grid_size, is_grid_valid,
    load_grid_corners_from_workspace_file
)

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import CollisionObject, MoveItErrorCodes, RobotTrajectory, RobotState, DisplayTrajectory

from nvblox_msgs.srv import EsdfAndGradients
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker

from rclpy.duration import Duration
from rclpy.time import Time


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


        # --- Visualization params/publishers ---
        self.declare_parameter('viz_enable_display', True)     # publish MoveIt DisplayTrajectory
        self.declare_parameter('viz_display_topic', '/display_planned_path')
        self.declare_parameter('viz_enable_ee_marker', False)  # optional: end-effector LINE_STRIP
        self.declare_parameter('viz_ee_topic', '/mpc_ee_path')

        self._viz_enable_display = self.get_parameter('viz_enable_display').get_parameter_value().bool_value
        self._viz_display_topic = self.get_parameter('viz_display_topic').get_parameter_value().string_value
        self._viz_enable_ee_marker = self.get_parameter('viz_enable_ee_marker').get_parameter_value().bool_value
        self._viz_ee_topic = self.get_parameter('viz_ee_topic').get_parameter_value().string_value

        self._viz_display_pub = self.create_publisher(DisplayTrajectory, self._viz_display_topic, 10)
        self._viz_ee_pub = self.create_publisher(Marker, self._viz_ee_topic, 10)


        # === MPC params ===
        self.declare_parameter('use_mpc', True)
        self.declare_parameter('mpc_autorun', True)
        self.declare_parameter('mpc_step_dt', 0.1) #(10 Hz)
        self.declare_parameter('mpc_cmd_topic', '/ur_arm_controller/joint_trajectory')  # change if your controller differs
        self.declare_parameter('mpc_world_update_period', 0.10)

        self.__voxel_pub = self.create_publisher(Marker, '/curobo/voxels', 10)
        self.planner_busy = False
        self.lock = threading.Lock()

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

        self._mpc_active = False
        self._update_goal = False
        self._motion_gen_result = None
        self._mpc_goal_buf = None
        self._last_world_update_ts = 0.0

        # ROS I/O
        self.subscription = self.create_subscription(
            JointState, self.__joint_states_topic, self.js_callback, 10
        )
        self.__js_buffer = None

        # Viz timer
        self.timer = self.create_timer(0.01, self.on_timer)

        # MPC publisher & timer
        self._mpc_cmd_pub = self.create_publisher(JointTrajectory, self._mpc_cmd_topic, 10)
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

    def _publish_display_from_joint_traj(self, jt: JointTrajectory, start_state_js: CuJointState = None):
        if not self._viz_enable_display or self._viz_display_pub.get_subscription_count() < 1:
            return
        disp = DisplayTrajectory()

        # trajectory_start: use measured/current as a safe start
        rs = RobotState()
        try:
            if start_state_js is not None:
                rs = self._robot_state_from_js(start_state_js)
            else:
                # fall back to current /joint_states
                if self.__js_buffer:
                    rs.joint_state.name = list(self.__js_buffer['joint_names'])
                    rs.joint_state.position = list(self.__js_buffer['position'])
        except Exception:
            pass
        disp.trajectory_start = rs

        rtraj = RobotTrajectory()
        rtraj.joint_trajectory = jt
        disp.trajectory = [rtraj]
        self._viz_display_pub.publish(disp)

    def _publish_ee_marker_from_chunk(self, q_list, joint_names_ctrl):
        if not self._viz_enable_ee_marker or self._viz_ee_pub.get_subscription_count() < 1:
            return
        # Build LINE_STRIP in robot base frame
        m = Marker()
        m.header.frame_id = self.__robot_base_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = '' # TODO
        m.id = 1
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.005  # 5mm line
        m.color.r = 0.1; m.color.g = 0.8; m.color.b = 1.0; m.color.a = 0.9

        # Sample sparsely for speed
        stride = max(1, len(q_list) // 100)
        for k in range(0, len(q_list), stride):
            qk = q_list[k]
            try:
                js = CuJointState.from_position(
                    position=self.tensor_args.to_device(qk).view(1, -1),
                    joint_names=joint_names_ctrl
                )
                ee = self.motion_gen.compute_kinematics(js).ee_pose
                x, y, z = ee.position()[0].tolist()
                m.points.append(Point(x=float(x), y=float(y), z=float(z)))
            except Exception:
                continue

        self._viz_ee_pub.publish(m)


    def js_callback(self, msg):
        self.__js_buffer = {'joint_names': msg.name, 'position': msg.position, 'velocity': msg.velocity}

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
            #collision_activation_distance=0.004,
            compute_metrics=True
        )

        self.mpc = MpcSolver(mpc_config)
        self.get_logger().info('MPC initialized (MPPI).')

    def warmup(self):
        self.get_logger().info('warming up cuMotion, wait until ready')
        self.motion_gen.warmup(enable_graph=True)
        self.get_logger().info('cuMotion is ready for planning queries!')

    def on_timer(self):
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

    # --- Build a safe RobotState for MoveIt result to avoid segfaults ---
    def _robot_state_from_js(self, js: CuJointState) -> RobotState:
        rs = RobotState()
        rs.joint_state.name = list(js.joint_names)
        pos = js.position.view(-1, js.position.shape[-1]).detach().cpu().numpy()
        rs.joint_state.position = pos[-1].tolist()
        # only attach velocity if sizes match (MoveIt is picky)
        if js.velocity is not None and js.velocity.numel() == js.position.numel():
            vel = js.velocity.view(-1, js.position.shape[-1]).detach().cpu().numpy()
            rs.joint_state.velocity = vel[-1].tolist()
        return rs


    def build_seed_tensor_from_moveit(self, jt, mpc) -> torch.Tensor:
        """Return a [1, H, DoF] tensor in MPC joint order, interpolated to MPC horizon."""
        moveit_jn = list(jt.joint_names)
        mpc_jn    = list(mpc.rollout_fn.joint_names)
        idx = [moveit_jn.index(j) for j in mpc_jn]  # reorder columns to MPC order

        pts = jt.points
        assert len(pts) > 0, "Empty trajectory"

        # source times (s) and positions [N, DoF_mpc]
        t_src = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in pts], dtype=np.float32)
        # ensure strictly non-decreasing (handles duplicated 0.0 stamps)
        t_src = np.maximum.accumulate(t_src + np.linspace(0, 1e-6*(len(pts)-1), len(pts), dtype=np.float32))
        q_src = np.array([p.positions for p in pts], dtype=np.float32)[:, idx]

        # MPC horizon/dt introspection
        rf = mpc.rollout_fn
        H  = getattr(rf, "horizon", getattr(rf, "traj_T", getattr(rf, "n_steps", len(pts))))
        dt = getattr(rf, "dt", getattr(rf, "traj_dt", getattr(rf, "base_dt", None)))

        if dt is not None:
            t_tgt = np.arange(H, dtype=np.float32) * float(dt)
            if t_src[-1] <= 0:
                t_tgt[:] = 0.0
        else:
            t_end = float(t_src[-1]) if t_src[-1] > 0 else 1e-3
            t_tgt = np.linspace(0.0, t_end, H, dtype=np.float32)

        # interpolate onto MPC horizon
        dof = q_src.shape[1]
        q_seed = np.empty((H, dof), dtype=np.float32)
        for j in range(dof):
            q_seed[:, j] = np.interp(t_tgt, t_src, q_src[:, j])

        # to torch: [1, H, DoF] on the right device
        return torch.from_numpy(q_seed).to(device=mpc.tensor_args.device) \
                                    .unsqueeze(0)  # [1,H,DoF]


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
        if self.__read_esdf_grid and (now - self._last_world_update_ts) >= self._mpc_world_update_period:
            # World/ESDF update for local planning
            world_objects = []
            if self.update_world_objects(world_objects):
                self._last_world_update_ts = now

        # Current measured state from joint_states callback
        state = CuJointState.from_position(
            position=self.tensor_args.to_device(self.__js_buffer['position']).unsqueeze(0),
            joint_names=self.__js_buffer['joint_names'],
        )
        if self.__js_buffer['velocity'] and len(self.__js_buffer['velocity']) == len(self.__js_buffer['position']):
            state.velocity = self.tensor_args.to_device(self.__js_buffer['velocity']).unsqueeze(0)
        current_state = self.mpc.get_active_js(state)
        self.get_logger().info(f'Current state: {current_state}')

        # Compute goal from motion_gen_result last point
        if self._motion_gen_result is not None:
            last_point = self._motion_gen_result.joint_trajectory.points[-1]
            self.get_logger().info(f'Last point: {last_point}')

            pos = torch.as_tensor(last_point.positions, dtype=torch.float32)
            pos = pos.unsqueeze(0).to(self.mpc.tensor_args.device)

            goal_js = CuJointState.from_position(
                position=pos,
                joint_names=self._motion_gen_result.joint_trajectory.joint_names
            )
            self.get_logger().info(f'Goal state: {goal_js}')

            # compute goal pose using forward kinematics
            goal_pose = self.motion_gen.compute_kinematics(goal_js).ee_pose.clone()

            if self._update_goal:
                goal = Goal(
                    current_state=current_state,
                    goal_state=goal_js,
                    goal_pose=goal_pose
                )

                self.goal_buffer = self.mpc.setup_solve_single(goal, 1)
                self.goal_buffer.goal_state.copy_(goal_js)
                self.mpc.update_goal(self.goal_buffer)
                self._update_goal = False

            seed_traj = self.build_seed_tensor_from_moveit(self._motion_gen_result.joint_trajectory, self.mpc)
            mpc_result = self.mpc.step(current_state, max_attempts=2)
            #current_error = mpc_result.metrics.pose_error.item()

            cmd_state_full = mpc_result.js_action
            self.get_logger().info(f'MPC command joint state: {cmd_state_full}')

            # Filter out any invalid joint states comparing with current_state
            valid_positions = []
            valid_names = []
            for i, name in enumerate(cmd_state_full.joint_names):
                if name in current_state.joint_names:
                    valid_names.append(name)
                    valid_positions.append(cmd_state_full.position[0, i])

            cmd_state_filtered = CuJointState.from_position(
                position=torch.stack(valid_positions, dim=0).unsqueeze(0),
                joint_names=valid_names
            )

            self.get_logger().info(f'MPC command filtered joint state: {cmd_state_filtered}')
            self.get_logger().info(f'MPC result: {mpc_result}')

            # Publish command state to joint controller
            if cmd_state_filtered is not None:
                # Check if position tensor has NaN values
                if torch.isnan(cmd_state_filtered.position).any():
                    self.get_logger().warn('MPC command joint state contains NaN values')
                    return

                # Create joint trajectory message
                joint_trajectory_msg = JointTrajectory()
                joint_trajectory_msg.joint_names = cmd_state_filtered.joint_names
                for i in range(len(cmd_state_filtered.position)):
                    joint_trajectory_msg.points.append(JointTrajectoryPoint(
                        positions=cmd_state_filtered.position[i],
                        velocities=cmd_state_filtered.velocity[i],
                        time_from_start=Duration(seconds=self._mpc_step_dt).to_msg()
                    ))
                self._mpc_cmd_pub.publish(joint_trajectory_msg)

        # ------------------- ESDF / World -------------------

    def update_voxel_grid(self):
        self.get_logger().info('Calling ESDF service')
        min_corner = get_grid_min_corner(self.__grid_center_m, self.__grid_size_m)
        aabb_min = Point(x=min_corner[0], y=min_corner[1], z=min_corner[2])
        aabb_size = Vector3(x=self.__grid_size_m[0], y=self.__grid_size_m[1], z=self.__grid_size_m[2])
        esdf_future = self.send_request(aabb_min, aabb_size)
        while not esdf_future.done():
            time.sleep(0.001)
        response = esdf_future.result()
        if not response.success:
            self.get_logger().info('ESDF request failed, try again after few seconds.')
            return False
        esdf_grid = self.get_esdf_voxel_grid(response)
        if torch.max(esdf_grid.feature_tensor) <= (-1000.0 + 0.5 * self.__voxel_size + 1e-5):
            self.get_logger().error('ESDF data is empty, try again after few seconds.')
            return False
        self.__world_collision.update_voxel_data(esdf_grid)
        if hasattr(self, 'mpc') and self.mpc is not None:
            try:
                self.mpc.world_collision.update_voxel_data(esdf_grid)
            except Exception as e:
                self.get_logger().warn(f'Failed to update MPC voxel grid: {e}')
        self.get_logger().info('Updated ESDF grid')
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
        q_traj = js.position.cpu().view(-1, js.position.shape[-1]).numpy()

        vel = None
        if getattr(js, 'velocity', None) is not None:
            vel = js.velocity.cpu().view(-1, js.position.shape[-1]).numpy()

        acc = None
        if getattr(js, 'acceleration', None) is not None:
            acc = js.acceleration.view(-1, js.position.shape[-1]).cpu().numpy()

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
            world_update_status = self.update_voxel_grid()
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
                    position=self.tensor_args.to_device(goal_config).view(1, -1),
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
                self._motion_gen_result = traj
                self._mpc_active = bool(self._mpc_autorun)
                self._update_goal = True
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
        vox = voxels.view(-1, 4).cpu().numpy()
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
