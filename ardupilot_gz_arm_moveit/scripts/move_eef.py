#!/usr/bin/env python3
import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
import tf2_ros
import tf_transformations
from tf2_ros import TransformException
from geometry_msgs.msg import Pose, PoseStamped, Vector3
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints, JointConstraint, MotionPlanRequest, MoveItErrorCodes,
    OrientationConstraint, PlanningOptions, PositionConstraint,
)
from sensor_msgs.msg import CameraInfo, Image
from shape_msgs.msg import SolidPrimitive

BASE_FRAME, EEF_LINK = "base_link", "End_Effector"
ARM_GROUP, GRIPPER_GROUP, GRIPPER_JOINT = "arm", "gripper", "Gripper"
MOVE_GROUP_ACTION = "/move_action"
GRIPPER_OPEN, GRIPPER_CLOSE = 0.7854, -0.13
VEL_SCALE, ACCEL_SCALE = 1.0, 1.0
REPLAN_ATTEMPTS, REPLAN_DELAY = 5, 0.1
EE_POS_TOL, EE_ORI_TOL = 0.05, 0.5
MAX_RETRIES, SETTLE_TIME = 10, 0.5

ROS_PARAMS = {
    "base_frame": BASE_FRAME, "eef_link": EEF_LINK, "arm_group": ARM_GROUP,
    "gripper_group": GRIPPER_GROUP, "gripper_joint": GRIPPER_JOINT,
    "move_group_action": MOVE_GROUP_ACTION,
    "gripper_open": GRIPPER_OPEN, "gripper_close": GRIPPER_CLOSE,
}

COLOR_PICK_PARAMS = {
    "image_topic": "/camera/image_raw",
    "camera_info_topic": "/camera/camera_info",
    "optical_frame": "camera_optical_frame",
    "table_z_in_base": 0.0,
    "approach_z_offset": 0.05,
    "hsv_lower": [40, 40, 40],
    "hsv_upper": [90, 255, 255],
    "min_area_px": 400,
    "single_shot": True,
    "enable_gripper_close": True,
    "enable_place": False,
    "place_x": 0.0,
    "place_y": 0.0,
    "place_z": 0.12,
    "place_approach_z_offset": 0.05,
    "pos_tol": 0.02,
    "position_only": True,
}

Quat = Tuple[float, float, float, float]
Pose7 = Tuple[float, float, float, float, float, float, float]


class EefVerifyResult(NamedTuple):
    pos_err: float
    ori_err: Optional[float]
    ok: bool
    failure: Optional[RuntimeError]


@dataclass
class MovePoseConfig:
    x: float
    y: float
    z: float
    quat: Optional[Quat]
    position_only: bool
    pos_tol: float
    ori_tol: float
    plan_only: bool
    ee_pos_tol: float
    ee_ori_tol: float
    max_retries: int
    settle_time: float

    @classmethod
    def from_args(cls, args, quat: Optional[Quat], position_only: bool) -> "MovePoseConfig":
        retries = 1 if args.no_retry or args.plan_only else args.max_retries
        return cls(
            args.x, args.y, args.z, quat, position_only,
            args.pos_tol, args.ori_tol, args.plan_only,
            args.ee_pos_tol, args.ee_ori_tol, retries, args.settle_time,
        )


def _move_group_goal(group_name: str, constraints: Constraints, plan_only: bool) -> MoveGroup.Goal:
    req = MotionPlanRequest(
        group_name=group_name, num_planning_attempts=10, allowed_planning_time=10.0,
        max_velocity_scaling_factor=VEL_SCALE, max_acceleration_scaling_factor=ACCEL_SCALE,
    )
    req.goal_constraints.append(constraints)
    options = PlanningOptions(
        plan_only=plan_only, look_around=False, replan=not plan_only,
        replan_attempts=REPLAN_ATTEMPTS, replan_delay=REPLAN_DELAY,
    )
    return MoveGroup.Goal(request=req, planning_options=options)


def pose_goal(pose, eef_link, group_name, pos_tol, ori_tol, position_only, plan_only):
    c = Constraints(name="goal")
    pc = PositionConstraint()
    pc.header, pc.link_name, pc.weight = pose.header, eef_link, 1.0
    pc.target_point_offset = Vector3()
    sphere = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[float(pos_tol)])
    region = Pose()
    region.position, region.orientation.w = pose.pose.position, 1.0
    pc.constraint_region.primitives.append(sphere)
    pc.constraint_region.primitive_poses.append(region)
    c.position_constraints.append(pc)
    if not position_only:
        oc = OrientationConstraint()
        oc.header, oc.link_name = pose.header, eef_link
        oc.orientation = pose.pose.orientation
        tol = float(ori_tol)
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = tol
        oc.absolute_z_axis_tolerance = tol
        oc.weight, oc.parameterization = 1.0, OrientationConstraint.XYZ_EULER_ANGLES
        c.orientation_constraints.append(oc)
    return _move_group_goal(group_name, c, plan_only)


def gripper_goal(joint_name, value, group_name, plan_only):
    c = Constraints(name="gripper")
    c.joint_constraints.append(JointConstraint(
        joint_name=joint_name, position=float(value),
        tolerance_above=0.01, tolerance_below=0.01, weight=1.0,
    ))
    return _move_group_goal(group_name, c, plan_only)


def verify_eef(target_xyz, target_q, current: Pose7, position_only, ee_pos_tol, ee_ori_tol):
    pos_err = math.dist(target_xyz, current[:3])
    pos_ok = pos_err <= ee_pos_tol
    ori_err, ori_ok = None, True
    if not position_only:
        ori_err = 2.0 * math.acos(min(1.0, abs(sum(a * b for a, b in zip(target_q, current[3:])))))
        ori_ok = ori_err <= ee_ori_tol
    if pos_ok and ori_ok:
        return EefVerifyResult(pos_err, ori_err, True, None)
    parts = [f"pos {pos_err:.4f} > {ee_pos_tol:.4f}"] if not pos_ok else []
    if not ori_ok:
        parts.append(f"ori {ori_err:.4f} > {ee_ori_tol:.4f}")
    return EefVerifyResult(pos_err, ori_err, False, RuntimeError(", ".join(parts)))


def pixel_to_base_point(u, v, k, tf_buffer, base_frame, optical_frame, table_z):
    fx, fy, cx, cy = k[0], k[4], k[2], k[5]
    x = (u - cx) / fx
    y = (v - cy) / fy
    norm = math.sqrt(x * x + y * y + 1.0)
    d_opt = (x / norm, y / norm, 1.0 / norm)

    tfm = tf_buffer.lookup_transform(
        base_frame, optical_frame, rclpy.time.Time(), timeout=Duration(seconds=0.5),
    )
    tr = tfm.transform.translation
    q = tfm.transform.rotation
    rot = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
    origin = (tr.x, tr.y, tr.z)
    direction = tuple(
        rot[i][0] * d_opt[0] + rot[i][1] * d_opt[1] + rot[i][2] * d_opt[2]
        for i in range(3)
    )
    if abs(direction[2]) < 1e-6:
        return None
    t_hit = (table_z - origin[2]) / direction[2]
    if t_hit < 0:
        return None
    return tuple(origin[i] + t_hit * direction[i] for i in range(3))


class MoveEef(Node):
    def __init__(self):
        super().__init__("move_eef")
        for name, default in ROS_PARAMS.items():
            self.declare_parameter(name, default)
            setattr(self, name, self.get_parameter(name).value)
        self._client = ActionClient(self, MoveGroup, self.move_group_action)
        self._tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self._tf_buffer, self)
        self.get_logger().info(
            f"move_eef: frame={self.base_frame}, eef={self.eef_link}, "
            f"action={self.move_group_action}"
        )

    def _spin(self, future):
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def wait_for_server(self, timeout=30.0):
        if not self._client.wait_for_server(timeout_sec=timeout):
            raise RuntimeError(f"MoveGroup not available after {timeout}s")

    def send_goal(self, goal: MoveGroup.Goal):
        handle = self._spin(self._client.send_goal_async(goal))
        if not handle.accepted:
            raise RuntimeError("MoveGroup goal rejected")
        code = self._spin(handle.get_result_async()).result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"MoveGroup failed (error {code})")

    def lookup_eef(self) -> Pose7:
        try:
            tfm = self._tf_buffer.lookup_transform(
                self.base_frame, self.eef_link, rclpy.time.Time(),
                timeout=Duration(seconds=2.0),
            )
        except TransformException as exc:
            raise RuntimeError(f"TF {self.base_frame}->{self.eef_link}: {exc}") from exc
        t, r = tfm.transform.translation, tfm.transform.rotation
        return (t.x, t.y, t.z, r.x, r.y, r.z, r.w)

    def _settle(self, settle_time):
        deadline = time.monotonic() + settle_time
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def make_pose(self, x, y, z, quat: Optional[Quat]) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = float(x), float(y), float(z)
        if quat:
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = quat
        else:
            pose.pose.orientation.w = 1.0
        return pose

    def _arm_goal(self, pose, cfg: MovePoseConfig, plan_only):
        return pose_goal(pose, self.eef_link, self.arm_group,
                         cfg.pos_tol, cfg.ori_tol, cfg.position_only, plan_only)

    def _retry_once(self, cfg, pose, target_q, target_xyz, attempt) -> Optional[RuntimeError]:
        try:
            self.send_goal(self._arm_goal(pose, cfg, False))
        except RuntimeError as exc:
            self.get_logger().warn(f"{attempt}/{cfg.max_retries}: {exc}")
            time.sleep(cfg.settle_time)
            return exc
        self._settle(cfg.settle_time)
        try:
            check = verify_eef(target_xyz, target_q, self.lookup_eef(),
                               cfg.position_only, cfg.ee_pos_tol, cfg.ee_ori_tol)
        except RuntimeError as exc:
            self.get_logger().warn(f"{attempt}/{cfg.max_retries}: {exc}")
            time.sleep(cfg.settle_time)
            return exc
        msg = f"{attempt}/{cfg.max_retries}: pos err {check.pos_err:.4f} m"
        if check.ori_err is not None:
            msg += f", ori err {check.ori_err:.4f} rad"
        self.get_logger().info(msg)
        if check.ok:
            self.get_logger().info("Goal reached")
            return None
        time.sleep(cfg.settle_time)
        return check.failure

    def move_pose(self, cfg: MovePoseConfig):
        pose = self.make_pose(cfg.x, cfg.y, cfg.z, cfg.quat)
        target_q = (pose.pose.orientation.x, pose.pose.orientation.y,
                    pose.pose.orientation.z, pose.pose.orientation.w)
        target_xyz = (cfg.x, cfg.y, cfg.z)
        mode = "position-only" if cfg.position_only else "pose"
        self.get_logger().info(f"Target ({cfg.x:.3f},{cfg.y:.3f},{cfg.z:.3f}) [{mode}]")
        if cfg.plan_only:
            self.send_goal(self._arm_goal(pose, cfg, True))
            return
        last_err = None
        for attempt in range(1, cfg.max_retries + 1):
            last_err = self._retry_once(cfg, pose, target_q, target_xyz, attempt)
            if last_err is None:
                return
        raise RuntimeError(f"Failed after {cfg.max_retries} tries: {last_err}")

    def move_gripper(self, open_gripper, plan_only):
        val = self.gripper_open if open_gripper else self.gripper_close
        label = "open" if open_gripper else "close"
        self.get_logger().info(f"Gripper {label}: {val:.3f}")
        self.send_goal(gripper_goal(self.gripper_joint, val, self.gripper_group, plan_only))


class ColorPickNode(MoveEef):

    def __init__(self):
        super().__init__()
        for name, default in COLOR_PICK_PARAMS.items():
            self.declare_parameter(name, default)
        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.optical_frame = self.get_parameter("optical_frame").value
        self.table_z = self.get_parameter("table_z_in_base").value
        self.approach_z = self.get_parameter("approach_z_offset").value
        self.min_area = self.get_parameter("min_area_px").value
        self.single_shot = self.get_parameter("single_shot").value
        self.enable_gripper_close = self.get_parameter("enable_gripper_close").value
        self.enable_place = self.get_parameter("enable_place").value
        self.place_x = self.get_parameter("place_x").value
        self.place_y = self.get_parameter("place_y").value
        self.place_z = self.get_parameter("place_z").value
        self.place_approach_z = self.get_parameter("place_approach_z_offset").value
        self.pick_pos_tol = self.get_parameter("pos_tol").value
        self.pick_position_only = self.get_parameter("position_only").value
        lo = list(self.get_parameter("hsv_lower").value)
        hi = list(self.get_parameter("hsv_upper").value)
        self.hsv_lo = np.array(lo, dtype=np.uint8)
        self.hsv_hi = np.array(hi, dtype=np.uint8)

        self._done = False
        self._busy = False
        self._bridge = CvBridge()
        self._k = None
        self._phase = "pick"

        self.pub_dbg = self.create_publisher(PoseStamped, "debug/grasp_pose", 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._on_camera_info, 1)
        self.create_subscription(Image, self.image_topic, self._on_image, 1)
        self.get_logger().info(
            f"color_pick: image={self.image_topic}, optical_frame={self.optical_frame}, "
            f"base_frame={self.base_frame}, table_z={self.table_z}"
        )

    def _on_camera_info(self, msg: CameraInfo):
        self._k = list(msg.k)

    def _on_image(self, msg: Image):
        if self._busy or (self.single_shot and self._done):
            return
        if self._k is None or not self._client.server_is_ready():
            return

        bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lo, self.hsv_hi)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return
        contour = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(contour) < self.min_area:
            return
        moments = cv2.moments(contour)
        if moments["m00"] < 1e-6:
            return
        u = moments["m10"] / moments["m00"]
        v = moments["m01"] / moments["m00"]

        try:
            hit = pixel_to_base_point(
                u, v, self._k, self._tf_buffer,
                self.base_frame, self.optical_frame, self.table_z,
            )
        except TransformException as exc:
            self.get_logger().warn(f"TF {self.base_frame}->{self.optical_frame}: {exc}")
            return
        if hit is None:
            return

        z = hit[2] + self.approach_z
        pose = self.make_pose(hit[0], hit[1], z, None)
        self.pub_dbg.publish(pose)
        self.get_logger().info(
            f"Pick target in {self.base_frame}: ({hit[0]:.3f}, {hit[1]:.3f}, {z:.3f}) "
            f"from pixel ({u:.0f}, {v:.0f})"
        )
        goal = pose_goal(
            pose, self.eef_link, self.arm_group, self.pick_pos_tol, 0.4,
            self.pick_position_only, False,
        )
        self._busy = True
        self._phase = "pick"
        self._client.send_goal_async(goal).add_done_callback(self._on_arm_goal_sent)

    def _on_arm_goal_sent(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn("Pick move goal rejected")
            self._busy = False
            return
        handle.get_result_async().add_done_callback(self._on_arm_result)

    def _on_arm_result(self, future):
        res = future.result().result
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().warn(f"Arm move failed (error {res.error_code.val})")
            self._busy = False
            return
        if self._phase == "place":
            self.get_logger().info("Place reached, opening gripper")
            self._send_gripper(True)
            return
        if self.enable_gripper_close:
            self.get_logger().info("Pick reached, closing gripper")
            self._send_gripper(False)
        elif self.enable_place:
            self._start_place()
        else:
            self._finish()

    def _send_gripper(self, open_gripper: bool):
        val = self.gripper_open if open_gripper else self.gripper_close
        label = "open" if open_gripper else "close"
        self.get_logger().info(f"Gripper {label}: {val:.3f}")
        goal = gripper_goal(self.gripper_joint, val, self.gripper_group, False)
        self._client.send_goal_async(goal).add_done_callback(self._on_gripper_goal_sent)

    def _on_gripper_goal_sent(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn("Gripper goal rejected")
            self._busy = False
            return
        handle.get_result_async().add_done_callback(self._on_gripper_result)

    def _on_gripper_result(self, future):
        res = future.result().result
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().warn(f"Gripper move failed (error {res.error_code.val})")
            self._busy = False
            return
        if self._phase == "pick" and self.enable_place:
            self._start_place()
        else:
            self._finish()

    def _start_place(self):
        self._phase = "place"
        z = self.place_z + self.place_approach_z
        pose = self.make_pose(self.place_x, self.place_y, z, None)
        self.pub_dbg.publish(pose)
        self.get_logger().info(
            f"Place target in {self.base_frame}: ({self.place_x:.3f}, {self.place_y:.3f}, {z:.3f})"
        )
        goal = pose_goal(
            pose, self.eef_link, self.arm_group, self.pick_pos_tol, 0.4,
            self.pick_position_only, False,
        )
        self._client.send_goal_async(goal).add_done_callback(self._on_arm_goal_sent)

    def _finish(self):
        self._done = True
        self._busy = False
        self.get_logger().info("Color pick/place done")


def _default_params_file(color_pick=False):
    try:
        from ament_index_python.packages import get_package_share_directory
        name = "move_eef_color_pick.yaml" if color_pick else "move_eef.yaml"
        path = os.path.join(get_package_share_directory("so_arm_100_bringup"), "config", name)
        return path if os.path.isfile(path) else None
    except Exception:
        return None


def _cli_argv(raw_argv):
    argv = remove_ros_args(raw_argv)[1:]
    if argv and argv[0].endswith((".yaml", ".yml")):
        sys.exit(2)
    return argv


def parse_args(argv, color_pick=False):
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--color-pick", action="store_true",
                   help="Camera HSV pick")
    if not color_pick:
        p.add_argument("x", type=float, help="Target x [m]")
        p.add_argument("y", type=float, help="Target y [m]")
        p.add_argument("z", type=float, help="Target z [m]")
    p.add_argument("--qx", type=float, default=None)
    p.add_argument("--qy", type=float, default=None)
    p.add_argument("--qz", type=float, default=None)
    p.add_argument("--qw", type=float, default=None)
    p.add_argument("--gripper", choices=("none", "open", "close"), default="none")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--pos-tol", type=float, default=0.02)
    p.add_argument("--ori-tol", type=float, default=0.4)
    p.add_argument("--ee-pos-tol", type=float, default=EE_POS_TOL)
    p.add_argument("--ee-ori-tol", type=float, default=EE_ORI_TOL)
    p.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    p.add_argument("--settle-time", type=float, default=SETTLE_TIME)
    p.add_argument("--no-retry", action="store_true")
    return p.parse_args(argv)


def resolve_quat(args):
    vals = (args.qx, args.qy, args.qz, args.qw)
    if all(v is None for v in vals):
        return None, True
    if any(v is None for v in vals):
        raise ValueError("Provide all of --qx --qy --qz --qw or none for position-only")
    return vals, False


def run_color_pick():
    node = ColorPickNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


def run_move_cli(args):
    quat, position_only = resolve_quat(args)
    node = MoveEef()
    cfg = MovePoseConfig.from_args(args, quat, position_only)
    try:
        node.wait_for_server()
        node.move_pose(cfg)
        if args.gripper == "open":
            node.move_gripper(True, args.plan_only)
        elif args.gripper == "close":
            node.move_gripper(False, args.plan_only)
        node.get_logger().info("Done.")
    except Exception as exc:
        node.get_logger().error(str(exc))
        return 1
    finally:
        node.destroy_node()
    return 0


def main(argv=None):
    raw_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    cli_argv = _cli_argv(list(raw_argv))
    color_pick = "--color-pick" in cli_argv

    init_argv = list(raw_argv)
    params_file = _default_params_file(color_pick) or _default_params_file(False)
    if params_file and "--params-file" not in init_argv:
        init_argv += ["--ros-args", "--params-file", params_file]

    rclpy.init(args=init_argv)

    if color_pick:
        try:
            run_color_pick()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
        return 0

    if not cli_argv:
        print("move_eef -- X Y Z   or   move_eef -- --color-pick", file=sys.stderr)
        rclpy.shutdown()
        return 2

    try:
        args = parse_args(cli_argv, color_pick=False)
    except SystemExit as exc:
        rclpy.shutdown()
        return int(exc.code) if exc.code is not None else 2

    try:
        return run_move_cli(args)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
