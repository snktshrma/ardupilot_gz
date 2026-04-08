#!/usr/bin/python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import Pose, PoseStamped, Vector3
from sensor_msgs.msg import CameraInfo, Image
import tf2_ros
import tf_transformations
from tf2_ros import TransformException
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningOptions,
    PositionConstraint,
)
from moveit_msgs.action import MoveGroup
from shape_msgs.msg import SolidPrimitive


def pose_stamped_to_move_group_goal(
    pose,
    eef_link,
    group_name,
    position_tolerance_m=0.02,
    orient_tolerance_rad=0.4,
    plan_only=False,
):
    req = MotionPlanRequest()
    req.group_name = group_name
    req.num_planning_attempts = 5
    req.allowed_planning_time = 5.0
    req.max_velocity_scaling_factor = 0.5
    req.max_acceleration_scaling_factor = 0.5

    pc = PositionConstraint()
    pc.header = pose.header
    pc.link_name = eef_link
    pc.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
    pc.weight = 1.0

    sp = SolidPrimitive()
    sp.type = SolidPrimitive.SPHERE
    sp.dimensions = [float(position_tolerance_m)]

    sp_pose = Pose()
    sp_pose.position = pose.pose.position
    sp_pose.orientation.w = 1.0

    pc.constraint_region.primitives.append(sp)
    pc.constraint_region.primitive_poses.append(sp_pose)

    oc = OrientationConstraint()
    oc.header = pose.header
    oc.link_name = eef_link
    oc.orientation = pose.pose.orientation
    oc.absolute_x_axis_tolerance = float(orient_tolerance_rad)
    oc.absolute_y_axis_tolerance = float(orient_tolerance_rad)
    oc.absolute_z_axis_tolerance = float(orient_tolerance_rad)
    oc.weight = 1.0
    oc.parameterization = OrientationConstraint.XYZ_EULER_ANGLES

    c = Constraints()
    c.name = "goal"
    c.position_constraints.append(pc)
    c.orientation_constraints.append(oc)
    req.goal_constraints.append(c)

    opts = PlanningOptions()
    opts.plan_only = plan_only
    opts.look_around = False
    opts.replan = False

    g = MoveGroup.Goal()
    g.request = req
    g.planning_options = opts
    return g


def joint_value_to_move_group_goal(joint_name, joint_value, group_name, plan_only=False):
    req = MotionPlanRequest()
    req.group_name = group_name
    req.num_planning_attempts = 3
    req.allowed_planning_time = 3.0
    req.max_velocity_scaling_factor = 0.5
    req.max_acceleration_scaling_factor = 0.5

    jc = JointConstraint()
    jc.joint_name = joint_name
    jc.position = float(joint_value)
    jc.tolerance_above = 0.01
    jc.tolerance_below = 0.01
    jc.weight = 1.0

    c = Constraints()
    c.name = "joint_goal"
    c.joint_constraints.append(jc)
    req.goal_constraints.append(c)

    opts = PlanningOptions()
    opts.plan_only = plan_only
    opts.look_around = False
    opts.replan = False

    g = MoveGroup.Goal()
    g.request = req
    g.planning_options = opts
    return g


class ColorPickNode(Node):
    def __init__(self):
        super().__init__("color_pick")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("optical_frame", "camera_optical_frame")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("eef_link", "grip_left_link")
        self.declare_parameter("planning_group", "arm")
        self.declare_parameter("gripper_group", "arm")
        self.declare_parameter("gripper_joint", "grip_left")
        self.declare_parameter("gripper_close_value", 0.0)
        self.declare_parameter("enable_gripper_close", True)
        self.declare_parameter("table_z_in_base", 0.0)
        self.declare_parameter("approach_z_offset", 0.05)
        self.declare_parameter("hsv_lower", [40, 40, 40])
        self.declare_parameter("hsv_upper", [90, 255, 255])
        self.declare_parameter("min_area_px", 400)
        self.declare_parameter("move_group_action", "/move_action")
        self.declare_parameter("single_shot", True)

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.optical_frame = self.get_parameter("optical_frame").get_parameter_value().string_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.eef_link = self.get_parameter("eef_link").get_parameter_value().string_value
        self.planning_group = self.get_parameter("planning_group").get_parameter_value().string_value
        self.gripper_group = self.get_parameter("gripper_group").get_parameter_value().string_value
        self.gripper_joint = self.get_parameter("gripper_joint").get_parameter_value().string_value
        self.gripper_close_value = self.get_parameter("gripper_close_value").get_parameter_value().double_value
        self.enable_gripper_close = self.get_parameter("enable_gripper_close").get_parameter_value().bool_value
        self.table_z = self.get_parameter("table_z_in_base").get_parameter_value().double_value
        self.approach_z = self.get_parameter("approach_z_offset").get_parameter_value().double_value
        lo = list(self.get_parameter("hsv_lower").get_parameter_value().integer_array_value)
        hi = list(self.get_parameter("hsv_upper").get_parameter_value().integer_array_value)
        self.hsv_lo = np.array(lo, dtype=np.uint8)
        self.hsv_hi = np.array(hi, dtype=np.uint8)
        self.min_area = self.get_parameter("min_area_px").get_parameter_value().integer_value
        action_name = self.get_parameter("move_group_action").get_parameter_value().string_value
        self.single_shot = self.get_parameter("single_shot").get_parameter_value().bool_value

        self._done = False
        self._busy = False
        self._bridge = CvBridge()
        self._K = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub_dbg = self.create_publisher(PoseStamped, "debug/grasp_pose", 10)

        self._action_client = ActionClient(self, MoveGroup, action_name)

        self.create_subscription(CameraInfo, self.camera_info_topic, self._on_ci, 1)
        self.create_subscription(Image, self.image_topic, self._on_image, 1)

    def _on_ci(self, msg):
        self._K = np.array(msg.k).reshape(3, 3)

    def _on_image(self, msg):
        if self._busy or (self.single_shot and self._done):
            return
        if self._K is None or not self._action_client.server_is_ready():
            return

        bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lo, self.hsv_hi)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < self.min_area:
            return
        M = cv2.moments(c)
        if M["m00"] < 1e-6:
            return
        u = M["m10"] / M["m00"]
        v = M["m01"] / M["m00"]

        fx, fy, cx, cy = self._K[0, 0], self._K[1, 1], self._K[0, 2], self._K[1, 2]
        x = (u - cx) / fx
        y = (v - cy) / fy
        d_opt = np.array([x, y, 1.0])
        d_opt = d_opt / np.linalg.norm(d_opt)

        try:
            tfm = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.optical_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
        except TransformException:
            return

        tr = tfm.transform.translation
        q = tfm.transform.rotation
        R = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        O = np.array([tr.x, tr.y, tr.z])
        D = R @ d_opt.reshape(3)
        if abs(D[2]) < 1e-6:
            return
        t_hit = (self.table_z - O[2]) / D[2]
        if t_hit < 0:
            return
        P = O + t_hit * D

        ps = PoseStamped()
        ps.header.frame_id = self.base_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(P[0])
        ps.pose.position.y = float(P[1])
        ps.pose.position.z = float(P[2]) + self.approach_z
        ps.pose.orientation.w = 1.0

        self.pub_dbg.publish(ps)

        goal = pose_stamped_to_move_group_goal(
            ps, self.eef_link, self.planning_group, plan_only=False
        )
        self._busy = True
        self._action_client.send_goal_async(goal).add_done_callback(self._on_arm_goal_sent)

    def _on_arm_goal_sent(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._busy = False
            return
        goal_handle.get_result_async().add_done_callback(self._on_arm_result)

    def _on_arm_result(self, future):
        res = future.result().result
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            self._busy = False
            return
        if self.enable_gripper_close:
            g = joint_value_to_move_group_goal(
                self.gripper_joint,
                self.gripper_close_value,
                self.gripper_group,
                plan_only=False,
            )
            self._action_client.send_goal_async(g).add_done_callback(self._on_gripper_goal_sent)
        else:
            self._done = True
            self._busy = False

    def _on_gripper_goal_sent(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._busy = False
            return
        goal_handle.get_result_async().add_done_callback(self._on_gripper_result)

    def _on_gripper_result(self, future):
        self._busy = False
        res = future.result().result
        if res.error_code.val == MoveItErrorCodes.SUCCESS:
            self._done = True


def main(args=None):
    rclpy.init(args=args)
    node = ColorPickNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
