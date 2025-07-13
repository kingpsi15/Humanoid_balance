#!/usr/bin/env python3
"""
Gazebo Integration for Humanoid Balance Controller
Integrates the perfect balance controller with Gazebo simulation
"""

import rospy
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray, Float64
from gazebo_msgs.msg import ModelStates
from walking_pattern_generator import WalkingPatternGenerator
from scipy.linalg import solve_discrete_are

class GazeboHumanoidController:
    def __init__(self):
        rospy.init_node('humanoid_balance_controller', anonymous=True)

        # Controller parameters
        self.T = rospy.get_param('~T', 0.005)
        self.zc = rospy.get_param('~zc', 0.814)
        self.g  = rospy.get_param('~g', 9.81)

        # System matrices
        self.A = np.array([
            [1,  self.T, self.T**2/2],
            [0,  1,      self.T],
            [0,  0,      1]
        ])
        self.B = np.array([[self.T**3/6], [self.T**2/2], [self.T]])
        self.C_zmp = np.array([[1, 0, -self.zc/self.g]])
        self.C_pos = np.array([[1, 0, 0]])

        # LQR controller
        self.setup_lqr_controller()

        # State
        self.x_state = np.zeros(3)
        self.y_state = np.zeros(3)
        self.x_int   = 0.0
        self.y_int   = 0.0

        # Sensors & refs
        self.com_pos  = None
        self.com_vel  = None
        self.com_acc  = None
        self.ref_traj = {
            'x_ref':0,'y_ref':0,
            'x_dot_ref':0,'y_dot_ref':0,
            'x_ddot_ref':0,'y_ddot_ref':0
        }

        # ZMP limits
        self.foot_length   = 0.24
        self.foot_width    = 0.12
        self.safety_margin = 0.8

        # ROS I/O
        self.setup_ros_interface()

        # Walking pattern
        self.walk_gen = WalkingPatternGenerator()

        # Model readiness
        self.model_ready = False
        rospy.Timer(rospy.Duration(0.5), self.check_model_ready, oneshot=True)

        rospy.loginfo("Humanoid Balance Controller initialized")

    def check_model_ready(self, event):
        try:
            ms = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=5.0)
            for nm in ms.name:
                if nm == 'humanoid_robot':
                    # ✅ CHANGED: no early return—always start control timer once the model appears
                    self.model_ready = True
                    rospy.sleep(5.0)  # allow Gazebo to settle
                    self.control_timer = rospy.Timer(
                        rospy.Duration(self.T),
                        self.control_callback
                    )
                    rospy.loginfo("Model found; control loop started")
                    break
        except rospy.ROSException:
            rospy.logwarn_throttle(5.0,
                "Waiting for humanoid_robot in /gazebo/model_states…")

    def setup_lqr_controller(self):
        A_aug = np.zeros((4,4))
        A_aug[:3,:3] = self.A
        A_aug[3,:3]  = self.C_pos.flatten()
        A_aug[3,3]   = 1
        B_aug = np.zeros((4,1))
        B_aug[:3,0]  = self.B.flatten()
        Q = np.diag([10000,1000,1,1000])
        R = np.array([[0.01]])
        P = solve_discrete_are(A_aug, B_aug, Q, R)
        self.K = np.linalg.inv(R + B_aug.T@P@B_aug) @ B_aug.T @ P @ A_aug
        eigs = np.linalg.eigvals(A_aug - B_aug@self.K)
        self.use_lqr = np.all(np.abs(eigs)<1)
        if self.use_lqr:
            rospy.loginfo("LQR initialized; system stable")
        else:
            rospy.logwarn("LQR unstable; using PD fallback")

    def setup_ros_interface(self):
        sub = rospy.Subscriber
        pub = rospy.Publisher
        sub('/gazebo/model_states', ModelStates, self.model_state_callback)
        sub('/imu/data',            Imu,        self.imu_callback)
        sub('/joint_states',        JointState, self.joint_state_callback)
        sub('/balance_controller/reference', PoseStamped,       self.reference_callback)
        sub('/balance_controller/trajectory', Float64MultiArray, self.trajectory_callback)
        self.ankle_roll_left_pub   = pub('/humanoid/ankle_roll_left_effort_controller/command',  Float64, queue_size=1)
        self.ankle_roll_right_pub  = pub('/humanoid/ankle_roll_right_effort_controller/command', Float64, queue_size=1)
        self.ankle_pitch_left_pub  = pub('/humanoid/ankle_pitch_left_effort_controller/command',Float64, queue_size=1)
        self.ankle_pitch_right_pub = pub('/humanoid/ankle_pitch_right_effort_controller/command',Float64, queue_size=1)
        self.zmp_pub               = pub('/balance_controller/zmp',   PoseStamped,      queue_size=1)
        self.debug_pub             = pub('/balance_controller/debug', Float64MultiArray,queue_size=1)
        self.error_pub             = pub('/balance_controller/tracking_error',PoseStamped,   queue_size=1)

    def model_state_callback(self, msg):
        if not self.model_ready:
            for idx, nm in enumerate(msg.name):
                if nm == 'humanoid_robot':
                    # ✅ CHANGED: store COM position & velocity for control loop
                    pose  = msg.pose[idx]
                    twist = msg.twist[idx]
                    self.com_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
                    self.com_vel = np.array([twist.linear.x, twist.linear.y, twist.linear.z])
                    rospy.loginfo("Model state for humanoid_robot received as '%s'" % nm)
                    break

    def imu_callback(self, msg):
        a = msg.linear_acceleration
        self.com_acc = np.array([a.x, a.y, a.z + self.g])

    def joint_state_callback(self, msg):
        pass  # unused

    def reference_callback(self, msg):
        self.ref_traj['x_ref'] = msg.pose.position.x
        self.ref_traj['y_ref'] = msg.pose.position.y

    def trajectory_callback(self, msg):
        d = msg.data
        if len(d)>=6:
            self.ref_traj.update({
                'x_ref':     d[0],
                'y_ref':     d[1],
                'x_dot_ref': d[2],
                'y_dot_ref': d[3],
                'x_ddot_ref':d[4],
                'y_ddot_ref':d[5]
            })

    def compute_zmp(self, state):
        return float((self.C_zmp @ state)[0])

    def enforce_zmp_constraints(self, u, state, length):
        zmp_pred = self.compute_zmp(self.A @ state + self.B.flatten() * u)
        lim = length/2 * self.safety_margin + state[0]
        if abs(zmp_pred) > abs(lim) and zmp_pred != 0:
            u *= min(abs(lim/zmp_pred), 0.8)
        return u

    def control_callback(self, event):
        if self.com_pos is None or self.com_vel is None or self.com_acc is None:
            return
        self.x_state[:] = [self.com_pos[0], self.com_vel[0], self.com_acc[0]]
        self.y_state[:] = [self.com_pos[1], self.com_vel[1], self.com_acc[1]]
        r = self.ref_traj
        # feedforward
        u_ff_x = (r['x_ddot_ref'] - self.x_state[2]) / self.T
        u_ff_y = (r['y_ddot_ref'] - self.y_state[2]) / self.T
        # feedback
        if self.use_lqr:
            self.x_int += (r['x_ref'] - self.x_state[0])*self.T
            self.y_int += (r['y_ref'] - self.y_state[1])*self.T
            xa = np.array([*self.x_state, self.x_int])
            ya = np.array([*self.y_state, self.y_int])
            xaug = np.array([r['x_ref'],r['x_dot_ref'],r['x_ddot_ref'],0])
            yaug = np.array([r['y_ref'],r['y_dot_ref'],r['y_ddot_ref'],0])
            u_fb_x = float(self.K@(xaug-xa))
            u_fb_y = float(self.K@(yaug-ya))
        else:
            kp, kd = 1000.0, 100.0
            u_fb_x = kp*(r['x_ref']-self.x_state[0]) + kd*(r['x_dot_ref']-self.x_state[1])
            u_fb_y = kp*(r['y_ref']-self.y_state[0]) + kd*(r['y_dot_ref']-self.y_state[1])
        # total
        ux = np.clip(u_ff_x+u_fb_x, -10, 10)
        uy = np.clip(u_ff_y+u_fb_y, -10, 10)
        # constrain
        ux = self.enforce_zmp_constraints(ux, self.x_state, self.foot_length)
        uy = self.enforce_zmp_constraints(uy, self.y_state, self.foot_width)
        # publish
        pitch_torque = ux*10.0
        roll_torque  = uy*10.0
        self.ankle_pitch_left_pub.publish(Float64(pitch_torque))
        self.ankle_pitch_right_pub.publish(Float64(pitch_torque))
        self.ankle_roll_left_pub.publish(Float64(roll_torque))
        self.ankle_roll_right_pub.publish(Float64(roll_torque))
        # debug
        zmp_x = self.compute_zmp(self.x_state)
        zmp_y = self.compute_zmp(self.y_state)
        dm = Float64MultiArray(data=[ux,uy,*self.x_state,*self.y_state,zmp_x,zmp_y])
        self.debug_pub.publish(dm)
        e = PoseStamped(); e.header.stamp=rospy.Time.now()
        e.pose.position.x = r['x_ref']-self.x_state[0]
        e.pose.position.y = r['y_ref']-self.y_state[1]
        self.error_pub.publish(e)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        c = GazeboHumanoidController()
        c.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutdown")
