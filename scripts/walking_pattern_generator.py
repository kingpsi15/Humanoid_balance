#!/usr/bin/env python3
"""
Walking Pattern Generator for Humanoid Robot
Extends the balance controller with walking capabilities
"""

import numpy as np
import rospy
from std_msgs.msg import Float64, Float64MultiArray
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates  # ✅ Added

def leg_inverse_kinematics_3d(dx, dy, dz, l1, l2):
    """3-DOF analytical inverse kinematics for leg (hip pitch, hip roll, knee)"""
    L = np.sqrt(dx**2 + dz**2)
    cos_k = (l1**2 + l2**2 - L**2) / (2 * l1 * l2)
    cos_k = np.clip(cos_k, -1.0, 1.0)
    q_knee = np.pi - np.arccos(cos_k)
    alpha = np.arctan2(dz, dx)
    beta  = np.arccos(np.clip((l1**2 + L**2 - l2**2) / (2 * l1 * L), -1.0, 1.0))
    q_pitch = alpha - beta
    q_roll = np.arctan2(dy, L)
    return {'hip_pitch': q_pitch, 'hip_roll': q_roll, 'knee': q_knee}

class WalkingPatternGenerator:
    def __init__(self):
        # Read controller sample period from parameter server
        self.T = rospy.get_param('~T', 0.005)

        # Walking parameters (physics tweaks applied)
        self.step_length = 0.1
        self.step_width = 0.16
        self.step_height = 0.10   # increased for larger knee lift
        self.step_time = 0.8
        self.double_support_ratio = 0.2

        # Walking state
        self.walking_enabled = False
        self.model_ready = False
        self.startup_delay_passed = False
        self.current_time = 0.0
        self.step_count = 0
        self.left_foot_stance = True

        # Target velocities
        self.target_vel_x = 0.0
        self.target_vel_y = 0.0
        self.target_vel_yaw = 0.0

        # Foot positions
        self.left_foot_pos = np.array([-self.step_width/2, 0.0, 0.0])
        self.right_foot_pos = np.array([ self.step_width/2, 0.0, 0.0])

        # Publishers/subscribers
        self.setup_joint_publishers()
        self.trajectory_pub = rospy.Publisher('/balance_controller/trajectory', Float64MultiArray, queue_size=1)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_state_callback)
        rospy.Timer(rospy.Duration(2.0), self.enable_startup_delay, oneshot=True)
        self.walk_timer = rospy.Timer(rospy.Duration(self.T), lambda evt: self.generate_walking_trajectory())

        rospy.loginfo("Walking Pattern Generator initialized and running")

    def enable_startup_delay(self, event):
        self.startup_delay_passed = True
        rospy.loginfo("Startup delay passed. Ready to walk if model is ready.")

    def model_state_callback(self, msg):
        if not self.model_ready:
            for nm in msg.name:
                if nm == 'humanoid_robot':
                    self.model_ready = True
                    rospy.loginfo("Model state for humanoid_robot received as '%s'" % nm)
                    break

    def setup_joint_publishers(self):
        self.hip_pitch_left_effort_pub = rospy.Publisher(
            '/humanoid/hip_pitch_left_effort_controller/command', Float64, queue_size=1)
        self.hip_roll_left_effort_pub  = rospy.Publisher(
            '/humanoid/hip_roll_left_effort_controller/command', Float64, queue_size=1)
        self.knee_left_effort_pub      = rospy.Publisher(
            '/humanoid/knee_left_effort_controller/command', Float64, queue_size=1)
        self.hip_pitch_right_effort_pub= rospy.Publisher(
            '/humanoid/hip_pitch_right_effort_controller/command', Float64, queue_size=1)
        self.hip_roll_right_effort_pub = rospy.Publisher(
            '/humanoid/hip_roll_right_effort_controller/command', Float64, queue_size=1)
        self.knee_right_effort_pub     = rospy.Publisher(
            '/humanoid/knee_right_effort_controller/command', Float64, queue_size=1)

    def cmd_vel_callback(self, msg):
        self.target_vel_x   = msg.linear.x
        self.target_vel_y   = msg.linear.y
        self.target_vel_yaw = msg.angular.z
        engaged = abs(msg.linear.x) > 0.01 or abs(msg.linear.y) > 0.01 or abs(msg.angular.z) > 0.01
        if engaged and not self.walking_enabled:
            rospy.loginfo("Walking enabled")
            self.walking_enabled = True
            self.current_time = 0.0
        elif not engaged and self.walking_enabled:
            rospy.loginfo("Walking disabled")
            self.walking_enabled = False

    def generate_walking_trajectory(self):
        if not (self.walking_enabled and self.model_ready and self.startup_delay_passed):
            return

        self.current_time += self.T
        phase = (self.current_time % self.step_time) / self.step_time
        if phase < 0.01 and self.current_time > 0.01:
            self.left_foot_stance = not self.left_foot_stance
            self.step_count += 1
            rospy.loginfo(f"Step {self.step_count}, Left stance: {self.left_foot_stance}")

        com_traj = self.generate_com_trajectory(phase)
        traj_msg = Float64MultiArray(data=[
            com_traj['x_ref'], com_traj['y_ref'],
            com_traj['x_dot_ref'], com_traj['y_dot_ref'],
            com_traj['x_ddot_ref'], com_traj['y_ddot_ref']])
        self.trajectory_pub.publish(traj_msg)

        left_ft, right_ft = self.generate_foot_trajectories(phase)
        self.compute_joint_commands(left_ft, right_ft, com_traj)

    def generate_com_trajectory(self, phase):
        com_x = self.target_vel_x * self.current_time
        shift = 0.05 * np.sin(np.pi * phase)  # increased lateral shift
        com_y = -shift if self.left_foot_stance else shift
        com_z = 0.814 - 0.01 * abs(np.sin(2*np.pi*phase))
        return {
            'x_ref': com_x, 'y_ref': com_y, 'z_ref': com_z,
            'x_dot_ref': self.target_vel_x, 'y_dot_ref': 0.0, 'z_dot_ref': 0.0,
            'x_ddot_ref': 0.0, 'y_ddot_ref': 0.0, 'z_ddot_ref': 0.0}

    def generate_foot_trajectories(self, phase):
        ds = self.double_support_ratio
        if phase < ds or phase > (1-ds):
            lz = rz = 0.0
        else:
            sp = (phase - ds)/(1-2*ds)
            h = self.step_height * np.sin(np.pi * sp)
            if self.left_foot_stance:
                lz, rz = 0.0, h
                self.right_foot_pos[0] = self.left_foot_pos[0] + self.step_length * sp
            else:
                lz, rz = h, 0.0
                self.left_foot_pos[0]  = self.right_foot_pos[0] + self.step_length * sp
        return (
            {'x':self.left_foot_pos[0],'y':self.left_foot_pos[1],'z':lz},
            {'x':self.right_foot_pos[0],'y':self.right_foot_pos[1],'z':rz})

    def compute_joint_commands(self, lft, rft, com):
        tl, sl = 0.2, 0.2
        lj = leg_inverse_kinematics_3d(
            com['x_ref']-lft['x'], com['y_ref']-lft['y'], com['z_ref']-lft['z'], tl, sl)
        rj = leg_inverse_kinematics_3d(
            com['x_ref']-rft['x'], com['y_ref']-rft['y'], com['z_ref']-rft['z'], tl, sl)

        # apply forward hip-pitch bias to prevent backward lean
        bias = 0.1  # radians
        self.hip_pitch_left_effort_pub.publish(Float64(lj['hip_pitch'] + bias))
        self.hip_roll_left_effort_pub.publish(Float64(lj['hip_roll']))
        self.knee_left_effort_pub.publish(Float64(lj['knee']))
        self.hip_pitch_right_effort_pub.publish(Float64(rj['hip_pitch'] + bias))
        self.hip_roll_right_effort_pub.publish(Float64(rj['hip_roll']))
        self.knee_right_effort_pub.publish(Float64(rj['knee']))

if __name__ == '__main__':
    rospy.init_node('walking_pattern_generator')
    wpg = WalkingPatternGenerator()
    rospy.spin()
