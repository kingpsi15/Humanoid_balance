#!/usr/bin/env python3
"""
Walking Pattern Generator for Humanoid Robot
Extends the balance controller with walking capabilities
"""

import numpy as np
import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist

class WalkingPatternGenerator:
    def __init__(self, balance_controller):
        self.balance_controller = balance_controller
        
        # Walking parameters
        self.step_length = 0.1    # meters
        self.step_width = 0.16    # meters (distance between feet)
        self.step_height = 0.05   # meters
        self.step_time = 0.8      # seconds per step
        self.double_support_ratio = 0.2  # fraction of step time in double support
        
        # Current walking state
        self.walking_enabled = False
        self.current_time = 0.0
        self.step_count = 0
        self.left_foot_stance = True  # True if left foot is stance leg
        
        # Target velocities
        self.target_vel_x = 0.0
        self.target_vel_y = 0.0
        self.target_vel_yaw = 0.0
        
        # Foot positions
        self.left_foot_pos = np.array([-self.step_width/2, 0.0, 0.0])
        self.right_foot_pos = np.array([self.step_width/2, 0.0, 0.0])
        
        # Additional joint publishers for walking
        self.setup_joint_publishers()
        
        # Subscribe to walking commands
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        
    def setup_joint_publishers(self):
        """Setup publishers for all joints needed for walking"""
        self.hip_pitch_left_pub = rospy.Publisher('/humanoid/hip_pitch_left_position_controller/command', 
                                                Float64, queue_size=1)
        self.hip_pitch_right_pub = rospy.Publisher('/humanoid/hip_pitch_right_position_controller/command', 
                                                 Float64, queue_size=1)
        self.hip_roll_left_pub = rospy.Publisher('/humanoid/hip_roll_left_position_controller/command', 
                                               Float64, queue_size=1)
        self.hip_roll_right_pub = rospy.Publisher('/humanoid/hip_roll_right_position_controller/command', 
                                                Float64, queue_size=1)
        self.knee_left_pub = rospy.Publisher('/humanoid/knee_left_position_controller/command', 
                                           Float64, queue_size=1)
        self.knee_right_pub = rospy.Publisher('/humanoid/knee_right_position_controller/command', 
                                            Float64, queue_size=1)
    
    def cmd_vel_callback(self, msg):
        """Handle walking velocity commands"""
        self.target_vel_x = msg.linear.x
        self.target_vel_y = msg.linear.y
        self.target_vel_yaw = msg.angular.z
        
        # Enable walking if any velocity is commanded
        if abs(msg.linear.x) > 0.01 or abs(msg.linear.y) > 0.01 or abs(msg.angular.z) > 0.01:
            if not self.walking_enabled:
                rospy.loginfo("Walking enabled")
                self.walking_enabled = True
                self.current_time = 0.0
        else:
            if self.walking_enabled:
                rospy.loginfo("Walking disabled")
                self.walking_enabled = False
    
    def generate_walking_trajectory(self):
        """Generate COM and foot trajectories for walking"""
        if not self.walking_enabled:
            return
            
        # Update time
        self.current_time += self.balance_controller.T
        
        # Determine current phase
        phase = (self.current_time % self.step_time) / self.step_time
        
        # Check if we need to switch support leg
        if phase < 0.01 and self.current_time > 0.01:  # New step
            self.left_foot_stance = not self.left_foot_stance
            self.step_count += 1
            rospy.loginfo(f"Step {self.step_count}, Left stance: {self.left_foot_stance}")
        
        # Generate COM trajectory
        com_trajectory = self.generate_com_trajectory(phase)
        
        # Generate foot trajectories  
        left_foot_traj, right_foot_traj = self.generate_foot_trajectories(phase)
        
        # Update balance controller reference
        self.balance_controller.reference_trajectory.update(com_trajectory)
        
        # Compute and send joint commands
        self.compute_joint_commands(left_foot_traj, right_foot_traj, com_trajectory)
    
    def generate_com_trajectory(self, phase):
        """Generate center of mass trajectory"""
        
        # Simple sinusoidal COM motion
        com_x = self.target_vel_x * self.current_time
        
        # Lateral COM shift for stability (shift toward stance leg)
        lateral_shift = 0.03  # 3cm shift
        if self.left_foot_stance:
            com_y = -lateral_shift * np.sin(np.pi * phase)
        else:
            com_y = lateral_shift * np.sin(np.pi * phase)
        
        # Vertical COM trajectory (slight bobbing)
        com_z_nominal = 0.814  # nominal height
        com_z = com_z_nominal - 0.01 * abs(np.sin(2 * np.pi * phase))
        
        # COM velocities (derivatives)
        com_x_dot = self.target_vel_x
        com_y_dot = 0.0
        com_z_dot = 0.0
        
        # COM accelerations
        com_x_ddot = 0.0
        com_y_ddot = 0.0
        com_z_ddot = 0.0
        
        return {
            'x_ref': com_x,
            'y_ref': com_y,
            'z_ref': com_z,
            'x_dot_ref': com_x_dot,
            'y_dot_ref': com_y_dot,
            'z_dot_ref': com_z_dot,
            'x_ddot_ref': com_x_ddot,
            'y_ddot_ref': com_y_ddot,
            'z_ddot_ref': com_z_ddot
        }
    
    def generate_foot_trajectories(self, phase):
        """Generate foot trajectories for swing and stance legs"""
        
        # Determine double support vs single support
        double_support = phase < self.double_support_ratio or phase > (1 - self.double_support_ratio)
        
        if double_support:
            # Both feet on ground
            left_foot_z = 0.0
            right_foot_z = 0.0
        else:
            # Single support - swing the non-stance foot
            swing_phase = (phase - self.double_support_ratio) / (1 - 2 * self.double_support_ratio)
            swing_height = self.step_height * np.sin(np.pi * swing_phase)
            
            if self.left_foot_stance:
                # Right foot swings
                left_foot_z = 0.0
                right_foot_z = swing_height
                
                # Move right foot forward
                step_progress = swing_phase
                self.right_foot_pos[0] = self.left_foot_pos[0] + self.step_length * step_progress
                
            else:
                # Left foot swings
                left_foot_z = swing_height
                right_foot_z = 0.0
                
                # Move left foot forward
                step_progress = swing_phase
                self.left_foot_pos[0] = self.right_foot_pos[0] + self.step_length * step_progress
        
        left_foot_traj = {
            'x': self.left_foot_pos[0],
            'y': self.left_foot_pos[1], 
            'z': left_foot_z
        }
        
        right_foot_traj = {
            'x': self.right_foot_pos[0],
            'y': self.right_foot_pos[1],
            'z': right_foot_z
        }
        
        return left_foot_traj, right_foot_traj
    
    def compute_joint_commands(self, left_foot_traj, right_foot_traj, com_traj):
        """Compute joint angles using simplified inverse kinematics"""
        
        # This is a simplified IK - you should replace with proper inverse kinematics
        # Based on your robot's dimensions and kinematic chain
        
        leg_length = 0.4  # approximate total leg length
        thigh_length = 0.2
        shin_length = 0.2
        
        # Compute leg joint angles for each leg
        left_joints = self.leg_inverse_kinematics(
            com_traj['x_ref'] - left_foot_traj['x'],
            com_traj['y_ref'] - left_foot_traj['y'], 
            com_traj['z_ref'] - left_foot_traj['z'],
            thigh_length, shin_length
        )
        
        right_joints = self.leg_inverse_kinematics(
            com_traj['x_ref'] - right_foot_traj['x'],
            com_traj['y_ref'] - right_foot_traj['y'],
            com_traj['z_ref'] - right_foot_traj['z'], 
            thigh_length, shin_length
        )
        
        # Publish joint commands
        self.hip_pitch_left_pub.publish(Float64(data=left_joints['hip_pitch']))
        self.hip_roll_left_pub.publish(Float64(data=left_joints['hip_roll']))
        self.knee_left_pub.publish(Float64(data=left_joints['knee']))
        
        self.hip_pitch_right_pub.publish(Float64(data=right_joints['hip_pitch']))
        self.hip_roll_right_pub.publish(Float64(data=right_joints['hip_roll']))
        self.knee_right_pub.publish(Float64(data=right_joints['knee']))
        
    def leg_inverse_kinematics(self, dx, dy, dz, l1, l2):
        """Simple 2D inverse kinematics for leg"""
        
        # Distance from hip to foot
        r = np.sqrt(dx**2 + dz**2)
        
        # Knee angle (using law of cosines)
        cos_knee = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
        cos_knee = np.clip(cos_knee, -1, 1)  # Ensure valid range
        knee_angle = np.pi - np.arccos(cos_knee)
        
        # Hip pitch angle
        alpha = np.arctan2(dz, dx)
        beta = np.arccos((l1**2 + r**2 - l2**2) / (2 * l1 * r))
        hip_pitch = alpha - beta
        
        # Hip roll angle (simplified)
        hip_roll = np.arctan2(dy, dz) * 0.5  # Simplified lateral balance
        
        return {
            'hip_pitch': hip_pitch,
            'hip_roll': hip_roll, 
            'knee': knee_angle
        }