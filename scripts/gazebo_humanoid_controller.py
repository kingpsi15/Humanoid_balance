#!/usr/bin/env python3
""" 
Gazebo Integration for Humanoid Balance Controller
Integrates the perfect balance controller with Gazebo simulation
"""

import rospy
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, Float64
from tf2_msgs.msg import TFMessage
import tf2_ros
import tf2_geometry_msgs
from scipy.linalg import solve_discrete_are
import threading
import time
from gazebo_msgs.msg import ModelStates
from walking_pattern_generator import WalkingPatternGenerator

class GazeboHumanoidController:
    def __init__(self):
        rospy.init_node('humanoid_balance_controller', anonymous=True)
        
        # Controller parameters (from original file)
        self.T = 0.005  # sampling time [s] 
        self.zc = 0.814  # height of center of mass [m]
        self.g = 9.81   # gravity [m/s^2]
        
        # System matrices
        self.A = np.array([
            [1, self.T, self.T**2/2],
            [0, 1, self.T],
            [0, 0, 1]
        ])
        self.B = np.array([[self.T**3/6], [self.T**2/2], [self.T]])
        self.C_zmp = np.array([[1, 0, -self.zc/self.g]])
        self.C_pos = np.array([[1, 0, 0]])
        
        # Initialize LQR controller
        self.setup_lqr_controller()
        
        # State variables
        self.x_state = np.zeros(3)  # [pos, vel, acc]
        self.y_state = np.zeros(3)
        self.x_integral = 0.0
        self.y_integral = 0.0
        
        # Robot state from sensors
        self.com_position = np.zeros(3)
        self.com_velocity = np.zeros(3)
        self.com_acceleration = np.zeros(3)
        self.imu_data = None
        self.joint_states = None
        
        # Reference trajectory
        self.reference_trajectory = {
            'x_ref': 0.0, 'y_ref': 0.0,
            'x_dot_ref': 0.0, 'y_dot_ref': 0.0,
            'x_ddot_ref': 0.0, 'y_ddot_ref': 0.0
        }
        
        # Control outputs
        self.control_commands = {'u_x': 0.0, 'u_y': 0.0}
        
        # ZMP constraints
        self.foot_length = 0.24
        self.foot_width = 0.12
        self.safety_margin = 0.8
        
        # ROS Publishers and Subscribers
        self.setup_ros_interface()
        
        # Control loop timer
        self.control_timer = rospy.Timer(rospy.Duration(self.T), self.control_callback)
        
        rospy.loginfo("Humanoid Balance Controller initialized")

    def setup_lqr_controller(self):
        """Setup LQR controller with integral action"""
        try:
            # Augmented system: [x, x_dot, x_ddot, x_integral]
            A_aug = np.zeros((4, 4))
            A_aug[:3, :3] = self.A
            A_aug[3, :3] = self.C_pos.flatten()
            A_aug[3, 3] = 1
            
            B_aug = np.zeros((4, 1))
            B_aug[:3, 0] = self.B.flatten()
            
            # LQR weights
            Q = np.diag([10000, 1000, 1, 1000])  # Heavy penalty on position and integral
            R = np.array([[0.01]])  # Allow aggressive control
            
            # Solve discrete-time Riccati equation
            P = solve_discrete_are(A_aug, B_aug, Q, R)
            self.K = np.linalg.inv(R + B_aug.T @ P @ B_aug) @ B_aug.T @ P @ A_aug
            
            # Check stability
            A_cl = A_aug - B_aug @ self.K
            cl_eigenvals = np.linalg.eigvals(A_cl)
            
            if np.all(np.abs(cl_eigenvals) < 1):
                rospy.loginfo("LQR controller initialized successfully - system is stable")
                self.use_lqr = True
            else:
                rospy.logwarn("LQR system is unstable, falling back to PD control")
                self.use_lqr = False
                
        except Exception as e:
            rospy.logerr(f"LQR initialization failed: {e}")
            self.use_lqr = False

    def setup_ros_interface(self):
        """Setup ROS publishers and subscribers"""
        
        # Subscribers - Robot state feedback
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_state_callback)
        rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
        
        # Subscribers - Reference commands
        rospy.Subscriber('/balance_controller/reference', PoseStamped, self.reference_callback)
        rospy.Subscriber('/balance_controller/trajectory', Float64MultiArray, self.trajectory_callback)
        
        # Publishers - Control outputs
        self.ankle_roll_left_pub = rospy.Publisher('/humanoid/ankle_roll_left_effort_controller/command', 
                                                 Float64, queue_size=1)
        self.ankle_roll_right_pub = rospy.Publisher('/humanoid/ankle_roll_right_effort_controller/command', 
                                                  Float64, queue_size=1)
        self.ankle_pitch_left_pub = rospy.Publisher('/humanoid/ankle_pitch_left_effort_controller/command', 
                                                  Float64, queue_size=1)
        self.ankle_pitch_right_pub = rospy.Publisher('/humanoid/ankle_pitch_right_effort_controller/command', 
                                                   Float64, queue_size=1)
        
        # Publishers - Debug/monitoring
        self.zmp_pub = rospy.Publisher('/balance_controller/zmp', PoseStamped, queue_size=1)
        self.control_debug_pub = rospy.Publisher('/balance_controller/debug', Float64MultiArray, queue_size=1)
        self.error_pub = rospy.Publisher('/balance_controller/tracking_error', PoseStamped, queue_size=1)

    def model_state_callback(self, msg):
        """Extract robot COM position from Gazebo model states"""
        try:
            # Find your robot model in the message
            robot_index = msg.name.index('humanoid')  # Change to your robot's name
            
            pose = msg.pose[robot_index]
            twist = msg.twist[robot_index]
            
            # Extract COM position and velocity
            self.com_position = np.array([
                pose.position.x,
                pose.position.y, 
                pose.position.z
            ])
            
            self.com_velocity = np.array([
                twist.linear.x,
                twist.linear.y,
                twist.linear.z
            ])
            
        except (ValueError, IndexError) as e:
            rospy.logwarn_throttle(5.0, f"Robot model not found in Gazebo states: {e}")

    def imu_callback(self, msg):
        """Process IMU data for acceleration feedback"""
        self.imu_data = msg
        
        # Extract linear acceleration (remove gravity)
        self.com_acceleration = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z + self.g  # Add gravity back for Z
        ])

    def joint_state_callback(self, msg):
        """Process joint states for full robot state"""
        self.joint_states = msg

    def reference_callback(self, msg):
        """Update reference trajectory from external command"""
        self.reference_trajectory['x_ref'] = msg.pose.position.x
        self.reference_trajectory['y_ref'] = msg.pose.position.y

    def trajectory_callback(self, msg):
        """Update full trajectory with derivatives"""
        if len(msg.data) >= 6:
            self.reference_trajectory['x_ref'] = msg.data[0]
            self.reference_trajectory['y_ref'] = msg.data[1] 
            self.reference_trajectory['x_dot_ref'] = msg.data[2]
            self.reference_trajectory['y_dot_ref'] = msg.data[3]
            self.reference_trajectory['x_ddot_ref'] = msg.data[4]
            self.reference_trajectory['y_ddot_ref'] = msg.data[5]

    def compute_feedforward_control(self, x_ddot_ref, x_ddot_current, y_ddot_ref, y_ddot_current):
        """Compute feedforward control for perfect tracking"""
        u_ff_x = (x_ddot_ref - x_ddot_current) / self.T
        u_ff_y = (y_ddot_ref - y_ddot_current) / self.T
        return u_ff_x, u_ff_y

    def compute_zmp(self, x_state, y_state):
        """Compute current ZMP from robot state"""
        zmp_x = (self.C_zmp @ x_state)[0]
        zmp_y = (self.C_zmp @ y_state)[0]
        return zmp_x, zmp_y

    def enforce_zmp_constraints(self, u_x, u_y, x_state, y_state):
        """Enforce ZMP constraints by modifying control inputs"""
        # Predict next states
        x_state_next = self.A @ x_state + self.B.flatten() * u_x
        y_state_next = self.A @ y_state + self.B.flatten() * u_y
        
        # Compute predicted ZMP
        zmp_x_pred, zmp_y_pred = self.compute_zmp(x_state_next, y_state_next)
        
        # Define ZMP bounds relative to current foot position
        zmp_x_min = -self.foot_length/2 * self.safety_margin + x_state[0]
        zmp_x_max = self.foot_length/2 * self.safety_margin + x_state[0]
        zmp_y_min = -self.foot_width/2 * self.safety_margin + y_state[0]
        zmp_y_max = self.foot_width/2 * self.safety_margin + y_state[0]
        
        # Constrain ZMP by scaling control inputs
        if zmp_x_pred < zmp_x_min or zmp_x_pred > zmp_x_max:
            if zmp_x_pred != 0:
                scale_factor = min(abs(zmp_x_min / zmp_x_pred), abs(zmp_x_max / zmp_x_pred), 0.8)
                u_x *= scale_factor
                
        if zmp_y_pred < zmp_y_min or zmp_y_pred > zmp_y_max:
            if zmp_y_pred != 0:
                scale_factor = min(abs(zmp_y_min / zmp_y_pred), abs(zmp_y_max / zmp_y_pred), 0.8)
                u_y *= scale_factor
        
        return u_x, u_y

    def control_callback(self, event):
        """Main control loop - runs at 200Hz"""
        
        # Update state from sensor feedback
        if self.com_position is not None and self.com_velocity is not None:
            self.x_state[0] = self.com_position[0]  # X position
            self.x_state[1] = self.com_velocity[0]  # X velocity
            
            self.y_state[0] = self.com_position[1]  # Y position  
            self.y_state[1] = self.com_velocity[1]  # Y velocity
            
            if self.com_acceleration is not None:
                self.x_state[2] = self.com_acceleration[0]  # X acceleration
                self.y_state[2] = self.com_acceleration[1]  # Y acceleration

        # Get reference trajectory
        x_ref = self.reference_trajectory['x_ref']
        y_ref = self.reference_trajectory['y_ref']
        x_dot_ref = self.reference_trajectory['x_dot_ref']
        y_dot_ref = self.reference_trajectory['y_dot_ref']
        x_ddot_ref = self.reference_trajectory['x_ddot_ref']
        y_ddot_ref = self.reference_trajectory['y_ddot_ref']

        # Compute feedforward control
        u_ff_x, u_ff_y = self.compute_feedforward_control(
            x_ddot_ref, self.x_state[2], y_ddot_ref, self.y_state[2])

        if self.use_lqr:
            # LQR feedback control
            pos_error_x = x_ref - self.x_state[0]
            pos_error_y = y_ref - self.y_state[0]
            
            # Update integral terms
            self.x_integral += pos_error_x * self.T
            self.y_integral += pos_error_y * self.T
            
            # Anti-windup
            integral_limit = 0.1
            self.x_integral = np.clip(self.x_integral, -integral_limit, integral_limit)
            self.y_integral = np.clip(self.y_integral, -integral_limit, integral_limit)
            
            # Augmented state vectors
            x_aug = np.array([self.x_state[0], self.x_state[1], self.x_state[2], self.x_integral])
            y_aug = np.array([self.y_state[0], self.y_state[1], self.y_state[2], self.y_integral])
            
            x_ref_aug = np.array([x_ref, x_dot_ref, x_ddot_ref, 0])
            y_ref_aug = np.array([y_ref, y_dot_ref, y_ddot_ref, 0])
            
            # LQR control
            u_fb_x = float(self.K @ (x_ref_aug - x_aug))
            u_fb_y = float(self.K @ (y_ref_aug - y_aug))
            
        else:
            # PD feedback control
            kp, kd = 1000.0, 100.0
            pos_error_x = x_ref - self.x_state[0]
            pos_error_y = y_ref - self.y_state[0]
            vel_error_x = x_dot_ref - self.x_state[1]
            vel_error_y = y_dot_ref - self.y_state[1]
            
            u_fb_x = kp * pos_error_x + kd * vel_error_x
            u_fb_y = kp * pos_error_y + kd * vel_error_y

        # Total control: feedforward + feedback
        u_x = u_ff_x + u_fb_x
        u_y = u_ff_y + u_fb_y
        
        # Enforce ZMP constraints
        u_x, u_y = self.enforce_zmp_constraints(u_x, u_y, self.x_state, self.y_state)
        
        # Saturate control inputs
        max_control = 10.0
        u_x = np.clip(u_x, -max_control, max_control)
        u_y = np.clip(u_y, -max_control, max_control)
        
        # Store control commands
        self.control_commands['u_x'] = u_x
        self.control_commands['u_y'] = u_y
        
        # Convert jerk commands to ankle joint commands
        self.send_ankle_commands(u_x, u_y)
        
        # Publish debug information
        self.publish_debug_info()

    def setup_ros_interface(self):
        """Setup ROS publishers and subscribers"""
        
        # --- Subscribers: State feedback ---
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_state_callback)
        rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
        
        # --- Subscribers: Reference commands ---
        rospy.Subscriber('/balance_controller/reference', PoseStamped, self.reference_callback)
        rospy.Subscriber('/balance_controller/trajectory', Float64MultiArray, self.trajectory_callback)

        # --- Publishers: Effort control outputs ---
        self.ankle_roll_left_pub = rospy.Publisher(
            '/humanoid/ankle_roll_left_effort_controller/command', Float64, queue_size=1)
        self.ankle_roll_right_pub = rospy.Publisher(
            '/humanoid/ankle_roll_right_effort_controller/command', Float64, queue_size=1)
        self.ankle_pitch_left_pub = rospy.Publisher(
            '/humanoid/ankle_pitch_left_effort_controller/command', Float64, queue_size=1)
        self.ankle_pitch_right_pub = rospy.Publisher(
            '/humanoid/ankle_pitch_right_effort_controller/command', Float64, queue_size=1)

        # --- Publishers: Debug and monitoring ---
        self.zmp_pub = rospy.Publisher(
            '/balance_controller/zmp', PoseStamped, queue_size=1)
        self.control_debug_pub = rospy.Publisher(
            '/balance_controller/debug', Float64MultiArray, queue_size=1)
        self.error_pub = rospy.Publisher(
            '/balance_controller/tracking_error', PoseStamped, queue_size=1)


    def send_ankle_commands(self, u_x, u_y):
        """Convert control inputs to ankle torque commands"""
        
        # Convert jerk commands to torques using your robot's dynamics
        # This requires proper scaling based on your robot's parameters
        torque_scale_x = 10.0  # Nm per unit control input - tune this
        torque_scale_y = 10.0  # Nm per unit control input - tune this
        
        # Ankle pitch torque for X direction (sagittal plane)
        ankle_pitch_torque = u_x * torque_scale_x
        
        # Ankle roll torque for Y direction (frontal plane)  
        ankle_roll_torque = u_y * torque_scale_y
        
        # Publish torque commands
        self.ankle_pitch_left_pub.publish(Float64(data=ankle_pitch_torque))
        self.ankle_pitch_right_pub.publish(Float64(data=ankle_pitch_torque))
        self.ankle_roll_left_pub.publish(Float64(data=ankle_roll_torque))
        self.ankle_roll_right_pub.publish(Float64(data=ankle_roll_torque))

    def publish_debug_info(self):
        """Publish debug and monitoring information"""
        
        # Publish ZMP
        zmp_x, zmp_y = self.compute_zmp(self.x_state, self.y_state)
        zmp_msg = PoseStamped()
        zmp_msg.header.stamp = rospy.Time.now()
        zmp_msg.header.frame_id = "base_link"
        zmp_msg.pose.position.x = zmp_x
        zmp_msg.pose.position.y = zmp_y
        self.zmp_pub.publish(zmp_msg)
        
        # Publish control debug info
        debug_msg = Float64MultiArray()
        debug_msg.data = [
            self.control_commands['u_x'], self.control_commands['u_y'],
            self.x_state[0], self.x_state[1], self.x_state[2],
            self.y_state[0], self.y_state[1], self.y_state[2],
            zmp_x, zmp_y
        ]
        self.control_debug_pub.publish(debug_msg)
        
        # Publish tracking error
        error_msg = PoseStamped()
        error_msg.header.stamp = rospy.Time.now()
        error_msg.header.frame_id = "base_link"
        error_msg.pose.position.x = self.reference_trajectory['x_ref'] - self.x_state[0]
        error_msg.pose.position.y = self.reference_trajectory['y_ref'] - self.y_state[0]
        self.error_pub.publish(error_msg)

    def run(self):
        """Main execution loop"""
        rospy.loginfo("Humanoid balance controller running...")
        rospy.spin()

if __name__ == '__main__':
    try:
        controller = GazeboHumanoidController()
        controller.run()

    except rospy.ROSInterruptException:
        rospy.loginfo("Humanoid balance controller shutting down...")