#!/usr/bin/env python3
"""
Trajectory Generator for Humanoid Balance Controller Testing
Generates sinusoidal reference trajectories matching the original test
"""

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
import math

class TrajectoryGenerator:
    def __init__(self):
        rospy.init_node('trajectory_generator', anonymous=True)
        
        # Trajectory parameters (matching original file)
        self.amplitude_x = 0.005  # 5mm amplitude
        self.amplitude_y = 0.003  # 3mm amplitude  
        self.freq_x = 0.3  # Hz
        self.freq_y = 0.5  # Hz
        
        # Publishers
        self.trajectory_pub = rospy.Publisher('/balance_controller/trajectory', 
                                            Float64MultiArray, queue_size=1)
        self.reference_pub = rospy.Publisher('/balance_controller/reference', 
                                           PoseStamped, queue_size=1)
        
        # Timer for trajectory generation
        self.rate = rospy.Rate(200)  # 200 Hz to match controller
        self.start_time = rospy.Time.now()
        
        rospy.loginfo("Trajectory Generator initialized")

    def generate_trajectory(self):
        """Generate sinusoidal reference trajectory with derivatives"""
        
        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            t = (current_time - self.start_time).to_sec()
            
            # Position references
            x_ref = self.amplitude_x * math.sin(2 * math.pi * self.freq_x * t)
            y_ref = self.amplitude_y * math.sin(2 * math.pi * self.freq_y * t)
            
            # Velocity references (first derivative)
            x_dot_ref = self.amplitude_x * 2 * math.pi * self.freq_x * math.cos(2 * math.pi * self.freq_x * t)
            y_dot_ref = self.amplitude_y * 2 * math.pi * self.freq_y * math.cos(2 * math.pi * self.freq_y * t)
            
            # Acceleration references (second derivative)
            x_ddot_ref = -self.amplitude_x * (2 * math.pi * self.freq_x)**2 * math.sin(2 * math.pi * self.freq_x * t)
            y_ddot_ref = -self.amplitude_y * (2 * math.pi * self.freq_y)**2 * math.sin(2 * math.pi * self.freq_y * t)
            
            # Publish full trajectory with derivatives
            traj_msg = Float64MultiArray()
            traj_msg.data = [x_ref, y_ref, x_dot_ref, y_dot_ref, x_ddot_ref, y_ddot_ref]
            self.trajectory_pub.publish(traj_msg)
            
            # Publish simple reference for other nodes
            ref_msg = PoseStamped()
            ref_msg.header.stamp = current_time
            ref_msg.header.frame_id = "base_link"
            ref_msg.pose.position.x = x_ref
            ref_msg.pose.position.y = y_ref
            ref_msg.pose.position.z = 0.0
            self.reference_pub.publish(ref_msg)
            
            self.rate.sleep()

    def run(self):
        """Main execution"""
        rospy.loginfo("Starting trajectory generation...")
        try:
            self.generate_trajectory()
        except rospy.ROSInterruptException:
            rospy.loginfo("Trajectory generator shutting down...")

if __name__ == '__main__':
    generator = TrajectoryGenerator()
    generator.run()