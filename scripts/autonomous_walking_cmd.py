#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist

def autonomous_walk():
    rospy.init_node('autonomous_walk_cmd')
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    rate = rospy.Rate(10)  # 10 Hz

    cmd = Twist()

    def send_command(duration_sec, linear_x=0.0, linear_y=0.0, angular_z=0.0):
        t_start = rospy.Time.now()
        cmd.linear.x = linear_x
        cmd.linear.y = linear_y
        cmd.angular.z = angular_z
        rospy.loginfo(f"Walking: x={linear_x}, y={linear_y}, yaw={angular_z}")
        while (rospy.Time.now() - t_start).to_sec() < duration_sec and not rospy.is_shutdown():
            pub.publish(cmd)
            rate.sleep()

    # Sequence of commands
    send_command(5.0, linear_x=0.1)          # Walk forward 5s
    send_command(3.0, angular_z=0.4)         # Turn left 3s
    send_command(4.0, linear_x=0.08, linear_y=0.04)  # Diagonal 4s
    send_command(2.0)                        # Stop (all zeros)

    rospy.loginfo("Autonomous walking sequence finished")

if __name__ == '__main__':
    try:
        autonomous_walk()
    except rospy.ROSInterruptException:
        pass
