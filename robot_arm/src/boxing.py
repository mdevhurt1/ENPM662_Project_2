#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray


class BoxingNode(Node):

    def __init__(self):
        super().__init__('boxing_node')

        # Publish to the position and velocity controller topics
        self.joint_velocities_pub = self.create_publisher(Float64MultiArray, '/velocity_controller/commands', 10)
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)

    def send_joint_velocities(self):
        joint_velocities = Float64MultiArray()

        # Publish the control message
        joint_velocities.data = [0.0, 
                                 0.0,
                                 0.0, 
                                 0.0, 
                                 0.0, 
                                 0.0, 
                                 0.0]

        self.joint_velocities_pub.publish(joint_velocities)

def main(args=None):
    rclpy.init(args=args)
    node = BoxingNode()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()