#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import sys
import select
import tty
import termios
from pynput import keyboard

# Define key codes
ANG_VEL_STEP_SIZE = (1/180) * 3.141592654

class KeyboardControlNode(Node):

    def __init__(self):
        super().__init__('keyboard_control_node')

        # Publish to the position and velocity controller topics
        self.joint_velocities_pub = self.create_publisher(Float64MultiArray, '/velocity_controller/commands', 10)
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)

        self.settings = termios.tcgetattr(sys.stdin)

    # Get the key press from the terminal
    def getKey(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def run_keyboard_control(self):
        self.msg = """
        Control Your Arm!
        ---------------------------
        Select Joint:
        w
        s

        Move joint:
        a d

        Stop:
        q

        Esc to quit
        """

        self.get_logger().info(self.msg)
        joint_velocities = Float64MultiArray()
        angular_vel=0.0
        joint = 0


        while True:
            key = self.getKey()
            if key is not None:
                if key == '\x1b':  # Escape key
                    break
                elif key == 'q':  # Quit
                    angular_vel=0.0
                elif key == 'a':  # Forward
                    angular_vel += ANG_VEL_STEP_SIZE
                elif key == 'd':  # Reverse
                    angular_vel -= ANG_VEL_STEP_SIZE
                elif key == 'w':
                    if joint < 7:
                        joint += 1
                elif key == 's':
                    if joint > 0:
                        joint -= 1

                print("Linear Velocity",angular_vel)
                print("Joint", joint)
                # Publish the control message
                joint_velocities.data = [0.0, 
                                         0.0,
                                         0.0, 
                                         0.0, 
                                         0.0, 
                                         0.0, 
                                         0.0]

                joint_velocities.data[joint] = angular_vel

                self.joint_velocities_pub.publish(joint_velocities)

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardControlNode()
    node.run_keyboard_control()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()