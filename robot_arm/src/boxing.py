#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
import sys
import select
import tty
import termios
from pynput import keyboard
from roboticstoolbox import DHRobot, RevoluteDH

class BoxingNode(Node):

    def __init__(self):
        super().__init__('boxing_node')

        # Publish to the position and velocity controller topics
        self.joint_velocities_pub = self.create_publisher(Float64MultiArray, '/velocity_controller/commands', 10)

        self.settings = termios.tcgetattr(sys.stdin)
        
        # path parameters
        self.update_rate = 10
        self.dt = 1/self.update_rate

        # (x, y, z) of strike
        self.xi = 420
        self.yi = 0
        self.zi = 500

        self.numberOfPunches = 1
        self.radius = 200

        self.theta_dot1_value_list = []
        self.theta_dot2_value_list = []
        self.theta_dot3_value_list = []
        self.theta_dot4_value_list = []
        self.theta_dot5_value_list = []
        self.theta_dot6_value_list = []
        self.theta_dot7_value_list = []

        self.theta1_value_list = [0]
        self.theta2_value_list = [0.0]
        self.theta3_value_list = [0]
        self.theta4_value_list = [-pi/2]
        self.theta5_value_list = [0.0]
        self.theta6_value_list = [pi/2]
        self.theta7_value_list = [0.0]

        # DH table for the UR3e documented in the homework
        self.theta1, self.theta2, self.theta3, self.theta4, self.theta5, self.theta6, self.theta7 = sym.symbols('theta1:8')
        #                      theta,         d,          a,      alpha
        self.DH = sym.Matrix([[self.theta1,   0.1651,     0,      pi/2    ],
                              [self.theta2,   0,          0,      -pi/2   ],
                              [self.theta3,   0.25503,    0,      pi/2    ],
                              [self.theta4,   0,          0,      -pi/2   ],
                              [self.theta5,   0.42746,    0,      pi/2    ],
                              [self.theta6,   0,          0,      -pi/2   ],
                              [self.theta7,   0,          -0.085, 0       ]])

        # Create successive transormation matrices for each row of the DH table
        self.An = []
        for i in range(7):
            self.An.append(self.dh_transform(self.DH[i,0], self.DH[i,1]*1000, self.DH[i,2]*1000, self.DH[i,3]))

        # Calculate cumulative transformations
        self.Tn = [self.An[0]]
        for i in range(6):
            self.Tn.append(self.Tn[i] @ self.An[i+1])

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

    # Create generalized transformation matrix for DH table
    def dh_transform(self, theta, d, a, alpha):
        return sym.Matrix([
            [sym.cos(theta), -sym.sin(theta)*sym.cos(alpha), sym.sin(theta)*sym.sin(alpha), a*sym.cos(theta)],
            [sym.sin(theta), sym.cos(theta)*sym.cos(alpha), -sym.cos(theta)*sym.sin(alpha), a*sym.sin(theta)],
            [0, sym.sin(alpha), sym.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    # Takes the (x, y, z) of S and returns the cartesian path parameters
    def cartesian_path(self):
        theta1, theta2, theta3, theta4, theta5, theta6, theta7 = sym.symbols('theta1:8')
        t = sym.Symbol("t")
        time = []
        x_dot = []
        y_dot = []
        z_dot = []

        T1 = 2  # time for approach
        T2 = 5  # time for circle

        # approach section
        t1 = np.linspace(0, T1, T1 * self.update_rate)

        # Calculate the starting point (Home position)
        T_start = self.Tn[5].subs(
            {
                theta1: self.theta1_value_list[0],
                theta2: self.theta2_value_list[0],
                theta3: self.theta3_value_list[0],
                theta4: self.theta4_value_list[0],
                theta5: self.theta5_value_list[0],
                theta6: self.theta6_value_list[0],
                theta7: self.theta7_value_list[0],
            }
        )

        sym.pprint(T_start[:3, 3])

        # Calculate cartesian positions
        dx = (self.xi - T_start[0,3]) / T1
        dy = (self.yi - T_start[1,3]) / T1
        dz = (self.zi - T_start[2,3]) / T1
        x1 = (dx * t)
        y1 = (dy * t)
        z1 = (dz * t)

        # Calculate cartesian velocities
        x1_dot = x1.diff(t)
        y1_dot = y1.diff(t)
        z1_dot = z1.diff(t)

        # Append to path
        for i in range(len(t1)):
            time.append(t1[i])
            x_dot.append(x1_dot.subs({t: t1[i]}))
            y_dot.append(y1_dot.subs({t: t1[i]}))
            z_dot.append(z1_dot.subs({t: t1[i]}))

        # semi-circle section
        t2 = np.linspace(0, T2, T2 * self.update_rate)

        # Calculate cartesian positions
        x2 = -(self.radius * sym.sin((2 * pi / (T2/self.numberOfPunches)) * t))
        y2 = (0 * t)
        z2 = (self.radius * sym.cos((2 * pi / (T2/self.numberOfPunches)) * t))

        # Calculate cartesian velocities
        x2_dot = x2.diff(t)
        y2_dot = y2.diff(t)
        z2_dot = z2.diff(t)

        # Append to path
        for i in range(len(t2)):
            time.append(t2[i] + T1)
            x_dot.append(x2_dot.subs({t: t2[i]}))
            y_dot.append(y2_dot.subs({t: t2[i]}))
            z_dot.append(z2_dot.subs({t: t2[i]}))

        return time, x_dot, y_dot, z_dot

    def calculate_robot_toolbox(self):
        # Define the robot using DH parameters
        robot = DHRobot([
            RevoluteDH(a=0, alpha=np.pi/2, d=165.1),
            RevoluteDH(a=0, alpha=-np.pi/2, d=0),
            RevoluteDH(a=0, alpha=np.pi/2, d=255.03),
            RevoluteDH(a=0, alpha=-np.pi/2, d=0),
            RevoluteDH(a=0, alpha=np.pi/2, d=427.46),
            RevoluteDH(a=0, alpha=-np.pi/2, d=0),
            RevoluteDH(a=-85, alpha=0, d=0),
        ], name="7-DOF_Robot")

        # q = [0,0,0,0,0,0,0]

        # # Visualize the robot
        # robot.plot(q, block=True)

        # Calculate cartesian path parameters
        time, x_dot, y_dot, z_dot = self.cartesian_path()

        # Initialize positions, joint angles and velocities lists
        x_value_list = []
        y_value_list = []
        z_value_list = []

        print("Calculating...")
        # Repeat a for loop for every increment in time. Calculate joint positions and velocities at each increment in time.
        for i in range(np.size(time)):
            t = time[i]
            thetas = np.array([self.theta1_value_list[i],
                               self.theta2_value_list[i],
                               self.theta3_value_list[i],
                               self.theta4_value_list[i],
                               self.theta5_value_list[i],
                               self.theta6_value_list[i],
                               self.theta7_value_list[i],])


            # Compute the Jacobian
            J = robot.jacob0(thetas)

            effector = np.array([[x_dot[i]],[y_dot[i]],[z_dot[i]], [0], [0], [0]])

            # Convert J and effector to numeric arrays
            J_numeric = np.array(J, dtype=np.float64)
            effector_numeric = np.array(effector, dtype=np.float64)

            # Perform the pseudo-inverse operation
            theta_dots = np.linalg.pinv(J_numeric).dot(effector_numeric)
            
            self.theta1_value_list.append(self.theta1_value_list[i] + theta_dots[0]*self.dt)
            self.theta2_value_list.append(self.theta2_value_list[i] + theta_dots[1]*self.dt)
            self.theta3_value_list.append(self.theta3_value_list[i] + theta_dots[2]*self.dt)
            self.theta4_value_list.append(self.theta4_value_list[i] + theta_dots[3]*self.dt)
            self.theta5_value_list.append(self.theta5_value_list[i] + theta_dots[4]*self.dt)
            self.theta6_value_list.append(self.theta6_value_list[i] + theta_dots[5]*self.dt)
            self.theta7_value_list.append(self.theta7_value_list[i] + theta_dots[6]*self.dt)
            
            
            thetas1 = np.array([float(self.theta1_value_list[i+1]),
                                float(self.theta2_value_list[i+1]),
                                float(self.theta3_value_list[i+1]),
                                float(self.theta4_value_list[i+1]),
                                float(self.theta5_value_list[i+1]),
                                float(self.theta6_value_list[i+1]),
                                float(self.theta7_value_list[i+1]),])

            # Compute forward kinematics for the updated joint angles
            T = robot.fkine(thetas1)
            translation = T.t
            x_value_list.append(translation[0])
            y_value_list.append(translation[1])
            z_value_list.append(translation[2])
            
            
            
            self.theta_dot1_value_list.append(theta_dots[0])
            self.theta_dot2_value_list.append(theta_dots[1])
            self.theta_dot3_value_list.append(theta_dots[2])
            self.theta_dot4_value_list.append(theta_dots[3])
            self.theta_dot5_value_list.append(theta_dots[4])
            self.theta_dot6_value_list.append(theta_dots[5])
            self.theta_dot7_value_list.append(theta_dots[6])
        
        # Plot joint velocities
        plt.figure("Joint Velocities")
        plt.plot(time, self.theta_dot1_value_list, label="theta_dot1")
        plt.plot(time, self.theta_dot2_value_list, label="theta_dot2")
        plt.plot(time, self.theta_dot3_value_list, label="theta_dot3")
        plt.plot(time, self.theta_dot4_value_list, label="theta_dot4")
        plt.plot(time, self.theta_dot5_value_list, label="theta_dot5")
        plt.plot(time, self.theta_dot6_value_list, label="theta_dot6")
        plt.plot(time, self.theta_dot7_value_list, label="theta_dot7")
        plt.xlabel("Time (s)")
        plt.ylabel("Joint Velocities (rad/s)")
        plt.title("Joint Velocities over Time")
        plt.legend()

        # Display the end effector path in xyz
        fig1 = plt.figure("End Effector Positions")
        ax1 = fig1.add_subplot(111, projection="3d")
        ax1.plot3D(x_value_list, y_value_list, z_value_list, color="blue", label="Actual IK solutions")
        ax1.set_xlabel("X (mm)")
        ax1.set_ylabel("Y (mm)")
        ax1.set_zlabel("Z (mm)")
        ax1.legend()

        plt.show()

    def calculate_trajectory(self):
        # Calculate the Jacobian via the 1st method discussed in lecture
        # Calculate Z and O components
        Z = [sym.Matrix([0, 0, 1])]
        O = [sym.Matrix([0, 0, 0])]
        for T in self.Tn:
            Z.append(sym.Matrix(T[:3, 2]))
            O.append(sym.Matrix(T[:3, 3]))
        
        # Calculate Jv and Jw
        Jv = sym.zeros(3, 7)
        Jw = sym.zeros(3, 7)
        for i in range(6):
            Jv[:,i] = Z[i].cross(O[7] - O[i])
            Jw[:,i] = Z[i]

        self.J = sym.Matrix.vstack(Jv, Jw)

        # Initialize positions, joint angles and velocities lists
        x_value_list = []
        y_value_list = []
        z_value_list = []

        # Calculate cartesian path parameters
        time, x_dot, y_dot, z_dot = self.cartesian_path()

        # Calculate the forward velocity kinematics
        theta_dot1, theta_dot2, theta_dot3, theta_dot4, theta_dot5, theta_dot6, theta_dot7  = sym.symbols("theta_dot1:8")

        # Calculate end effector velocities
        forward_velocity_kinematics = self.J @ sym.Matrix(
            [[theta_dot1], [theta_dot2], [theta_dot3], [theta_dot4], [theta_dot5], [theta_dot6], [theta_dot7]]
        )

        # Calculate end effector velocities
        x_dot_symbol = sym.Symbol("x_dot_symbol")
        y_dot_symbol = sym.Symbol("y_dot_symbol")
        z_dot_symbol = sym.Symbol("z_dot_symbol")
        end_effector_velocities = sym.Matrix(
            [[x_dot_symbol], [y_dot_symbol], [z_dot_symbol], [0], [0], [0]]
        )

        print("Starting calculation")
        # Loop through toolpath to get inverse velocity kinematics
        for i in range(len(time)):
            if i % self.update_rate == 0:
                print(i, " out of ", len(time), " timestamps calculated")

            # Calculate numerical Jacobian
            J_Numerical = self.J.subs(
                {
                    self.theta1: self.theta1_value_list[0],
                    self.theta2: self.theta2_value_list[0],
                    self.theta3: self.theta3_value_list[0],
                    self.theta4: self.theta4_value_list[0],
                    self.theta5: self.theta5_value_list[0],
                    self.theta6: self.theta6_value_list[0],
                    self.theta7: self.theta7_value_list[0],
                }
            )
            J_inverse = J_Numerical.pinv()

            # Calculate the new end effector position
            positions_val = self.Tn[5].subs(
                {
                    self.theta1: self.theta1_value_list[i],
                    self.theta2: self.theta2_value_list[i],
                    self.theta3: self.theta3_value_list[i],
                    self.theta4: self.theta4_value_list[i],
                    self.theta5: self.theta5_value_list[i],
                    self.theta6: self.theta6_value_list[i],
                    self.theta7: self.theta7_value_list[i],
                }
            )
            x_value_list.append(positions_val[0, 3])
            y_value_list.append(positions_val[1, 3])
            z_value_list.append(positions_val[2, 3])

            # Calculate the new joint velocities
            joint_velocities_val = J_inverse @ end_effector_velocities.subs(
                {x_dot_symbol: x_dot[i], y_dot_symbol: y_dot[i], z_dot_symbol: z_dot[i]}
            )
            
            # Save joint velocitie
            self.theta_dot1_value_list.append(joint_velocities_val[0])
            self.theta_dot2_value_list.append(joint_velocities_val[1])
            self.theta_dot3_value_list.append(joint_velocities_val[2])
            self.theta_dot4_value_list.append(joint_velocities_val[3])
            self.theta_dot5_value_list.append(joint_velocities_val[4])
            self.theta_dot6_value_list.append(joint_velocities_val[5])
            self.theta_dot7_value_list.append(joint_velocities_val[6])

            # Calculate the new joint angles after 1 time step
            self.theta1_value_list.append(self.theta1_value_list[i] + self.theta_dot1_value_list[i] * self.dt)
            self.theta2_value_list.append(self.theta2_value_list[i] + self.theta_dot2_value_list[i] * self.dt)
            self.theta3_value_list.append(self.theta3_value_list[i] + self.theta_dot3_value_list[i] * self.dt)
            self.theta4_value_list.append(self.theta4_value_list[i] + self.theta_dot4_value_list[i] * self.dt)
            self.theta5_value_list.append(self.theta5_value_list[i] + self.theta_dot5_value_list[i] * self.dt)
            self.theta6_value_list.append(self.theta6_value_list[i] + self.theta_dot6_value_list[i] * self.dt)
            self.theta7_value_list.append(self.theta7_value_list[i] + self.theta_dot7_value_list[i] * self.dt)

        # Plot joint velocities
        plt.figure("Joint Velocities")
        plt.plot(time, self.theta_dot1_value_list, label="theta_dot1")
        plt.plot(time, self.theta_dot2_value_list, label="theta_dot2")
        plt.plot(time, self.theta_dot3_value_list, label="theta_dot3")
        plt.plot(time, self.theta_dot4_value_list, label="theta_dot4")
        plt.plot(time, self.theta_dot5_value_list, label="theta_dot5")
        plt.plot(time, self.theta_dot6_value_list, label="theta_dot6")
        plt.plot(time, self.theta_dot7_value_list, label="theta_dot7")
        plt.xlabel("Time (s)")
        plt.ylabel("Joint Velocities (rad/s)")
        plt.title("Joint Velocities over Time")
        plt.legend()

        # Plot joint angles
        plt.figure("Joint Angles")
        plt.plot(time, self.theta1_value_list[0:len(time)], label="theta1")
        plt.plot(time, self.theta2_value_list[0:len(time)], label="theta2")
        plt.plot(time, self.theta3_value_list[0:len(time)], label="theta3")
        plt.plot(time, self.theta4_value_list[0:len(time)], label="theta4")
        plt.plot(time, self.theta5_value_list[0:len(time)], label="theta5")
        plt.plot(time, self.theta6_value_list[0:len(time)], label="theta6")
        plt.plot(time, self.theta7_value_list[0:len(time)], label="theta7")
        plt.xlabel("Time (s)")
        plt.ylabel("Joint Angles (rad)")
        plt.title("Joint Angles over Time")
        plt.legend()

        # Display the end effector path in xyz
        fig1 = plt.figure("End Effector Positions")
        ax1 = fig1.add_subplot(111, projection="3d")
        ax1.plot3D(x_value_list, y_value_list, z_value_list, color="blue", label="Actual IK solutions")
        ax1.set_xlabel("X (mm)")
        ax1.set_ylabel("Y (mm)")
        ax1.set_zlabel("Z (mm)")
        ax1.legend()

        plt.show()

        return True
    
    def send_joint_velocities(self):
        joint_velocities = Float64MultiArray()
        
        time_previous = self.get_clock().now()
        i = 0
        approaching = True
        while approaching :
            time_current = self.get_clock().now()
            if (time_current - time_previous) >= rclpy.duration.Duration(seconds=1.0) :
                approaching = False
            joint_velocities.data = [float(self.theta1_value_list[0]),
                                     float(self.theta2_value_list[0]),
                                     float(self.theta3_value_list[0]),
                                     float(self.theta4_value_list[0]),
                                     float(self.theta5_value_list[0]),
                                     float(self.theta6_value_list[0]),
                                     float(self.theta7_value_list[0])]

            self.joint_velocities_pub.publish(joint_velocities)

        print("finished approach")

        while i < len(self.theta_dot1_value_list):
            time_current = self.get_clock().now()
            if ((time_current - time_previous) >= rclpy.duration.Duration(seconds=self.dt)):
                time_previous = time_current
                joint_velocities.data = [float(self.theta_dot1_value_list[i]),
                                         float(self.theta_dot2_value_list[i]),
                                         float(self.theta_dot3_value_list[i]),
                                         float(self.theta_dot4_value_list[i]),
                                         float(self.theta_dot5_value_list[i]),
                                         float(self.theta_dot6_value_list[i]),
                                         float(self.theta_dot7_value_list[i])]
                i += 1
                self.joint_velocities_pub.publish(joint_velocities)
     
    def run_control(self):
        joint_velocities = Float64MultiArray()
        self.msg = """
        Lets Go Boxing!
        ---------------------------
        p : plan path

        e : execute path

        Esc to quit
        """

        self.get_logger().info(self.msg)


        while True:
            key = self.getKey()
            if key is not None:
                if key == '\x1b':  # Escape key
                    break
                elif key == 'p':  # Plan
                    self.get_logger().info('Planning Trajectory...')
                    self.calculate_trajectory()
                    self.get_logger().info('Trajectory Planned!')
                elif key == 'r':  # Plan
                    self.get_logger().info('Planning Trajectory...')
                    self.calculate_robot_toolbox()
                    self.get_logger().info('Trajectory Planned!')
                elif key == 'e':  # Execute
                    self.get_logger().info('Executing Trajectory...')
                    self.send_joint_velocities()
                    self.get_logger().info('Trajectory Executed!')

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
    node.run_control()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()