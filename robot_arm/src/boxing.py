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
        self.joint_position_pub = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        
        self.settings = termios.tcgetattr(sys.stdin)
        
        # path parameters
        self.update_rate = 10
        self.dt = 1/self.update_rate
        self.maximum_joint_velocity = (180 / 180) * pi # limit the joint velocity
        self.scale = 0.1 # amount to scale joint velocities by if they exceed maximum
        self.offset = (1 / 180) * pi # amount of offset to escape singularity

        # (x, y, z) of strike
        self.xi = 300
        self.yi = 0
        self.zi = 1500

        self.theta_dot1_value_list = []
        self.theta_dot2_value_list = []
        self.theta_dot3_value_list = []
        self.theta_dot4_value_list = []
        self.theta_dot5_value_list = []
        self.theta_dot6_value_list = []
        self.theta_dot7_value_list = []

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

        T1 = 5  # time for approach
        T2 = 5  # time for circle

        # approach section
        t1 = np.linspace(0, T1, T1 * self.update_rate)

        # Calculate the starting point (Home position)
        T_start = self.Tn[5].subs(
            {
                theta1: 0.0,
                theta2: 0.0,
                theta3: -pi/2,
                theta4: -pi/2,
                theta5: 0.0,
                theta6: 0.0,
                theta7: 0.0,
            }
        )

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
        x2 = 50 - (50 * sym.cos((2 * pi / T2) * t))
        y2 = (0 * t)
        z2 = (50 * sym.sin((2 * pi / T2) * t))

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
            RevoluteDH(a=0, alpha=np.pi/2, d=0.1651),
            RevoluteDH(a=0, alpha=-np.pi/2, d=0),
            RevoluteDH(a=0, alpha=np.pi/2, d=0.25503),
            RevoluteDH(a=0, alpha=-np.pi/2, d=0),
            RevoluteDH(a=0, alpha=np.pi/2, d=0.42746),
            RevoluteDH(a=0, alpha=-np.pi/2, d=0),
            RevoluteDH(a=-0.085, alpha=0, d=0),
        ], name="7-DOF_Robot")

        q = [0,pi/2,0,0,0,0,0]

        # Visualize the robot
        robot.plot(q, block=True)

        time = np.linspace(0,5,5*self.update_rate)
        delta_T = self.dt
        theta1_v = pi/4
        theta2_v = pi/4
        theta3_v = pi/4 
        theta4_v = pi/4 
        theta5_v = pi/4
        theta6_v = pi/4
        theta7_v = pi/4
        thetas = np.array([theta1_v, theta2_v, theta3_v, theta4_v, theta5_v, theta6_v, theta7_v])

        Px_v_array = np.zeros(51)
        Py_v_array = np.zeros(51)
        Pz_v_array = np.zeros(51)

        theta1_v_array = np.zeros(51)
        theta2_v_array = np.zeros(51)
        theta3_v_array = np.zeros(51)
        theta4_v_array = np.zeros(51)
        theta5_v_array = np.zeros(51)
        theta6_v_array = np.zeros(51)
        theta7_v_array = np.zeros(51)

        theta1_dot_v_array = np.zeros(51)
        theta2_dot_v_array = np.zeros(51)
        theta3_dot_v_array = np.zeros(51)
        theta4_dot_v_array = np.zeros(51)
        theta5_dot_v_array = np.zeros(51)
        theta6_dot_v_array = np.zeros(51)
        theta7_dot_v_array = np.zeros(51)

        print("Calculating...")
        # Repeat a for loop for every increment in time. Calculate joint positions and velocities at each increment in time.
        for i in range(np.size(time)):
            t = time[i]

            # Circle velocity profile
            Px_prime = 0.0
            Py_prime = 0.01
            Pz_prime = 0.0
            
            # Compute the Jacobian
            Jacobian_v = robot.jacob0(thetas)
            
            effector = np.array([[Px_prime],[Py_prime],[Pz_prime], [0], [0], [0]])
            theta_dots = np.linalg.pinv(Jacobian_v).dot(effector)
            
            theta1_v = theta1_v + theta_dots[0]*delta_T
            theta2_v = theta2_v + theta_dots[1]*delta_T
            theta3_v = theta3_v + theta_dots[2]*delta_T
            theta4_v = theta4_v + theta_dots[3]*delta_T
            theta5_v = theta5_v + theta_dots[4]*delta_T
            theta6_v = theta6_v + theta_dots[5]*delta_T
            theta7_v = theta7_v + theta_dots[6]*delta_T
            
            theta1_v = float(theta1_v[0])
            theta2_v = float(theta2_v[0])
            theta3_v = float(theta3_v[0])
            theta4_v = float(theta4_v[0])
            theta5_v = float(theta5_v[0])
            theta6_v = float(theta6_v[0])
            theta7_v = float(theta7_v[0])
            thetas = np.array([theta1_v, theta2_v, theta3_v, theta4_v, theta5_v, theta6_v, theta7_v])
            
            # Compute forward kinematics for the updated joint angles
            T = robot.fkine(thetas)
            translation = T.t
            Px_v = translation[0]
            Py_v = translation[1]
            Pz_v = translation[2]
            
            Px_v_array[i] = Px_v
            Py_v_array[i] = Py_v
            Pz_v_array[i] = Pz_v
            theta1_v_array[i] = theta1_v
            theta2_v_array[i] = theta2_v
            theta3_v_array[i] = theta3_v
            theta4_v_array[i] = theta4_v
            theta5_v_array[i] = theta5_v
            theta6_v_array[i] = theta6_v
            theta7_v_array[i] = theta7_v
            
            theta1_dot_v_array[i] = theta_dots[0]
            theta2_dot_v_array[i] = theta_dots[1]
            theta3_dot_v_array[i] = theta_dots[2]
            theta4_dot_v_array[i] = theta_dots[3]
            theta5_dot_v_array[i] = theta_dots[4]
            theta6_dot_v_array[i] = theta_dots[5]
            theta7_dot_v_array[i] = theta_dots[6]
            
        self.theta_dot1_value_list = theta1_dot_v_array
        self.theta_dot2_value_list = theta2_dot_v_array
        self.theta_dot3_value_list = theta3_dot_v_array
        self.theta_dot4_value_list = theta4_dot_v_array
        self.theta_dot5_value_list = theta5_dot_v_array
        self.theta_dot6_value_list = theta6_dot_v_array
        self.theta_dot7_value_list = theta7_dot_v_array

    def calculate_trajectory(self):
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
        theta1_value_list = [0.0]
        theta2_value_list = [0.0]
        theta3_value_list = [-pi/2]
        theta4_value_list = [-pi/2]
        theta5_value_list = [0.0]
        theta6_value_list = [0.0]
        theta7_value_list = [0.0]

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

            # Escape the home position singularity
            # Calculate numerical Jacobian
            J_Numerical = self.J.subs(
                {
                    self.theta1: theta1_value_list[0],
                    self.theta2: theta2_value_list[0],
                    self.theta3: theta3_value_list[0],
                    self.theta4: theta4_value_list[0],
                    self.theta5: theta5_value_list[0],
                    self.theta6: theta6_value_list[0],
                    self.theta7: theta7_value_list[0],
                }
            )

            # Check if the determinant is close to 0, if so we need to offset to escape singularity
            det = max(J_Numerical) / min(J_Numerical)
            k = 0
            while abs(det) <= 0.05:
                print('Escaping singularity using joint ', k)
                # try moving each joint by a small offset until we escape the singularity
                if k == 0:
                    theta1_value_list[0] += offset
                elif k == 1:
                    theta2_value_list[0] += offset
                elif k == 2:
                    theta3_value_list[0] += offset
                elif k == 3:
                    theta4_value_list[0] += offset
                elif k == 4:
                    theta5_value_list[0] += offset
                elif k == 5:
                    theta6_value_list[0] += offset
                elif k == 6:
                    theta7_value_list[0] += offset
                elif k == 7:
                    k = -1 # if we are still in a singularity after moving each joint, restart
                k += 1

                # Calculate numerical Jacobian
                J_Numerical = self.J.subs(
                    {
                        self.theta1: theta1_value_list[0],
                        self.theta2: theta2_value_list[0],
                        self.theta3: theta3_value_list[0],
                        self.theta4: theta4_value_list[0],
                        self.theta5: theta5_value_list[0],
                        self.theta6: theta6_value_list[0],
                        self.theta7: theta7_value_list[0],
                    }
                )

                det = max(J_Numerical) / min(J_Numerical)
            
            J_inverse = J_Numerical.pinv()

            # Calculate the new end effector position
            positions_val = self.Tn[5].subs(
                {
                    self.theta1: theta1_value_list[i],
                    self.theta2: theta2_value_list[i],
                    self.theta3: theta3_value_list[i],
                    self.theta4: theta4_value_list[i],
                    self.theta5: theta5_value_list[i],
                    self.theta6: theta6_value_list[i],
                    self.theta7: theta7_value_list[i],
                }
            )
            x_value_list.append(positions_val[0, 3])
            y_value_list.append(positions_val[1, 3])
            z_value_list.append(positions_val[2, 3])

            # Calculate the new joint velocities
            joint_velocities_val = J_inverse @ end_effector_velocities.subs(
                {x_dot_symbol: x_dot[i], y_dot_symbol: y_dot[i], z_dot_symbol: z_dot[i]}
            )

            # Limit the joint velocities
            for j in range(7):
                if abs(joint_velocities_val[j]) > self.maximum_joint_velocity:
                    print("q_dot ", j, " large: ", joint_velocities_val[j])
                    joint_velocities_val[j] *= self.scale
            
            # Save joint velocitie
            self.theta_dot1_value_list.append(float(joint_velocities_val[0]))
            self.theta_dot2_value_list.append(float(joint_velocities_val[1]))
            self.theta_dot3_value_list.append(float(joint_velocities_val[2]))
            self.theta_dot4_value_list.append(float(joint_velocities_val[3]))
            self.theta_dot5_value_list.append(float(joint_velocities_val[4]))
            self.theta_dot6_value_list.append(float(joint_velocities_val[5]))
            self.theta_dot7_value_list.append(float(joint_velocities_val[6]))

            # Calculate the new joint angles after 1 time step
            theta1_value_list.append(theta1_value_list[i] + self.theta_dot1_value_list[i] * self.dt)
            theta2_value_list.append(theta2_value_list[i] + self.theta_dot2_value_list[i] * self.dt)
            theta3_value_list.append(theta3_value_list[i] + self.theta_dot3_value_list[i] * self.dt)
            theta4_value_list.append(theta4_value_list[i] + self.theta_dot4_value_list[i] * self.dt)
            theta5_value_list.append(theta5_value_list[i] + self.theta_dot5_value_list[i] * self.dt)
            theta6_value_list.append(theta6_value_list[i] + self.theta_dot6_value_list[i] * self.dt)
            theta7_value_list.append(theta7_value_list[i] + self.theta_dot7_value_list[i] * self.dt)

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
        plt.plot(time, theta1_value_list[0:len(time)], label="theta1")
        plt.plot(time, theta2_value_list[0:len(time)], label="theta2")
        plt.plot(time, theta3_value_list[0:len(time)], label="theta3")
        plt.plot(time, theta4_value_list[0:len(time)], label="theta4")
        plt.plot(time, theta5_value_list[0:len(time)], label="theta5")
        plt.plot(time, theta6_value_list[0:len(time)], label="theta6")
        plt.plot(time, theta7_value_list[0:len(time)], label="theta7")
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
            joint_velocities.data = [0.0,
                                     0.0,
                                     -pi/2,
                                     -pi/2,
                                     0.0,
                                     0.0,
                                     0.0]

            self.joint_velocities_pub.publish(joint_velocities)

        print("finished approach")

        while i < len(self.theta_dot1_value_list):
            time_current = self.get_clock().now()
            if ((time_current - time_previous) >= rclpy.duration.Duration(seconds=self.dt)):
                time_previous = time_current
                joint_velocities.data = [self.theta_dot1_value_list[i],
                                         self.theta_dot2_value_list[i],
                                         self.theta_dot3_value_list[i],
                                         self.theta_dot4_value_list[i],
                                         self.theta_dot5_value_list[i],
                                         self.theta_dot6_value_list[i],
                                         self.theta_dot7_value_list[i]]
                i += 1
                self.joint_velocities_pub.publish(joint_velocities)

    def home(self):
        joint_positions = Float64MultiArray()
        joint_positions.data = [0.0,
                                 0.0,
                                 0.0,
                                 0.0,
                                 0.0,
                                 0.0,
                                 0.0]

        self.joint_position_pub.publish(joint_positions)

     
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
                elif key == 'h':  # Execute
                    self.get_logger().info('Sending to home...')
                    self.home()
                    self.get_logger().info('Homed!')

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