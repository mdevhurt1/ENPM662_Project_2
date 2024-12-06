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

        ###############
        # world x -> robot y
        # world y -> robot z
        # world z -> robot x
        ###############
        self.maximum_joint_velocity = 2*pi
        self.scale = 0.01
        self.offset = (1/180) * pi
        self.time_per_punch_in_seconds = 3
        self.radius = 150

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

    # Set equal aspect ratio
    def set_axes_equal(self, ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])

        max_range = max(x_range, y_range, z_range)

        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)

        ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
        ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
        ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

    def calculate_robot_toolbox(self):
        """
        Calculate joint angles and velocities for a 7-DOF robot arm, 
        and plot joint velocities and end effector positions over time.
        """
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

        # Visualize the robot with initial joint angles
        initial_angles = [self.theta1_value_list[0], self.theta2_value_list[0],
                        self.theta3_value_list[0], self.theta4_value_list[0],
                        self.theta5_value_list[0], self.theta6_value_list[0],
                        self.theta7_value_list[0]]
        robot.plot(initial_angles, block=True)

        # Get Cartesian path parameters
        time, x_dot, y_dot, z_dot = self.cartesian_path()

        # Initialize position lists
        x_value_list, y_value_list, z_value_list = [], [], []
        print("Calculating...")

        for i, t in enumerate(time):
            # Current joint angles
            thetas = np.array([self.theta1_value_list[i], self.theta2_value_list[i],
                            self.theta3_value_list[i], self.theta4_value_list[i],
                            self.theta5_value_list[i], self.theta6_value_list[i],
                            self.theta7_value_list[i]])

            # Compute Jacobian
            J = robot.jacob0(thetas)
            effector = np.array([x_dot[i], y_dot[i], z_dot[i], 0, 0, 0])
            
            # Ensure J and effector are numeric
            J_numeric = np.array(J, dtype=np.float64)
            effector_numeric = np.array(effector, dtype=np.float64)

            # Perform the pseudo-inverse operation
            try:
                theta_dots = np.linalg.pinv(J_numeric).dot(effector_numeric)
            except np.linalg.LinAlgError as e:
                print(f"Pseudo-inverse failed at time {t} with error: {e}")
                return
            except TypeError as e:
                print(f"Type conversion failed for Jacobian or effector at time {t} with error: {e}")
                return

            # Update joint angles and store velocities
            for j in range(7):
                theta_next = thetas[j] + theta_dots[j] * self.dt
                getattr(self, f"theta{j+1}_value_list").append(theta_next)
                getattr(self, f"theta_dot{j+1}_value_list").append(theta_dots[j])

            # Compute forward kinematics
            T = robot.fkine(thetas + theta_dots * self.dt)
            translation = T.t
            x_value_list.append(translation[0])
            y_value_list.append(translation[1])
            z_value_list.append(translation[2])

        # Plot joint velocities
        plt.figure("Joint Velocities")
        for j in range(7):
            plt.plot(time, getattr(self, f"theta_dot{j+1}_value_list"), label=f"theta_dot{j+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Joint Velocities (rad/s)")
        plt.title("Joint Velocities over Time")
        plt.legend()

        # Plot end effector positions
        fig = plt.figure("End Effector Positions")
        ax = fig.add_subplot(111, projection="3d")
        ax.plot3D(x_value_list, y_value_list, z_value_list, color="blue", label="End Effector Path")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        self.set_axes_equal(ax)
        ax.legend()
        plt.show()

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

    def strike(self, x, y, z, punch_number):
        t = sym.Symbol("t")
        time = []
        x_dot = []
        y_dot = []
        z_dot = []

        t1 = np.linspace(0, self.time_per_punch_in_seconds, self.time_per_punch_in_seconds * self.update_rate)

        # Calculate cartesian velocities
        x1_dot = x.diff(t)
        y1_dot = y.diff(t)
        z1_dot = z.diff(t)

        # Append to path
        for i in range(len(t1)):
            time.append(t1[i] + self.time_per_punch_in_seconds*punch_number)
            x_dot.append(x1_dot.subs({t: t1[i]}))
            y_dot.append(y1_dot.subs({t: t1[i]}))
            z_dot.append(z1_dot.subs({t: t1[i]}))

        return time, x_dot, y_dot, z_dot

    # Takes the (x, y, z) of S and returns the cartesian path parameters
    def cartesian_path(self):
        t = sym.Symbol("t")
        time = []
        x_dot = []
        y_dot = []
        z_dot = []
        temp_time = []
        temp_x_dot = []
        temp_y_dot = []
        temp_z_dot = []

        # # Upper Cut
        # x = -(self.radius * sym.sin((2 * pi / (self.time_per_punch_in_seconds)) * t))
        # y = (self.radius/10 * sym.cos((2 * pi / (self.time_per_punch_in_seconds)) * t))
        # z = -(self.radius * sym.cos((2 * pi / (self.time_per_punch_in_seconds)) * t))

        # temp_time, temp_x_dot, temp_y_dot, temp_z_dot = self.strike(x, y, z, 0)
        # for i in range(len(temp_time)):
        #     time.append(temp_time[i])
        #     x_dot.append(temp_x_dot[i])
        #     y_dot.append(temp_y_dot[i])
        #     z_dot.append(temp_z_dot[i])

        # # Hook
        # x = (self.radius * sym.sin((2 * pi / (self.time_per_punch_in_seconds)) * t))
        # y = (self.radius/4 * sym.cos((2 * pi / (self.time_per_punch_in_seconds)) * t))
        # z = -(self.radius/10 * sym.cos((2 * pi / (self.time_per_punch_in_seconds)) * t))

        # temp_time, temp_x_dot, temp_y_dot, temp_z_dot = self.strike(x, y, z, 1)
        # for i in range(len(temp_time)):
        #     time.append(temp_time[i])
        #     x_dot.append(temp_x_dot[i])
        #     y_dot.append(temp_y_dot[i])
        #     z_dot.append(temp_z_dot[i])
        
        # # Jab
        # x = -(self.radius * sym.cos((2 * pi / (self.time_per_punch_in_seconds)) * t))
        # y = 0*t
        # z = (self.radius/2 * sym.cos((2 * pi / (self.time_per_punch_in_seconds)) * t))

        # temp_time, temp_x_dot, temp_y_dot, temp_z_dot = self.strike(x, y, z, 2)
        # for i in range(len(temp_time)):
        #     time.append(temp_time[i])
        #     x_dot.append(temp_x_dot[i])
        #     y_dot.append(temp_y_dot[i])
        #     z_dot.append(temp_z_dot[i])
        
        # omega = 2*pi / self.time_per_punch_in_seconds

        # # Jab
        # x = -(self.radius * sym.cos(omega * t))
        # y = 0*t
        # z = 0*t

        # temp_time, temp_x_dot, temp_y_dot, temp_z_dot = self.strike(x, y, z, 0)
        # for i in range(len(temp_time)):
        #     time.append(temp_time[i])
        #     x_dot.append(temp_x_dot[i])
        #     y_dot.append(temp_y_dot[i])
        #     z_dot.append(temp_z_dot[i])

        x = -(self.radius * sym.sin((2 * pi / self.time_per_punch_in_seconds) * t))
        y = (0 * t)
        z = (self.radius * sym.cos((2 * pi / self.time_per_punch_in_seconds) * t))

        temp_time, temp_x_dot, temp_y_dot, temp_z_dot = self.strike(x, y, z, 0)
        for i in range(len(temp_time)):
            time.append(temp_time[i])
            x_dot.append(temp_x_dot[i])
            y_dot.append(temp_y_dot[i])
            z_dot.append(temp_z_dot[i])

        return time, x_dot, y_dot, z_dot

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
                    self.theta1: self.theta1_value_list[i],
                    self.theta2: self.theta2_value_list[i],
                    self.theta3: self.theta3_value_list[i],
                    self.theta4: self.theta4_value_list[i],
                    self.theta5: self.theta5_value_list[i],
                    self.theta6: self.theta6_value_list[i],
                    self.theta7: self.theta7_value_list[i],
                }
            )

            # Check if the determinant is close to 0, if so we need to offset to escape singularity
            det = max(J_Numerical) / min(J_Numerical)
            k = 0
            while abs(det) <= 0.05:
                print("Close to singularity")
                # try moving each joint by a small offset until we escape the singularity
                if k == 0:
                    theta1_value_list[0] += self.offset
                elif k == 1:
                    theta2_value_list[0] += self.offset
                elif k == 2:
                    theta3_value_list[0] += self.offset
                elif k == 3:
                    theta4_value_list[0] += self.offset
                elif k == 4:
                    theta5_value_list[0] += self.offset
                elif k == 5:
                    theta6_value_list[0] += self.offset
                elif k == 6:
                    k = -1 # if we are still in a singularity after moving each joint, restart
                k += 1

                # Calculate numerical Jacobian
                J_Numerical = J.subs(
                    {
                        theta1: theta1_value_list[i],
                        theta2: theta2_value_list[i],
                        theta3: theta3_value_list[i],
                        theta4: theta4_value_list[i],
                        theta5: theta5_value_list[i],
                        theta6: theta6_value_list[i],
                    }
                )

                det = max(J_Numerical) / min(J_Numerical)

            # J_inverse = np.linalg.pinv(J_Numerical)
            J_inverse = J_Numerical.pinv()

            # Calculate the new end effector position
            positions_val = self.Tn[6].subs(
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


            # Limit the joint velocities
            for j in range(6):
                if abs(joint_velocities_val[j]) > self.maximum_joint_velocity:
                    print("q_dot ", j, " large: ", joint_velocities_val[j])
                    joint_velocities_val[j] *= self.scale
            
            # Save joint velocitie
            self.theta_dot1_value_list.append(joint_velocities_val[0])
            self.theta_dot2_value_list.append(joint_velocities_val[1])
            self.theta_dot3_value_list.append(joint_velocities_val[2])
            self.theta_dot4_value_list.append(joint_velocities_val[3])
            self.theta_dot5_value_list.append(joint_velocities_val[4])
            self.theta_dot6_value_list.append(joint_velocities_val[5])
            self.theta_dot7_value_list.append(joint_velocities_val[6])

            # Calculate the new joint angles after 1 time step
            self.theta1_value_list.append(self.theta1_value_list[i] + (self.theta_dot1_value_list[i] * self.dt))
            self.theta2_value_list.append(self.theta2_value_list[i] + (self.theta_dot2_value_list[i] * self.dt))
            self.theta3_value_list.append(self.theta3_value_list[i] + (self.theta_dot3_value_list[i] * self.dt))
            self.theta4_value_list.append(self.theta4_value_list[i] + (self.theta_dot4_value_list[i] * self.dt))
            self.theta5_value_list.append(self.theta5_value_list[i] + (self.theta_dot5_value_list[i] * self.dt))
            self.theta6_value_list.append(self.theta6_value_list[i] + (self.theta_dot6_value_list[i] * self.dt))
            self.theta7_value_list.append(self.theta7_value_list[i] + (self.theta_dot7_value_list[i] * self.dt))

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
        self.set_axes_equal(ax1)

        plt.show()

        return True

    def prepare_arm(self):
        joint_velocities = Float64MultiArray()
        time_previous = self.get_clock().now()
        time_current = time_previous
        print(time_current-time_previous)
        print(rclpy.duration.Duration(seconds=1.0))
        print((time_current - time_previous) >= (rclpy.duration.Duration(seconds=1.0)))
        while ((time_current - time_previous) <= (rclpy.duration.Duration(seconds=1.0))) :
            time_current = self.get_clock().now()
            joint_velocities.data = [float(self.theta1_value_list[0]),
                                     float(self.theta2_value_list[0]),
                                     float(self.theta3_value_list[0]),
                                     float(self.theta4_value_list[0]),
                                     float(self.theta5_value_list[0]),
                                     float(self.theta6_value_list[0]),
                                     float(self.theta7_value_list[0])]

            self.joint_velocities_pub.publish(joint_velocities)

        print("finished approach")
    
    def send_joint_velocities(self):
        joint_velocities = Float64MultiArray()
        
        time_previous = self.get_clock().now()
        i = 0
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
                    self.prepare_arm()
                    self.get_logger().info('Trajectory Executed!')
                elif key == 'a':  # Execute again
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