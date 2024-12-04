#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi


class BoxingNode(Node):
    # Create generalized transformation matrix for DH table
    def dh_transform(self, theta, d, a, alpha):
        return sym.Matrix([
            [sym.cos(theta), -sym.sin(theta)*sym.cos(alpha), sym.sin(theta)*sym.sin(alpha), a*sym.cos(theta)],
            [sym.sin(theta), sym.cos(theta)*sym.cos(alpha), -sym.cos(theta)*sym.sin(alpha), a*sym.sin(theta)],
            [0, sym.sin(alpha), sym.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    # Takes the (x, y, z) of S and returns the cartesian path parameters
    def cartesian_path(self, xi, yi, zi):
        theta1, theta2, theta3, theta4, theta5, theta6, theta7 = sym.symbols('theta1:8')
        t = sym.Symbol("t")
        time = []
        x_dot = []
        y_dot = []
        z_dot = []
        # Virtual joint gain
        kv = .2


        theta1_value_list = [float(np.deg2rad(45))]
        theta2_value_list = [kv*theta1_value_list[0]]
        theta3_value_list = [float(np.deg2rad(45))]
        theta4_value_list = [float(np.deg2rad(45))]
        theta5_value_list = [float(np.deg2rad(0))]
        theta6_value_list = [float(np.deg2rad(0))]
        theta7_value_list = [float(np.deg2rad(0))]

        scale_factor = 1

        T1 = 5 * scale_factor  # time for approach (5s)
        T2 = 4 * scale_factor  # time for semi-circle (4s)
        T3 = 3 * scale_factor  # time for left side of the square (3s)
        T4 = 1 * scale_factor  # time for pause at A (1s)
        T5 = 3 * scale_factor  # time for bottom side of the square (3s)
        T6 = 1 * scale_factor  # time for pause at B (1s)
        T7 = 3 * scale_factor  # time for right side of square (3s)

        # approach section
        t1 = np.linspace(0, T1, T1 * self.update_rate)

        # Calculate the starting point (Home position)
        T_start = self.Tn[5].subs(
            {
                theta1: theta1_value_list[0],
                theta2: theta2_value_list[0],
                theta3: theta3_value_list[0],
                theta4: theta4_value_list[0],
                theta5: theta5_value_list[0],
                theta6: theta6_value_list[0],
                theta7: theta7_value_list[0],
            }
        )

        # Calculate cartesian positions
        dx = (xi - T_start[0,3]) / T1
        dy = (yi - T_start[1,3]) / T1
        dz = (zi - T_start[2,3]) / T1
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
        x2 = 50 - (50 * sym.cos((2 * pi / (T2 * 2)) * t))
        y2 = (0 * t)
        z2 = (50 * sym.sin((2 * pi / (T2 * 2)) * t))

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

        # Left side of the square section
        t3 = np.linspace(0, T3, T3 * self.update_rate)

        # Calculate cartesian positions
        x3 = 0 * t
        y3 = 0 * t
        z3 = -(50 / T3) * t

        # Calculate cartesian velocities
        x3_dot = x3.diff(t)
        y3_dot = y3.diff(t)
        z3_dot = z3.diff(t)

        # Append to path
        for i in range(len(t3)):
            time.append(t3[i] + T1 + T2)
            x_dot.append(x3_dot.subs({t: t3[i]}))
            y_dot.append(y3_dot.subs({t: t3[i]}))
            z_dot.append(z3_dot.subs({t: t3[i]}))

        # Pause at A section
        t4 = np.linspace(0, T4, T4 * self.update_rate)

        # Calculate cartesian positions
        x4 = 0 * t
        y4 = 0 * t
        z4 = 0 * t

        # Calculate cartesian velocities
        x4_dot = x4.diff(t)
        y4_dot = y4.diff(t)
        z4_dot = z4.diff(t)

        # Append to path
        for i in range(len(t4)):
            time.append(t4[i] + T1 + T2 + T3)
            x_dot.append(x4_dot.subs({t: t4[i]}))
            y_dot.append(y4_dot.subs({t: t4[i]}))
            z_dot.append(z4_dot.subs({t: t4[i]}))

        # Bottom side of the square sections
        t5 = np.linspace(0, T5, T5 * self.update_rate)

        # Calculate cartesian positions
        x5 = -(100 / T5) * t
        y5 = 0 * t
        z5 = 0 * t

        # Calculate cartesian velocities
        x5_dot = x5.diff(t)
        y5_dot = y5.diff(t)
        z5_dot = z5.diff(t)

        # Append to path
        for i in range(len(t5)):
            time.append(t5[i] + T1 + T2 + T3 + T4)
            x_dot.append(x5_dot.subs({t: t5[i]}))
            y_dot.append(y5_dot.subs({t: t5[i]}))
            z_dot.append(z5_dot.subs({t: t5[i]}))

        # Pause at B section
        t6 = np.linspace(0, T6, T6 * self.update_rate)

        # Calculate cartesian positions
        x6 = 0 * t
        y6 = 0 * t
        z6 = 0 * t

        # Calculate cartesian velocities
        x6_dot = x6.diff(t)
        y6_dot = y6.diff(t)
        z6_dot = z6.diff(t)

        # Append to path
        for i in range(len(t6)):
            time.append(t6[i] + T1 + T2 + T3 + T4 + T5)
            x_dot.append(x6_dot.subs({t: t6[i]}))
            y_dot.append(y6_dot.subs({t: t6[i]}))
            z_dot.append(z6_dot.subs({t: t6[i]}))

        # Right side of square section
        t7 = np.linspace(0, T7, T7 * self.update_rate)

        # Calculate cartesian positions
        x7 = 0 * t
        y7 = 0 * t
        z7 = (50 / T7) * t

        # Calculate cartesian velocities
        x7_dot = x7.diff(t)
        y7_dot = y7.diff(t)
        z7_dot = z7.diff(t)

        # Append to path
        for i in range(len(t7)):
            time.append(t7[i] + T1 + T2 + T3 + T4 + T5 + T6)
            x_dot.append(x7_dot.subs({t: t7[i]}))
            y_dot.append(y7_dot.subs({t: t7[i]}))
            z_dot.append(z7_dot.subs({t: t7[i]}))

        return time, x_dot, y_dot, z_dot

    def calculate_trajectory(self):
        # DH table for the UR3e documented in the homework
        self.theta1, self.theta2, self.theta3, self.theta4, self.theta5, self.theta6, self.theta7 = sym.symbols('theta1:8')
        #                 theta,    d,          a,      alpha
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

        print("Calculated successive transforms")

        # Calculate cumulative transformations
        self.Tn = [self.An[0]]
        for i in range(6):
            self.Tn.append(self.Tn[i] @ self.An[i+1])

        print("Calculated cumulative transforms")

        # Calculate the Jacobian via the 1st method discussed in lecture
        # Calculate Z and O components
        Z = [sym.Matrix([0, 0, 1])]
        O = [sym.Matrix([0, 0, 0])]
        for T in self.Tn:
            Z.append(sym.Matrix(T[:3, 2]))
            O.append(sym.Matrix(T[:3, 3]))

        print("Calculated O an Z")
        
        # Calculate Jv and Jw
        Jv = sym.zeros(3, 7)
        Jw = sym.zeros(3, 7)
        for i in range(6):
            Jv[:,i] = Z[i].cross(O[7] - O[i])
            Jw[:,i] = Z[i]

        print("Calculated Jv and Jw")

        self.J = sym.Matrix.vstack(Jv, Jw)

        print("Calculated J")
        print(self.J.shape)

        # # Virtual joint gain
        # kv = .2

        # # Create a 6x6 simplified jacobian where theta2 = kv*theta1
        # J_simplified = self.J.copy()
        # J_simplified[:,0] = J_simplified[:,0] + J_simplified[:,1]
        # J_simplified.col_del(1)
        # J_simplified = J_simplified.subs([(self.theta2, kv*self.theta1)])

        # print("Calculated J_simplified")

        # path parameters
        self.update_rate = 10
        self.dt = 1/self.update_rate
        maximum_joint_velocity = (180 / 180) * pi # limit the joint velocity
        scale = 0.1 # amount to scale joint velocities by if they exceed maximum
        offset = (1 / 180) * pi # amount of offset to escape singularity

        # (x, y, z) of S in the path diagram
        x_val = 100
        y_val = 400
        z_val = 800

        #Initial Conditions
        time = np.linspace(0,5,101)
        delta_T = .05

        # Initialize positions, joint angles and velocities lists
        x_value_list = []
        y_value_list = []
        z_value_list = []
        theta1_value_list = [float(np.deg2rad(45))]
        theta2_value_list = [theta1_value_list[0]]
        theta3_value_list = [float(np.deg2rad(45))]
        theta4_value_list = [float(np.deg2rad(45))]
        theta5_value_list = [float(np.deg2rad(0))]
        theta6_value_list = [float(np.deg2rad(0))]
        theta7_value_list = [float(np.deg2rad(0))]
        # theta1_value_list = [0]
        # theta2_value_list = [0]
        # theta3_value_list = [0]
        # theta4_value_list = [0]
        # theta5_value_list = [0]
        # theta6_value_list = [0]
        # theta7_value_list = [0]
        theta_dot1_value_list = []
        theta_dot2_value_list = []
        theta_dot3_value_list = []
        theta_dot4_value_list = []
        theta_dot5_value_list = []
        theta_dot6_value_list = []
        theta_dot7_value_list = []

        # Calculate cartesian path parameters
        time, x_dot, y_dot, z_dot = self.cartesian_path(x_val, y_val, z_val)

        print("Calculated cartesian path")

        # Calculate the forward velocity kinematics
        theta_dot1, theta_dot2, theta_dot3, theta_dot4, theta_dot5, theta_dot6, theta_dot7  = sym.symbols("theta_dot1:8")
        
        print("Calculated cartesian path")

        # Calculate end effector velocities
        forward_velocity_kinematics = self.J @ sym.Matrix(
            [[theta_dot1], [theta_dot2], [theta_dot3], [theta_dot4], [theta_dot5], [theta_dot6], [theta_dot7]]
        )

        print("Calculated forward velocity kinematics")

        # Calculate end effector velocities
        x_dot_symbol = sym.Symbol("x_dot_symbol")
        y_dot_symbol = sym.Symbol("y_dot_symbol")
        z_dot_symbol = sym.Symbol("z_dot_symbol")
        end_effector_velocities = sym.Matrix(
            [[x_dot_symbol], [y_dot_symbol], [z_dot_symbol], [0], [0], [0]]
        )

        print("Escaping home singularity")
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

        print("Calculated J_Numerical")
        # Check if the determinant is close to 0, if so we need to offset to escape singularity
        det = max(J_Numerical) / min(J_Numerical)
        k = 0
        while abs(det) <= 0.05:
            sym.pprint(J_Numerical)
            print("Determinant: ", det)
            print(theta1_value_list[0])
            print(theta2_value_list[0])
            print(theta3_value_list[0])
            print(theta4_value_list[0])
            print(theta5_value_list[0])
            print(theta6_value_list[0])
            print(theta7_value_list[0])
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

        print("Starting calculation")
        # Loop through toolpath to get inverse velocity kinematics
        for i in range(len(time)):
            if i % self.update_rate == 0:
                print(i, " out of ", len(time), " timestamps calculated")

            J_Numerical = self.J.subs(
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
                if abs(joint_velocities_val[j]) > maximum_joint_velocity:
                    print("q_dot ", j, " large: ", joint_velocities_val[j])
                    joint_velocities_val[j] *= scale
            
            # Save joint velocitie
            theta_dot1_value_list.append(joint_velocities_val[0])
            theta_dot2_value_list.append(joint_velocities_val[1])
            theta_dot3_value_list.append(joint_velocities_val[2])
            theta_dot4_value_list.append(joint_velocities_val[3])
            theta_dot5_value_list.append(joint_velocities_val[4])
            theta_dot6_value_list.append(joint_velocities_val[5])
            theta_dot7_value_list.append(joint_velocities_val[6])

            # Calculate the new joint angles after 1 time step
            theta1_value_list.append(theta1_value_list[i] + theta_dot1_value_list[i] * self.dt)
            theta2_value_list.append(theta2_value_list[i] + theta_dot2_value_list[i] * self.dt)
            theta3_value_list.append(theta3_value_list[i] + theta_dot3_value_list[i] * self.dt)
            theta4_value_list.append(theta4_value_list[i] + theta_dot4_value_list[i] * self.dt)
            theta5_value_list.append(theta5_value_list[i] + theta_dot5_value_list[i] * self.dt)
            theta6_value_list.append(theta6_value_list[i] + theta_dot6_value_list[i] * self.dt)
            theta7_value_list.append(theta7_value_list[i] + theta_dot7_value_list[i] * self.dt)

        # Plot joint velocities
        plt.figure("Joint Velocities")
        plt.plot(time, theta_dot1_value_list, label="theta_dot1")
        plt.plot(time, theta_dot2_value_list, label="theta_dot2")
        plt.plot(time, theta_dot3_value_list, label="theta_dot3")
        plt.plot(time, theta_dot4_value_list, label="theta_dot4")
        plt.plot(time, theta_dot5_value_list, label="theta_dot5")
        plt.plot(time, theta_dot6_value_list, label="theta_dot6")
        plt.plot(time, theta_dot7_value_list, label="theta_dot7")
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

    def __init__(self):
        super().__init__('boxing_node')

        # Publish to the position and velocity controller topics
        self.joint_velocities_pub = self.create_publisher(Float64MultiArray, '/velocity_controller/commands', 10)

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
    node.calculate_trajectory()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()