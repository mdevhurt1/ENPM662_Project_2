#!/usr/bin/env python3

import sys
import select
import tty
import termios

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

from roboticstoolbox import DHRobot, RevoluteDH

##########################################
# Helper functions for plotting and geometry
##########################################
def set_axes_equal(ax):
    """
    Adjust the 3D plot axes so that they have equal scale, ensuring
    that objects (like trajectories) appear proportionally correct.
    """
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

def plot_joint_velocities(time, joint_velocity_lists):
    """
    Plot all seven joint velocities over time.
    """
    plt.figure("Joint Velocities")
    for j in range(7):
        plt.plot(time, joint_velocity_lists[j], label=f"theta_dot{j+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Velocities (rad/s)")
    plt.title("Joint Velocities over Time")
    plt.legend()

def plot_joint_angles(time, joint_angle_lists):
    """
    Plot all seven joint angles over time.
    """
    plt.figure("Joint Angles")
    for j in range(7):
        length = min(len(time), len(joint_angle_lists[j]))
        plt.plot(time[:length], joint_angle_lists[j][:length], label=f"theta{j+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Angles (rad)")
    plt.title("Joint Angles over Time")
    plt.legend()

def plot_end_effector_positions(x_values, y_values, z_values, x_values_check, y_values_check, z_values_check, label="End Effector Path"):
    """
    Plot the 3D trajectory of the end-effector.
    """
    fig = plt.figure("End Effector Positions")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot3D(x_values, y_values, z_values, color="blue", label=label + " actual")
    ax.plot3D(x_values_check, y_values_check, z_values_check, color="red", label=label + " desired")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.legend()
    return ax


##########################################
# Boxing Node
##########################################
class BoxingNode(Node):
    def __init__(self):
        super().__init__('boxing_node')

        self.get_logger().info("Initializing Boxing Node")

        # Publisher for joint velocities
        self.joint_velocities_pub = self.create_publisher(Float64MultiArray, '/velocity_controller/commands', 10)

        # Store original terminal settings to restore after reading keyboard input
        self.settings = termios.tcgetattr(sys.stdin)

        # Robot and simulation parameters
        self.update_rate = 100
        self.dt = 1 / self.update_rate
        self.maximum_joint_velocity = 4 * pi   # Max allowed joint velocity
        self.scale = 0.01                      # Scale factor for velocities if exceeded
        self.offset = (1/180)*pi               # Small offset for resolving singularities
        self.time_per_punch_in_seconds = 1
        self.radius = 150                      # Radius used in punch trajectory calculations

        # Define symbolic variables for joint angles
        self.theta_symbols = sym.symbols('theta1 theta2 theta3 theta4 theta5 theta6 theta7', real=True)
        (self.theta1, self.theta2, self.theta3, 
         self.theta4, self.theta5, self.theta6, 
         self.theta7) = self.theta_symbols

        # Denavit-Hartenberg parameters for the robot
        self.DH = sym.Matrix([
            [self.theta1, 165.1,   0,    pi/2],
            [self.theta2, 0,       0,   -pi/2],
            [self.theta3, 255.03,  0,    pi/2],
            [self.theta4, 0,       0,   -pi/2],
            [self.theta5, 427.46,  0,    pi/2],
            [self.theta6, 0,       0,   -pi/2],
            [self.theta7, 0,      -85,   0]
        ])

        # Compute forward kinematics transformation matrices for each link
        self.An = [self.dh_transform(*self.DH[i, :]) for i in range(7)]
        self.Tn = [self.An[0]]
        for i in range(6):
            self.Tn.append(self.Tn[i] @ self.An[i+1])

        # Compute the full symbolic Jacobian
        self.J = self.compute_symbolic_jacobian()

        self.J = self.J

        # Create a numerical evaluation function for the Jacobian
        self.jacobian_func = sym.lambdify(self.theta_symbols, self.J, "numpy")

        # Now that Tn is defined, we can initialize or reset joint data safely
        self.reset_joint_data()

    def reset_joint_data(self):
        """
        Reset the joint angle, velocity, and end-effector position data to initial conditions.
        """
        # Initial joint angles
        self.joint_angle_lists = [
            [0],                # theta1
            [0],                # theta2
            [0],                # theta3
            [-(120/180)*pi],    # theta4
            [0],                # theta5
            [pi/4],             # theta6
            [0.0]               # theta7
        ]

        # Start joint velocities at zero
        self.joint_velocity_lists = [[0.0] for _ in range(7)]

        # Compute the initial end-effector position based on initial angles
        initial_thetas = [lst[0] for lst in self.joint_angle_lists]
        positions_val = self.Tn[6].subs({
            self.theta1: initial_thetas[0],
            self.theta2: initial_thetas[1],
            self.theta3: initial_thetas[2],
            self.theta4: initial_thetas[3],
            self.theta5: initial_thetas[4],
            self.theta6: initial_thetas[5],
            self.theta7: initial_thetas[6]
        })

        # Store end-effector trajectory
        self.x_vals = [float(positions_val[0,3])]
        self.y_vals = [float(positions_val[1,3])]
        self.z_vals = [float(positions_val[2,3])]
        self.x_vals_check = [float(positions_val[0,3])]
        self.y_vals_check = [float(positions_val[1,3])]
        self.z_vals_check = [float(positions_val[2,3])]

    def publish_joint_velocities(self, velocities):
        """
        Publish a set of joint velocities to the robot velocity controller.
        """
        msg = Float64MultiArray()
        msg.data = velocities
        self.joint_velocities_pub.publish(msg)

    def get_current_thetas_from_last(self):
        """
        Retrieve the latest joint angles from the recorded lists.
        """
        return np.array([joint_list[-1] for joint_list in self.joint_angle_lists])

    def getKey(self):
        """
        Capture a single character from stdin without blocking for too long.
        """
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        key = sys.stdin.read(1) if rlist else ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def dh_transform(self, theta, d, a, alpha):
        """
        Compute a single Denavit-Hartenberg transformation matrix for given parameters.
        """
        return sym.Matrix([
            [sym.cos(theta), -sym.sin(theta)*sym.cos(alpha),  sym.sin(theta)*sym.sin(alpha), a*sym.cos(theta)],
            [sym.sin(theta),  sym.cos(theta)*sym.cos(alpha), -sym.cos(theta)*sym.sin(alpha), a*sym.sin(theta)],
            [0,               sym.sin(alpha),                 sym.cos(alpha),                d],
            [0,               0,                              0,                             1]
        ])

    def compute_symbolic_jacobian(self):
        """
        Compute the full symbolic Jacobian for the end-effector, combining both linear and angular parts.
        """
        Z = [sym.Matrix([0, 0, 1])]
        O = [sym.Matrix([0, 0, 0])]
        for T in self.Tn:
            Z.append(sym.Matrix(T[:3, 2]))
            O.append(sym.Matrix(T[:3, 3]))

        Jv = sym.zeros(3, 7)
        Jw = sym.zeros(3, 7)

        # Each column of Jv and Jw is derived from the position and orientation of each joint
        for i in range(7):
            Jv[:, i] = Z[i].cross(O[7] - O[i])  # Linear velocity part
            Jw[:, i] = Z[i]                     # Angular velocity part

        return sym.Matrix.vstack(Jv, Jw)

    def substitute_jacobian(self, current_thetas):
        """
        Evaluate the numeric Jacobian at the given joint angles.
        """
        return self.jacobian_func(*current_thetas)

    def get_pattern_trajectory(self, pattern_index):
        """
        Given a pattern index, return the desired Cartesian velocity profiles (x_dot, y_dot, z_dot) over time.
        Patterns are defined symbolically using sinusoidal functions.
        """
        t = sym.Symbol("t")
        # Define multiple punching trajectories
        patterns = [
            # 0: Jab
            (-(self.radius*sym.sin((2*pi/self.time_per_punch_in_seconds)*t)),
             0*t,
             (self.radius*sym.cos((2*pi/self.time_per_punch_in_seconds)*t))),
            
            # 1: Hook
            (-(self.radius*sym.sin((2*pi/self.time_per_punch_in_seconds)*t)),
             -(self.radius * sym.cos((2*pi/self.time_per_punch_in_seconds)*t)),
             -(self.radius/2 * sym.cos((2*pi/self.time_per_punch_in_seconds)*t))),

            # 2: Upper Cut
            (-(self.radius*sym.sin((2*pi/self.time_per_punch_in_seconds)*t)),
             (self.radius/10 * sym.cos((2*pi/self.time_per_punch_in_seconds)*t)),
             -(self.radius*sym.cos((2*pi/self.time_per_punch_in_seconds)*t))),

            # 3-9: Future patterns (currently placeholders)
            (0*t,0*t,0*t),
            (0*t,0*t,0*t),
            (0*t,0*t,0*t),
            (0*t,0*t,0*t),
            (0*t,0*t,0*t),
            (0*t,0*t,0*t),
            (0*t,0*t,0*t),
        ]

        x_expr, y_expr, z_expr = patterns[pattern_index]

        # Discretize time for the given pattern
        punch_timestamps = np.linspace(0, self.time_per_punch_in_seconds, self.time_per_punch_in_seconds * self.update_rate)

        # Compute derivatives wrt time to get x_dot, y_dot, z_dot
        x_dot_expr = sym.diff(x_expr, t)
        y_dot_expr = sym.diff(y_expr, t)
        z_dot_expr = sym.diff(z_expr, t)

        x_dot_vals = [x_dot_expr.subs({t: ti}) for ti in punch_timestamps]
        y_dot_vals = [y_dot_expr.subs({t: ti}) for ti in punch_timestamps]
        z_dot_vals = [z_dot_expr.subs({t: ti}) for ti in punch_timestamps]

        return punch_timestamps, x_dot_vals, y_dot_vals, z_dot_vals

    def scale_if_exceeded(self, joint_vels):
        """
        Scale the joint velocities if any exceed the maximum allowed velocity.
        """
        if any(abs(jv) > self.maximum_joint_velocity for jv in joint_vels):
            return self.scale
        return 1.0

    def resolve_singularity(self, J_numeric, current_thetas):
        """
        Attempt to resolve singularities by slightly adjusting some joint angles.
        If the Jacobian condition measure is too low, increment angles by a small offset.
        """
        try:
            condition_measure = np.max(J_numeric) / np.min(J_numeric)
        except ZeroDivisionError:
            condition_measure = 0

        k = 0
        while abs(condition_measure) <= 0.05 and k < 7:
            self.joint_angle_lists[k][-1] += self.offset
            current_thetas = self.get_current_thetas_from_last()
            J_numeric = self.substitute_jacobian(current_thetas)
            try:
                condition_measure = np.max(J_numeric) / np.min(J_numeric)
            except ZeroDivisionError:
                condition_measure = 0
            k += 1

        return J_numeric

    def plan_actual_pattern(self, pattern_index):
        """
        Plan a pattern incrementally using the actual Jacobian-based inverse kinematics approach.
        Each step calculates joint velocities to follow the desired Cartesian trajectory.
        """
        self.get_logger().info(f"Planning Pattern {pattern_index}...")
        time_vals, x_dot_vals, y_dot_vals, z_dot_vals = self.get_pattern_trajectory(pattern_index)
        
        current_thetas = self.get_current_thetas_from_last()
        total_steps = len(time_vals)
        last_val_check_position = len(self.x_vals_check) - 1

        for i in range(total_steps):
            if (i % (self.update_rate / 10)) == 0 :
                self.get_logger().info(f"{i / self.update_rate} / {total_steps / self.update_rate} seconds planned!")
            # Compute Jacobian at current configuration
            J_numeric = self.substitute_jacobian(current_thetas)
            J_numeric = self.resolve_singularity(J_numeric, current_thetas)

            # Desired end-effector velocities
            effector_velocity = np.array([x_dot_vals[i], y_dot_vals[i], z_dot_vals[i], 0, 0, 0], dtype=float)
            self.x_vals_check.append(float(self.x_vals_check[last_val_check_position+i] + x_dot_vals[i]*self.dt))
            self.y_vals_check.append(float(self.y_vals_check[last_val_check_position+i] + y_dot_vals[i]*self.dt))
            self.z_vals_check.append(float(self.z_vals_check[last_val_check_position+i] + z_dot_vals[i]*self.dt))

            # Compute joint velocities via the pseudoinverse of the Jacobian
            current_joint_velocities = np.linalg.pinv(J_numeric) @ effector_velocity

            # Limit and scale if necessary
            current_joint_velocities = np.clip(current_joint_velocities, -self.maximum_joint_velocity, self.maximum_joint_velocity)
            current_joint_velocities *= self.scale_if_exceeded(current_joint_velocities)

            # Integrate joint velocities to get new joint angles
            new_thetas = current_thetas + current_joint_velocities * self.dt

            # Record the new joint angles and velocities
            for j in range(7):
                self.joint_angle_lists[j].append(float(new_thetas[j]))
                self.joint_velocity_lists[j].append(float(current_joint_velocities[j]))

            # Compute and store new end-effector position
            positions_val = self.Tn[6].subs({
                self.theta1: new_thetas[0],
                self.theta2: new_thetas[1],
                self.theta3: new_thetas[2],
                self.theta4: new_thetas[3],
                self.theta5: new_thetas[4],
                self.theta6: new_thetas[5],
                self.theta7: new_thetas[6]
            })
            self.x_vals.append(float(positions_val[0,3]))
            self.y_vals.append(float(positions_val[1,3]))
            self.z_vals.append(float(positions_val[2,3]))

            current_thetas = new_thetas

        self.get_logger().info(f"Pattern {pattern_index} planned!")

    def move_to_home(self):
        """
        Move the robot back to its initial "home" position by publishing
        the initial joint angles as velocities over a short duration.
        """
        joint_velocities = Float64MultiArray()
        time_previous = self.get_clock().now()
        time_current = time_previous
        while (time_current - time_previous) <= rclpy.duration.Duration(seconds=1.0):
            time_current = self.get_clock().now()
            # Publish initial angles as velocities just to ensure stable home approach
            joint_velocities.data = [float(lst[0]) for lst in self.joint_angle_lists]
            self.joint_velocities_pub.publish(joint_velocities)
        self.get_logger().info("Finished approach to home")

    def execute_joint_velocities(self):
        """
        Execute the planned joint velocities in real-time. This function
        steps through each recorded velocity, publishing them at the defined rate.
        """
        if not self.joint_velocity_lists or len(self.joint_velocity_lists[0]) <= 1:
            self.get_logger().warning("No joint velocities recorded. Please plan a trajectory first.")
            return

        num_steps = len(self.joint_velocity_lists[0])
        time_previous = self.get_clock().now()
        i = 0
        self.get_logger().info("Executing Joint Velocities...")

        while i < num_steps:
            time_current = self.get_clock().now()
            if (time_current - time_previous) >= rclpy.duration.Duration(seconds=self.dt):
                time_previous = time_current
                current_velocities = [self.joint_velocity_lists[j][i] for j in range(7)]
                self.publish_joint_velocities(current_velocities)
                i += 1

        self.get_logger().info("Execution Complete!")

    def plot_full_trajectory(self):
        """
        Plot the entire recorded trajectory including joint angles, velocities,
        and end-effector positions.
        """
        n_steps = len(self.joint_angle_lists[0])
        if n_steps < 2:
            self.get_logger().warning("Not enough data to plot. Please plan a trajectory first.")
            return

        # Create a time vector based on number of steps and dt
        time = np.linspace(0, (n_steps - 1)*self.dt, n_steps)

        # Plot all recorded data
        plot_joint_velocities(time, self.joint_velocity_lists)
        plot_joint_angles(time, self.joint_angle_lists)
        ax = plot_end_effector_positions(self.x_vals, self.y_vals, self.z_vals, self.x_vals_check, self.y_vals_check, self.z_vals_check, label="Planned Path")
        set_axes_equal(ax)
        plt.show()

    def handle_user_input(self, key):
        """
        Handle user keyboard input:
        - Number keys plan different punching patterns.
        - 'p' plots the currently planned path.
        - 'e' moves the robot to home.
        - 'a' executes the planned joint velocities.
        - 'c' clears all planned data.
        - ESC quits.
        """
        if key == '\x1b':  # Escape key
            return False
        elif key in ['1','2','3','4','5','6','7','8','9']:
            # Plan the pattern corresponding to the pressed number
            pattern_index = int(key) - 1
            self.plan_actual_pattern(pattern_index)
        elif key == 'p':
            self.get_logger().info('Plotting...')
            self.plot_full_trajectory()
            self.get_logger().info('Finished...')
        elif key == 'e':
            self.get_logger().info('Moving to Home...')
            self.move_to_home()
            self.get_logger().info('Movement Complete!')
        elif key == 'a':
            self.get_logger().info('Executing Joint Velocities...')
            self.execute_joint_velocities()
            self.get_logger().info('Execution Complete!')
        elif key == 'c':
            self.get_logger().info('Clearing joint angles, velocities, and stored path data...')
            self.reset_joint_data()
            self.get_logger().info('Data cleared. Ready for a new trajectory.')

        # After any input, publish zero velocities as a default
        self.publish_joint_velocities([0.0]*7)
        return True

    def run_control(self):
        """
        Main loop for user interaction. Provides instructions and waits for user input.
        """
        msg = """
        Let's Go Boxing (Incremental Planning)!
        --------------------------------------
        1 : plan Jab
        2 : plan Hook
        3 : plan Upper Cut
        4-9: Future patterns
        p : plot the planned path
        e : move to home
        a : execute joint velocities
        c : clear recorded joint data and stored path
        Esc: quit
        """
        self.get_logger().info(msg)

        # Publish zero velocities to ensure robot stays still initially
        self.publish_joint_velocities([0.0]*7)

        # Continuously read keyboard inputs and handle them
        while True:
            key = self.getKey()
            if not self.handle_user_input(key):
                break

##########################################
# Main Entry Point
##########################################
def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)
    node = BoxingNode()
    node.run_control()  # Start the interactive control loop
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
