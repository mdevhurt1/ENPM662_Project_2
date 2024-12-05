import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from roboticstoolbox import DHRobot, RevoluteDH

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

print(robot)

# #Define Joint Angles
# theta1 = sp.Symbol('th1')
# theta2 = sp.Symbol('th2') # +90 degrees to put in home position
# theta3 = sp.Symbol('th3')
# theta4 = sp.Symbol('th4')
# theta5 = sp.Symbol('th5')
# theta6 = sp.Symbol('th6') # -90 degrees to put in home position
# theta7 = sp.Symbol('th7')

# # DH Parameters
# a1 = 0
# a2 = 0
# a3 = 0
# a4 = 0
# a5 = 0
# a6 = 0
# a7 = -.085 # Negative because it goes in negative x7 direction.
# alpha1 = sp.pi/2
# alpha2 = -sp.pi/2
# alpha3 = sp.pi/2
# alpha4 = -sp.pi/2
# alpha5 = sp.pi/2
# alpha6 = -sp.pi/2
# alpha7 = 0
# d1 = .1651
# d2 = 0
# d3 = .25503
# d4 = 0
# d5 = .42746
# d6 = 0
# d7 = 0

# t_01 = sp.Matrix([[sp.cos(theta1), -sp.sin(theta1)*sp.cos(alpha1), sp.sin(theta1)*sp.sin(alpha1), a1*sp.cos(theta1)],
#                  [sp.sin(theta1), sp.cos(theta1)*sp.cos(alpha1), -sp.cos(theta1)*sp.sin(alpha1), a1*sp.sin(theta1)],
#                  [0, sp.sin(alpha1), sp.cos(alpha1), d1],
#                  [0, 0, 0, 1]])

# t_12 = sp.Matrix([[sp.cos(theta2 + sp.pi/2), -sp.sin(theta2 + sp.pi/2)*sp.cos(alpha2), sp.sin(theta2 + sp.pi/2)*sp.sin(alpha2), a2*sp.cos(theta2 + sp.pi/2)],
#                  [sp.sin(theta2 + sp.pi/2), sp.cos(theta2 + sp.pi/2)*sp.cos(alpha2), -sp.cos(theta2 + sp.pi/2)*sp.sin(alpha2), a2*sp.sin(theta2 + sp.pi/2)],
#                  [0, sp.sin(alpha2), sp.cos(alpha2), d2],
#                  [0, 0, 0, 1]])

# t_23 = sp.Matrix([[sp.cos(theta3), -sp.sin(theta3)*sp.cos(alpha3), sp.sin(theta3)*sp.sin(alpha3), a3*sp.cos(theta3)],
#                  [sp.sin(theta3), sp.cos(theta3)*sp.cos(alpha3), -sp.cos(theta3)*sp.sin(alpha3), a3*sp.sin(theta3)],
#                  [0, sp.sin(alpha3), sp.cos(alpha3), d3],
#                  [0, 0, 0, 1]])

# t_34 = sp.Matrix([[sp.cos(theta4), -sp.sin(theta4)*sp.cos(alpha4), sp.sin(theta4)*sp.sin(alpha4), a4*sp.cos(theta4)],
#                  [sp.sin(theta4), sp.cos(theta4)*sp.cos(alpha4), -sp.cos(theta4)*sp.sin(alpha4), a4*sp.sin(theta4)],
#                  [0, sp.sin(alpha4), sp.cos(alpha4), d4],
#                  [0, 0, 0, 1]])

# t_45 = sp.Matrix([[sp.cos(theta5), -sp.sin(theta5)*sp.cos(alpha5), sp.sin(theta5)*sp.sin(alpha5), a5*sp.cos(theta5)],
#                  [sp.sin(theta5), sp.cos(theta5)*sp.cos(alpha5), -sp.cos(theta5)*sp.sin(alpha5), a5*sp.sin(theta5)],
#                  [0, sp.sin(alpha5), sp.cos(alpha5), d5],
#                  [0, 0, 0, 1]])

# t_56 = sp.Matrix([[sp.cos(theta6 - sp.pi/2), -sp.sin(theta6 - sp.pi/2)*sp.cos(alpha6), sp.sin(theta6 - sp.pi/2)*sp.sin(alpha6), a6*sp.cos(theta6 - sp.pi/2)],
#                  [sp.sin(theta6 - sp.pi/2), sp.cos(theta6 - sp.pi/2)*sp.cos(alpha6), -sp.cos(theta6 - sp.pi/2)*sp.sin(alpha6), a6*sp.sin(theta6 - sp.pi/2)],
#                  [0, sp.sin(alpha6), sp.cos(alpha6), d6],
#                  [0, 0, 0, 1]])

# t_67 = sp.Matrix([[sp.cos(theta7), -sp.sin(theta7)*sp.cos(alpha7), sp.sin(theta7)*sp.sin(alpha7), a7*sp.cos(theta7)],
#                  [sp.sin(theta7), sp.cos(theta7)*sp.cos(alpha7), -sp.cos(theta7)*sp.sin(alpha7), a7*sp.sin(theta7)],
#                  [0, sp.sin(alpha7), sp.cos(alpha7), d7],
#                  [0, 0, 0, 1]])

# t_07 = t_01*t_12*t_23*t_34*t_45*t_56*t_67
# print("t_07 matrix: ")
# sp.pprint(t_07)

# # Point 1 verification. Home position
# t_07_p1 = t_07.subs([(theta1, 0),(theta2, 0),(theta3, 0),(theta4, 0),(theta5, 0),(theta6, 0),(theta7, 0)])
# print("Point 1: ")
# sp.pprint(t_07_p1)

# # Point 2 verification. 
# t_07_p2 = t_07.subs([(theta1, np.pi/2),(theta2, 0),(theta3, -np.pi/2),(theta4, -np.pi/2),(theta5, 0),(theta6, 0),(theta7, 0)])
# print("Point 2: ")
# sp.pprint(t_07_p2)

# # Point 3 verification
# t_07_p3 = t_07.subs([(theta1, 0),(theta2, -np.pi/2),(theta3, 0),(theta4, 0),(theta5, 0),(theta6, -np.pi/2),(theta7, 0)])
# print("Point 3: ")
# sp.pprint(t_07_p3)

#Initial Conditions
# time = np.linspace(0,20,401)
time = np.linspace(0,5,101)
delta_T = .05
theta1_v = float(np.deg2rad(20))
theta2_v = float(np.deg2rad(20))
theta3_v = float(np.deg2rad(20)) 
theta4_v = float(np.deg2rad(20)) 
theta5_v = float(np.deg2rad(20))
theta6_v = float(np.deg2rad(20))
theta7_v = float(np.deg2rad(20))
thetas = np.array([theta1_v, theta2_v, theta3_v, theta4_v, theta5_v, theta6_v, theta7_v])

Px_v_array = np.zeros(101)
Py_v_array = np.zeros(101)
Pz_v_array = np.zeros(101)

theta1_v_array = np.zeros(101)
theta2_v_array = np.zeros(101)
theta3_v_array = np.zeros(101)
theta4_v_array = np.zeros(101)
theta5_v_array = np.zeros(101)
theta6_v_array = np.zeros(101)
theta7_v_array = np.zeros(101)

theta1_dot_v_array = np.zeros(101)
theta2_dot_v_array = np.zeros(101)
theta3_dot_v_array = np.zeros(101)
theta4_dot_v_array = np.zeros(101)
theta5_dot_v_array = np.zeros(101)
theta6_dot_v_array = np.zeros(101)
theta7_dot_v_array = np.zeros(101)

print("Calculating...")
# Repeat a for loop for every increment in time. Calculate joint positions and velocities at each increment in time.
for i in range(np.size(time)):
    t = time[i]

    # Circle velocity profile
    # Px_prime = -.02*(2*np.pi/5)*np.sin(np.pi/5*t)
    # Py_prime = .02*(2*np.pi/5)*np.cos(np.pi/5*t)
    # Pz_prime = 0.0
    Px_prime = .025
    Py_prime = .025
    Pz_prime = .025
    
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
    
print("Graphing...")
    
# fig = plt.figure()
# plt.plot(Py_v_array,Px_v_array, label="Motion")
# plt.title("End Effector Position")
# plt.xlabel("Y (mm)")
# plt.ylabel("X (mm)")
# plt.legend(loc="upper right")
# plt.show()

# Display the end effector path in xyz
fig1 = plt.figure("End Effector Positions")
ax1 = fig1.add_subplot(111, projection="3d")
ax1.plot3D(Px_v_array, Py_v_array, Pz_v_array, color="blue", label="Actual IK solutions")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_zlabel("Z (m)")
ax1.legend()

plt.show()

plt.plot(time, theta1_dot_v_array, label="theta1")
plt.plot(time, theta2_dot_v_array, label="theta2")
plt.plot(time, theta3_dot_v_array, label="theta3")
plt.plot(time, theta4_dot_v_array, label="theta4")
plt.plot(time, theta5_dot_v_array, label="theta5")
plt.plot(time, theta6_dot_v_array, label="theta6")
plt.plot(time, theta7_dot_v_array, label="theta7")
plt.title("Joint Velocities VS Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Angle (radians)")
plt.legend(loc="upper right")
plt.show()

plt.plot(time, Px_v_array, label="Px")
plt.plot(time, Py_v_array, label="Py")
plt.plot(time, Pz_v_array, label="Pz")
plt.title("End Effector Positions VS Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Position (mm)")
plt.legend(loc="upper right")
plt.show()
    
    
    


