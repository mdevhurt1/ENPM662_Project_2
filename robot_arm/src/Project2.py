import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

#Define Joint Angles
theta1 = sp.Symbol('th1')
theta2 = sp.Symbol('th2') # +90 degrees to put in home position
theta3 = sp.Symbol('th3')
theta4 = sp.Symbol('th4')
theta5 = sp.Symbol('th5')
theta6 = sp.Symbol('th6') # -90 degrees to put in home position
theta7 = sp.Symbol('th7')

# DH Parameters
a1 = 0
a2 = 0
a3 = 0
a4 = 0
a5 = 0
a6 = 0
a7 = -.085 # Negative because it goes in negative x7 direction.
alpha1 = sp.pi/2
alpha2 = -sp.pi/2
alpha3 = sp.pi/2
alpha4 = -sp.pi/2
alpha5 = sp.pi/2
alpha6 = -sp.pi/2
alpha7 = 0
d1 = .1651
d2 = 0
d3 = .25503
d4 = 0
d5 = .42746
d6 = 0
d7 = 0

t_01 = sp.Matrix([[sp.cos(theta1), -sp.sin(theta1)*sp.cos(alpha1), sp.sin(theta1)*sp.sin(alpha1), a1*sp.cos(theta1)],
                 [sp.sin(theta1), sp.cos(theta1)*sp.cos(alpha1), -sp.cos(theta1)*sp.sin(alpha1), a1*sp.sin(theta1)],
                 [0, sp.sin(alpha1), sp.cos(alpha1), d1],
                 [0, 0, 0, 1]])

t_12 = sp.Matrix([[sp.cos(theta2 + sp.pi/2), -sp.sin(theta2 + sp.pi/2)*sp.cos(alpha2), sp.sin(theta2 + sp.pi/2)*sp.sin(alpha2), a2*sp.cos(theta2 + sp.pi/2)],
                 [sp.sin(theta2 + sp.pi/2), sp.cos(theta2 + sp.pi/2)*sp.cos(alpha2), -sp.cos(theta2 + sp.pi/2)*sp.sin(alpha2), a2*sp.sin(theta2 + sp.pi/2)],
                 [0, sp.sin(alpha2), sp.cos(alpha2), d2],
                 [0, 0, 0, 1]])

t_23 = sp.Matrix([[sp.cos(theta3), -sp.sin(theta3)*sp.cos(alpha3), sp.sin(theta3)*sp.sin(alpha3), a3*sp.cos(theta3)],
                 [sp.sin(theta3), sp.cos(theta3)*sp.cos(alpha3), -sp.cos(theta3)*sp.sin(alpha3), a3*sp.sin(theta3)],
                 [0, sp.sin(alpha3), sp.cos(alpha3), d3],
                 [0, 0, 0, 1]])

t_34 = sp.Matrix([[sp.cos(theta4), -sp.sin(theta4)*sp.cos(alpha4), sp.sin(theta4)*sp.sin(alpha4), a4*sp.cos(theta4)],
                 [sp.sin(theta4), sp.cos(theta4)*sp.cos(alpha4), -sp.cos(theta4)*sp.sin(alpha4), a4*sp.sin(theta4)],
                 [0, sp.sin(alpha4), sp.cos(alpha4), d4],
                 [0, 0, 0, 1]])

t_45 = sp.Matrix([[sp.cos(theta5), -sp.sin(theta5)*sp.cos(alpha5), sp.sin(theta5)*sp.sin(alpha5), a5*sp.cos(theta5)],
                 [sp.sin(theta5), sp.cos(theta5)*sp.cos(alpha5), -sp.cos(theta5)*sp.sin(alpha5), a5*sp.sin(theta5)],
                 [0, sp.sin(alpha5), sp.cos(alpha5), d5],
                 [0, 0, 0, 1]])

t_56 = sp.Matrix([[sp.cos(theta6 - sp.pi/2), -sp.sin(theta6 - sp.pi/2)*sp.cos(alpha6), sp.sin(theta6 - sp.pi/2)*sp.sin(alpha6), a6*sp.cos(theta6 - sp.pi/2)],
                 [sp.sin(theta6 - sp.pi/2), sp.cos(theta6 - sp.pi/2)*sp.cos(alpha6), -sp.cos(theta6 - sp.pi/2)*sp.sin(alpha6), a6*sp.sin(theta6 - sp.pi/2)],
                 [0, sp.sin(alpha6), sp.cos(alpha6), d6],
                 [0, 0, 0, 1]])

t_67 = sp.Matrix([[sp.cos(theta7), -sp.sin(theta7)*sp.cos(alpha7), sp.sin(theta7)*sp.sin(alpha7), a7*sp.cos(theta7)],
                 [sp.sin(theta7), sp.cos(theta7)*sp.cos(alpha7), -sp.cos(theta7)*sp.sin(alpha7), a7*sp.sin(theta7)],
                 [0, sp.sin(alpha7), sp.cos(alpha7), d7],
                 [0, 0, 0, 1]])

t_07 = t_01*t_12*t_23*t_34*t_45*t_56*t_67
print("t_07 matrix: ")
sp.pprint(t_07)

# Point 1 verification. Home position
t_07_p1 = t_07.subs([(theta1, 0),(theta2, 0),(theta3, 0),(theta4, 0),(theta5, 0),(theta6, 0),(theta7, 0)])
print("Point 1: ")
sp.pprint(t_07_p1)

# Point 2 verification. 
t_07_p2 = t_07.subs([(theta1, np.pi/2),(theta2, 0),(theta3, -np.pi/2),(theta4, -np.pi/2),(theta5, 0),(theta6, 0),(theta7, 0)])
print("Point 2: ")
sp.pprint(t_07_p2)

# Point 3 verification
t_07_p3 = t_07.subs([(theta1, 0),(theta2, -np.pi/2),(theta3, 0),(theta4, 0),(theta5, 0),(theta6, -np.pi/2),(theta7, 0)])
print("Point 3: ")
sp.pprint(t_07_p3)

# Problem 2. Using method 2 from lecture 8.
t_01 = t_01
t_02 = t_01*t_12
t_03 = t_02*t_23
t_04 = t_03*t_34
t_05 = t_04*t_45
t_06 = t_05*t_56
t_07 = t_06*t_67

z_01 = t_01[0:3, 2]
z_02 = t_02[0:3, 2]
z_03 = t_03[0:3, 2]
z_04 = t_04[0:3, 2]
z_05 = t_05[0:3, 2]
z_06 = t_06[0:3, 2]
z_07 = t_07[0:3, 2]

Px = t_07[0,3]
Py = t_07[1,3]
Pz = t_07[2,3]

Px1 = sp.diff(Px, theta1)
Py1 = sp.diff(Py, theta1)
Pz1 = sp.diff(Pz, theta1)

Px2 = sp.diff(Px, theta2)
Py2 = sp.diff(Py, theta2)
Pz2 = sp.diff(Pz, theta2)

Px3 = sp.diff(Px, theta3)
Py3 = sp.diff(Py, theta3)
Pz3 = sp.diff(Pz, theta3)

Px4 = sp.diff(Px, theta4)
Py4 = sp.diff(Py, theta4)
Pz4 = sp.diff(Pz, theta4)

Px5 = sp.diff(Px, theta5)
Py5 = sp.diff(Py, theta5)
Pz5 = sp.diff(Pz, theta5)

Px6 = sp.diff(Px, theta6)
Py6 = sp.diff(Py, theta6)
Pz6 = sp.diff(Pz, theta6)

Px7 = sp.diff(Px, theta7)
Py7 = sp.diff(Py, theta7)
Pz7 = sp.diff(Pz, theta7)

Jac_full = sp.Matrix([[Px1, Px2, Px3, Px4, Px5, Px6, Px7],
               [Py1, Py2, Py3, Py4, Py5, Py6, Py7],
               [Pz1, Pz2, Pz3, Pz4, Pz5, Pz6, Pz7],
               [z_01[0], z_02[0], z_03[0], z_04[0], z_05[0], z_06[0], z_07[0]],
               [z_01[1], z_02[1], z_03[1], z_04[1], z_05[1], z_06[1], z_07[1]],
               [z_01[2], z_02[2], z_03[2], z_04[2], z_05[2], z_06[2], z_07[2]]])

# Virtual joint gain
kv = .2

# Create a 6x6 simplified jacobian where theta2 = kv*theta1
Jac_simp = Jac_full.copy()
Jac_simp[:,0] = Jac_simp[:,0] + Jac_simp[:,1]
Jac_simp.col_del(1)
Jac_simp = Jac_simp.subs([(theta2, kv*theta1)])

print("Jacobian Matrix: ")
sp.pprint(Jac_simp)

#Initial Conditions
# time = np.linspace(0,20,401)
time = np.linspace(0,5,101)
delta_T = .05
theta1_v = float(np.deg2rad(45))
theta2_v = kv*theta1_v
theta3_v = float(np.deg2rad(45)) 
theta4_v = float(np.deg2rad(45)) 
theta5_v = float(np.deg2rad(0))
theta6_v = float(np.deg2rad(0))
theta7_v = float(np.deg2rad(0))

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

print("Calculating...")
# Repeat a for loop for every increment in time. Calculate joint positions and velocities at each increment in time.
for i in range(np.size(time)):
    t = time[i]
    
    # alpha = -4*np.pi/125 * t**3 + 6*np.pi/25 * t**2
    # alpha_prime = -8*np.pi/125 * t**2 + 12*np.pi/25 * t
    # Px_prime = .150*np.cos(alpha)*alpha_prime
    # Py_prime = -.150*np.sin(alpha)*alpha_prime
    # Pz_prime = 0
    
    # Travel in a straight line
    Px_prime = .025
    Py_prime = .025
    Pz_prime = .025
    
    Jacobian_v = Jac_simp.subs([(theta1,theta1_v),(theta3,theta3_v),(theta4,theta4_v),
                                (theta5,theta5_v),(theta6,theta6_v),(theta7,theta7_v)])
    Jacobian_v = np.asarray(Jacobian_v, dtype=np.float64)
    
    effector = np.array([[Px_prime],[Py_prime],[Pz_prime], [0], [0], [0]])
    theta_dots = np.linalg.pinv(Jacobian_v).dot(effector)
    
    # #///Old version
    # Jacobian_inv_v = np.linalg.inv(Jacobian_v)
    
    # effector = np.array([[Px_prime],[Py_prime],[Pz_prime], [0], [0], [0]])
    
    # theta_dots = np.matmul(Jacobian_inv_v, effector)
    # #//Old version
    
    theta1_v = theta1_v + theta_dots[0]*delta_T
    theta2_v = kv*theta1_v
    theta3_v = theta3_v + theta_dots[1]*delta_T
    theta4_v = theta4_v + theta_dots[2]*delta_T
    theta5_v = theta5_v + theta_dots[3]*delta_T
    theta6_v = theta6_v + theta_dots[4]*delta_T
    theta7_v = theta7_v + theta_dots[5]*delta_T
    
    theta1_v = float(theta1_v[0])
    theta2_v = float(theta2_v[0])
    theta3_v = float(theta3_v[0])
    theta4_v = float(theta4_v[0])
    theta5_v = float(theta5_v[0])
    theta6_v = float(theta6_v[0])
    theta7_v = float(theta7_v[0])
    
    Px_v = Px.subs([(theta1,theta1_v),(theta2,theta2_v),(theta3,theta3_v),
                    (theta4,theta4_v),(theta5,theta5_v),(theta6,theta6_v),(theta7,theta7_v)])
    Py_v = Py.subs([(theta1,theta1_v),(theta2,theta2_v),(theta3,theta3_v),
                    (theta4,theta4_v),(theta5,theta5_v),(theta6,theta6_v),(theta7,theta7_v)])
    Pz_v = Pz.subs([(theta1,theta1_v),(theta2,theta2_v),(theta3,theta3_v),
                    (theta4,theta4_v),(theta5,theta5_v),(theta6,theta6_v),(theta7,theta7_v)])
    
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
    
print("Graphing...")
    
fig = plt.figure()
plt.plot(Py_v_array,Px_v_array, label="Motion")
plt.title("End Effector Position")
plt.xlabel("Y (mm)")
plt.ylabel("X (mm)")
plt.legend(loc="upper right")
plt.show()

# # plt.plot(time, theta1_v_array, label="theta1")
# # plt.plot(time, theta2_v_array, label="theta2")
# # plt.plot(time, theta3_v_array, label="theta3")
# # plt.plot(time, theta4_v_array, label="theta4")
# # plt.plot(time, theta5_v_array, label="theta5")
# # plt.plot(time, theta6_v_array, label="theta6")
# # plt.title("Joint Angles VS Time")
# # plt.xlabel("Time (seconds)")
# # plt.ylabel("Angle (radians)")
# # plt.legend(loc="upper right")
# # plt.show()


plt.plot(time, Px_v_array, label="Px")
plt.plot(time, Py_v_array, label="Py")
plt.plot(time, Pz_v_array, label="Pz")
plt.title("End Effector Positions VS Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Position (mm)")
plt.legend(loc="upper right")
plt.show()
    
    
    


