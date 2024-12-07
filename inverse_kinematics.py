#!/usr/bin/env python3

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

def compute_symbolic_jacobian(Tn):
    """
    Compute the full symbolic Jacobian for the end-effector, combining both linear and angular parts.
    """
    Z = [sym.Matrix([0, 0, 1])]
    O = [sym.Matrix([0, 0, 0])]
    for T in Tn:
        Z.append(sym.Matrix(T[:3, 2]))
        O.append(sym.Matrix(T[:3, 3]))

    Jv = sym.zeros(3, 7)
    Jw = sym.zeros(3, 7)

    # Each column of Jv and Jw is derived from the position and orientation of each joint
    for i in range(7):
        Jv[:, i] = Z[i].cross(O[7] - O[i])  # Linear velocity part
        Jw[:, i] = Z[i]                     # Angular velocity part

    return sym.Matrix.vstack(Jv, Jw)

def dh_transform(theta, d, a, alpha):
        """
        Compute a single Denavit-Hartenberg transformation matrix for given parameters.
        """
        return sym.Matrix([
            [sym.cos(theta), -sym.sin(theta)*sym.cos(alpha),  sym.sin(theta)*sym.sin(alpha), a*sym.cos(theta)],
            [sym.sin(theta),  sym.cos(theta)*sym.cos(alpha), -sym.cos(theta)*sym.sin(alpha), a*sym.sin(theta)],
            [0,               sym.sin(alpha),                 sym.cos(alpha),                d],
            [0,               0,                              0,                             1]
        ])

# Define symbolic variables for joint angles
theta_symbols = sym.symbols('theta1 theta2 theta3 theta4 theta5 theta6 theta7', real=True)
(theta1, theta2, theta3, 
theta4, theta5, theta6, 
theta7) = theta_symbols

# Denavit-Hartenberg parameters for the robot
DH = sym.Matrix([
    [theta1, 165.1,   0,    pi/2],
    [theta2, 0,       0,   -pi/2],
    [theta3, 255.03,  0,    pi/2],
    [theta4, 0,       0,   -pi/2],
    [theta5, 427.46,  0,    pi/2],
    [theta6, 0,       0,   -pi/2],
    [theta7, 0,      -85,   0]
])

# Compute forward kinematics transformation matrices for each link
An = [dh_transform(*DH[i, :]) for i in range(7)]
Tn = [An[0]]
for i in range(6):
    Tn.append(Tn[i] @ An[i+1])


# Compute the full symbolic Jacobian
J = compute_symbolic_jacobian(Tn)

sym.pprint(J)