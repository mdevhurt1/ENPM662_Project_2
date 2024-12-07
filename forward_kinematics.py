#!/usr/bin/env python3

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

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

sym.pprint(Tn[-1])