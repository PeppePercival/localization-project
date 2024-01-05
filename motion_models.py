import sympy as sp
import numpy as np
sp.init_printing(use_latex='mathjax')
from sympy import symbols, Matrix, latex

def velocity_motion_model():
    x, y, theta, v, w, dt = symbols('x y theta v w dt')
    gux = Matrix([x,y,theta]) + Matrix([[-v/w*sp.sin(theta)+v/w*sp.sin(theta+w*dt)],[v/w*sp.cos(theta)-v/w*sp.cos(theta+w*dt)],[w*dt]])
    eval_gux = sp.lambdify((x, y, theta, v, w, dt), gux, 'numpy')

    Gt = Matrix([[1,0,v/w*(-sp.cos(theta)+sp.cos(theta+w*dt))],[0,1,v/w*(-sp.sin(theta)+sp.sin(theta+w*dt))],[0,0,1]])
    eval_Gt = sp.lambdify((theta, v, w, dt), Gt, 'numpy')
    Vt = Matrix([[(-sp.sin(theta)+sp.sin(theta+w*dt))/w, v/w**2*(sp.sin(theta)-sp.sin(theta+w*dt)) + v/w*sp.cos(theta+w*dt)*dt],
                 [(sp.cos(theta)-sp.cos(theta+w*dt))/w, -(v/w**2*(sp.cos(theta)-sp.cos(theta+w*dt))) + v/w*sp.sin(theta+w*dt)*dt],
                [0, dt]])
    eval_Vt = sp.lambdify((theta, v, w, dt), Vt, 'numpy')

    return eval_gux, eval_Gt, eval_Vt

def get_odometry_input(x, x_prev):
    x0 = x_prev[0,0]
    y0 = x_prev[1,0]
    theta0 = x_prev[2,0]
    x1 = x[0,0]
    y1 = x[1,0]
    theta1 = x[2,0]
    rot1 = np.arctan2((y1-y0), (x1-x0))-theta0
    trasl = np.sqrt((x1-x0)**2+(y1-y0)**2)
    rot2 = theta1 - rot1 - theta0
    return np.array([[rot1, trasl, rot2]]).T

def odometry_motion_model():
    x, y, theta, rot1, trasl, rot2 = symbols('x y theta rot1 trasl rot2')
    gux_odom = Matrix([x, y, theta]) + Matrix([[trasl*sp.cos(rot1+theta)], [trasl*sp.sin(rot1+theta)], [rot1+rot2]])
    Gt_odom = gux_odom.jacobian(Matrix([x,y,theta]))
    Vt_odom = gux_odom.jacobian(Matrix([ rot1, trasl, rot2]))

    eval_gux_odom = sp.lambdify((x, y, theta, trasl, rot1, rot2), gux_odom, 'numpy')
    eval_Gt_odom = sp.lambdify((theta, trasl, rot1), Gt_odom, 'numpy')
    eval_Vt_odom = sp.lambdify((theta, trasl, rot1), Vt_odom, 'numpy')

    return eval_gux_odom, eval_Gt_odom, eval_Vt_odom

