import numpy as np 
import sympy as sp
import pprint
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot

#given constants 
frequency = 100
max_joint_velocity = 1
max_cartesian_velocity = 1
max_joint_acceleration = 1
max_cartesian_acceleration = 1
junction = 100
L = [1, 4 , 3]

q1, q2, q3, l1, l2, l3 = sp.symbols('q1 q2 q3 l1 l2 l3')

#transform matrices 
def R_x(q):
  return np.array([[1,     0,           0,   0],
                  [0, sp.cos(q), -sp.sin(q), 0],
                  [0, sp.sin(q), sp.cos(q),  0],
                  [0,      0,           0,   1]
                   ])


def R_y(q):                
  return np.array([[sp.cos(q),  0,      sp.sin(q), 0],
                  [0,           1,          0,     0],
                  [-sp.sin(q),  0,      sp.cos(q), 0],
                  [0,           0,          0,     1]
                  ])
def R_z(q):            
  return np.array([[sp.cos(q),    -sp.sin(q),    0,  0],
                  [ sp.sin(q),     sp.cos(q),    0,  0],
                  [0,                     0,     1,  0],
                  [0,                     0,     0,  1]
                  ])
def T_z(l):
     return np.array([ [1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, l],
                       [0, 0, 0, 1]
                      ])
def T_x(l):
  return np.array([ [1, 0, 0, l],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                  ])
 #RRR robot 
def FK(q,l):
  return np.array([
        (L[1] * np.cos(q[1]) + L[2] * np.cos(q[1] + q[2])) * np.cos(q[0]),
        (L[1] * np.cos(q[1]) + L[2] * np.cos(q[1] + q[2])) * np.sin(q[0]),
        L[0] - L[1] * np.sin(q[1]) - L[2] * np.sin(q[1] + q[2])])

def IK(x,l):
  x, y, z = x[0], x[1], x[2]

  a = (x ** 2 + y ** 2 + (z - l[0]) ** 2 - l[1] ** 2 - l[2] ** 2) / (2 * l[1] * l[2])
  b = - (np.sqrt(1 - a ** 2))

  q1 = np.arctan2(y, x)
  q2 = np.arctan2(z - l[0], np.sqrt(x ** 2 + y ** 2)) - np.arctan2(l[2] * b, l[1] + l[2] * a)
  q3 = np.arctan2(b, a)
  ik = np.array([q1,q2,q3])
  return ik

# Jacobian Skew theory
A1 = multi_dot([R_z(q1),T_z(l1)])
A2 = multi_dot([A1,R_y(q2),T_x(l2)])
A3 = multi_dot([A2,R_y(q3),T_x(l3)])

O_1 = A1[:3,3]
O_2 = A2[:3,3]
O_3 = A3[:3,3]

z1 = A1[:3, 2]
z2 = A2[:3, 2]
z3 = A3[:3, 2]

J1 = np.hstack(( np.cross(z1,(O_3 - O_1)),z1))
J2 = np.hstack(( np.cross(z2,(O_3 - O_2)),z2)) 
J3 = np.hstack(( np.cross(z3,(O_3 - O_3)),z3))
# -------
J = np.column_stack((J1,J2,J3))
# Inverse jacobian 
def J_inv(q):
    q1, q2, q3 = q[0], q[1], q[2]
    l1, l2, l3 = L[0], L[1], L[2]
    J_inv = \
    np.array([\
    [-np.sin(q1) / (l2 * np.cos(q2) + l3 * np.cos(q2 + q3)), np.cos(q1) / (l2 * np.cos(q2) + l3 * np.cos(q2 + q3)), 0],
    [np.cos(q1) * np.cos(q2 + q3) / (l2 * np.sin(q3)), np.sin(q1) * np.cos(q2 + q3) / (l2 * np.sin(q3)), -np.sin(q2 + q3) / (l2 * np.sin(q3))],
    [-(l2 * np.cos(q2) + l3 * np.cos(q2 + q3)) * np.cos(q1) / (l2 * l3 * np.sin(q3)), -(l2 * np.cos(q2) + \
    l3 * np.cos(q2 + q3)) * np.sin(q1) / (l2 * l3 * np.sin(q3)), (l2 * np.sin(q2) + l3 * np.sin(q2 + q3)) / (l2 * l3 * np.sin(q3))]])
    return J_inv

def ptp(q0, qf):
    dq = qf - q0
    dq_abs = np.abs(dq)
    # dq is a figure area, max_joint_velocity is a height
    t_ba = np.around(dq_abs / max_joint_velocity, 2)
    # t_b is a time for which acceleration > 0 and constant
    t_b = np.around(max_joint_velocity / max_joint_acceleration, 2)
    if np.any(t_b < t_ba):
        # trapezium
        t_a = t_ba - t_b  # t_a is a time of constant velocity
        t_a_max = np.amax(t_a)
        t_f = t_a_max + 2 * t_b

        
        each_joint_velocity = dq / (t_a_max + t_b)
        each_joint_acceleration = each_joint_velocity / t_b

        time = np.arange(0, t_f+0.005, 0.01)  
        v = np.zeros(shape=(dq.shape[0], time.shape[0]))
        x = np.zeros(shape=(dq.shape[0], time.shape[0]))
        a = np.zeros(shape=(dq.shape[0], time.shape[0]))

        x[:, 0] = q0

        for (i,), cur_time in np.ndenumerate(time):
            if cur_time < t_b:
                v[:, i] = each_joint_acceleration * cur_time
            elif cur_time < t_a_max + t_b:
                v[:, i] = each_joint_velocity
            else:
                v[:, i] = each_joint_acceleration * (t_f - cur_time)

        for (i,), cur_time in np.ndenumerate(time):
            if i > 0:
                a[:, i] = (v[:, i] - v[:, i - 1]) * frequency
                x[:, i] = x[:, i - 1] + (v[:, i] + v[:, i - 1]) / 2 * 0.01

        return x, v, a
    else:
        t_b = np.around(np.sqrt(dq_abs/max_joint_acceleration), 2)
        t_b_max = np.amax(t_b)
        t_f = 2 * t_b_max
        each_joint_velocity = dq / t_b_max
        each_joint_acceleration = each_joint_velocity / t_b_max
        time = np.arange(0, t_f + 0.005, 0.01)
        v = np.zeros(shape=(dq.shape[0], time.shape[0]))
        x = np.zeros(shape=(dq.shape[0], time.shape[0]))
        a = np.zeros(shape=(dq.shape[0], time.shape[0]))


        x[:, 0] = q0

        for (i,), cur_time in np.ndenumerate(time):
            if cur_time < t_b_max:
                v[:, i] = each_joint_acceleration * cur_time
            else:
                v[:, i] = each_joint_acceleration * (t_f - cur_time)

        for (i,), cur_time in np.ndenumerate(time):
            if i > 0:
                a[:, i-1] = (v[:, i] - v[:, i - 1]) * frequency
                x[:, i] = x[:, i - 1] + (v[:, i] + v[:, i - 1]) / 2 * 0.01

        return x, v, a

def lin(x0, xf):
    dist = np.linalg.norm(xf - x0)
    dist_z = xf[2] - x0[2]
    dist_y = xf[1] - x0[1]
    dist_x = xf[0] - x0[0]
    dist_xy = np.sqrt(dist_x**2 + dist_y**2)
    sin_a = dist_z / dist
    cos_a = dist_xy / dist
    cos_b = dist_x / dist_xy
    sin_b = dist_y / dist_xy

    q0 = IK(x0, L)
    qf = IK(xf, L)

    def get_joint_components(module):
        return np.array([module * cos_a * cos_b, module * cos_a * sin_b, module * sin_a])

    t_ba = np.around(dist / max_cartesian_velocity, 2)
    t_b = np.around(max_cartesian_velocity / max_cartesian_acceleration, 2)

    time = 0
    v_cartesian = 0

    if t_b < t_ba:
        t_a = t_ba - t_b
        t_f = t_ba + t_b
        time = np.arange(0, t_f + 0.005, 0.01)  
        v_cartesian = np.zeros(time.shape[0])  

        for (i,), cur_time in np.ndenumerate(time):
            if cur_time < t_b:
                v_cartesian[i] = max_cartesian_acceleration * cur_time
            elif cur_time < t_a + t_b:
                v_cartesian[i] = max_cartesian_velocity
            else:
                v_cartesian[i] = max_cartesian_acceleration * (t_f - cur_time)

    else:
        t_b = np.around(np.sqrt(dist/max_joint_acceleration), 2)
        t_f = 2 * t_b
        time = np.arange(0, t_f + 0.005, 0.01)  
        v_cartesian = np.zeros(time.shape[0])  
        for (i,), cur_time in np.ndenumerate(time):
            if cur_time < t_b:
                v_cartesian[i] = max_cartesian_acceleration * cur_time
            else:
                v_cartesian[i] = max_cartesian_acceleration * (t_f - cur_time)

    plt.figure(0)
    plt.ylabel('$v$, $\\dfrac{deg}{sec}$')
    plt.xlabel('time, s')
    plt.title('Cartesian $v$')
    x1 = time[0:-1]
    x2 = time[1:]
    y1 = v_cartesian[0:-1]
    y2 = v_cartesian[1:]
    plt.plot(x1, y1, x2, y2)
    plt.savefig('figs/Cartesian.jpg')
    plt.show()

    x = np.zeros(shape=(3, time.shape[0]))
    v = np.zeros(shape=(3, time.shape[0])) 
    a = np.zeros(shape=(3, time.shape[0]))

    x[:, 0] = q0
   
    for (i,), cur_time in np.ndenumerate(time):
        if i == 0:
            continue
        v[:, i] = np.dot(J_inv(x[:, i - 1]), get_joint_components(v_cartesian[i]))
        a[:, i - 1] = (v[:, i] - v[:, i - 1]) * frequency
        x[:, i] = x[:, i - 1] + (v[:, i] + v[:, i - 1]) / 2 * 0.01  

    return x, v, a

def plots(x, v, a):
  time = np.arange(0, v.shape[1] * 0.01 - 0.005, 0.01)

  plt.figure(1)
  plt.ylabel('$x$, rad')
  plt.xlabel('$t$, seconds')
  plt.title('Joints positions')

  for i in range(0, v.shape[0]):
      
      plt.plot(time, x[i, :],'C'+str(i+1), label = "Joint {}".format(i+1) )
  plt.legend()
  plt.savefig('figs/Joint positions.jpg')
  plt.show()

  plt.figure(2)
  plt.ylabel('$v$, $\\dfrac{deg}{sec}$')
  plt.xlabel('time, s')
  plt.title('Joints velocities')

  for i in range(0, v.shape[0]):
      plt.plot(time, v[i, :], 'C'+str(i+1), label = "Joint {}".format(i+1) )
  plt.legend()
  plt.savefig('figs/Joint vel.jpg')
  plt.show()

  plt.figure(3)
  plt.ylabel("$\\dfrac{rad}{s^2}$")
  plt.xlabel('$t$, $s*10^{-3}$')
  plt.title('joint accelerations')
  cartesian_pos = np.zeros(shape=(3, x.shape[1]))
  for i in range(0, v.shape[0]):
      plt.plot(time, a[i, :], 'C'+str(i+1), label = "Joint {}".format(i+1))
  plt.legend()
  plt.savefig('figs/Joint acc.jpg')
  plt.show()

  for i in range(0, x.shape[1]):
      cartesian_pos[:, i] = FK(x[:, i].flatten(), L)

  plt.figure(4)
  plt.ylabel('$x$, meters')
  plt.xlabel('$t$, seconds')
  plt.title('X-cartesian')
  plt.plot(time, cartesian_pos[0])
  plt.savefig('figs/X-cartesian.jpg')
  plt.show()


  plt.figure(5)
  plt.ylabel('$x$, meters')
  plt.xlabel('$t$, seconds')
  plt.title('Y-cartesian')
  plt.plot(time, cartesian_pos[1])
  plt.savefig('figs/Y-cartesian.jpg')
  plt.show()


  plt.figure(6)
  plt.ylabel('$x$, meters')
  plt.xlabel('$t$, seconds')
  plt.title('Z-cartesian')
  plt.plot(time, cartesian_pos[2])
  plt.savefig('figs/Z-cartesian.jpg')
  plt.show()

def junction(t1, t2,):
    v1 = t1[1]
    new_x = t1[0]
    new_a = t1[2]
    v2 = t2[1]
    junction = 100
    for i in range(-100, 0):
        v1[:, i] += v2[:, junction+i]
        new_a[:, i] = (v1[:, i] - v1[:, i - 1]) * frequency
        new_x[:, i] = new_x[:, i - 1] + (v1[:, i] + v1[:, i - 1]) / 2 * 0.01
    v1 = np.hstack((v1, v2[:, junction:]))
    new_x = np.hstack((new_x, t2[0][:, junction:]))
    new_a = np.hstack((new_a, t2[2][:, junction:]))
    return new_x, v1, new_a

q_0 = np.array([0, 0, 0])
x1 = np.array([4, 5, 1])
q_1 = IK(x1, L)

x2 = np.array([1, 5, 0.9])
q_3 = np.array([1, 1, 1])
q_4 = np.array([1.5, 1.5, 1.5])

trajectory1 = ptp(q_0, q_1)
trajectory2 = lin(x1, x2)
trajectory3 = ptp(trajectory2[0][:, -1], q_3)
trajectory4 = ptp(q_3, q_4)

t12 = junction(trajectory1, trajectory2)
t13 = junction(t12, trajectory3)
t14 = junction(t13, trajectory4)

x = t14[0]
v = t14[1]
a = t14[2]

plots(x, v, a)