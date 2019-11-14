import numpy as np 
import sympy as sp
import pprint
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot

q1, q2, q3, q4, q5, q6, l1, l2, l3, l4, l5, l6, l7,l8 = sp.symbols('q1 q2 q3 q4 q5 q6 l1 l2 l3 l4 l5 l6 l7 l8')
q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18= sp.symbols('q7 q8 q9 q10 q11 q12 q13 q14 q15 q16 q17 q18')
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
def T_y(l):
  return np.array([ [1, 0, 0, 0],
                    [0, 1, 0, l],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                  ])

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

# matrix derivatives
# translation
def HT_z(l):
  HT_z = sp.diff(sp.Matrix(T_z(l)), l)
  return np.array(HT_z)
def HT_x(l):
  HT_x = sp.diff(sp.Matrix(T_x(l)), l)
  return np.array(HT_x)
# rotation
def HR_x(q):
  HR_x = sp.diff(sp.Matrix(R_x(q)), q)
  return np.array(HR_x)
def HR_y(q):
  HR_y = sp.diff(sp.Matrix(R_y(q)), q)
  return np.array(HR_y)
def HR_z(q):
  HR_z = sp.diff(sp.Matrix(R_z(q)), q)
  return np.array(HR_z) 


Ttool = np.eye(4)
# FANUC R-2000iC/165F Initial parameters
#joints limits
upper_lim = np.array([370, 136, 312, 720, 250, 720])*np.pi/180
lower_lim = np.array([370, 136, 312, 720, 250, 720])*-np.pi/180
# ideal parameters / measured parameters
idealParams = np.array([312, 0, 0, 0, 1075, 0, 0, 0, 225, 1280, 0, 0, 0, 0, 0, 0, 0, 0])
# real parameters
realParams = np.array([312+1.5, 0+1.5, 0.1, 0.1, 1075+1.5, 0.1, 0.1, 0.1, 225+1.5, 1280+1.5, 0.1, 0.1, 0+1.5, 0+1.5, 0.1, 0.1, 0+1.5, 0.1])
# real base and tools transformation matrices
TbaseR = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0+0.5], [0, 0, 0, 1]])
Ttool1R= T_z(90-0.1)
print(Ttool1R)
Ttool2R= sp.Matrix(multi_dot([R_x(q2), T_z(l1), R_x(q1)])).subs([(q2, 2*np.pi/3),(l1,90-0.1),(q1,-2*np.pi/3)])
Ttool3R= sp.Matrix(multi_dot([R_x(q2), T_z(l1), R_x(q1)])).subs([(q2, -2*np.pi/3),(l1,90+0.1),(q1,2*np.pi/3)])
# ideal base and tools transformation matrices
Tbase = np.eye(4)
Ttool1 =T_z(90)
Ttool2 = sp.Matrix(multi_dot([R_x(q2), T_z(l1), R_x(q1)])).subs([(q2, 2*np.pi/3),(l1,90),(q1,-2*np.pi/3)])
Ttool3 = sp.Matrix(multi_dot([R_x(q2), T_z(l1), R_x(q1)])).subs([(q2,-2*np.pi/3),(l1,90),(q1, 2*np.pi/3)])

B=np.zeros((15,1))
# noise
delta = 10*1e-3;


# use random configurations
qu = []
for i in range(100):
             qu.append(np.array([np.random.rand(1)*(-lower_lim[0] + upper_lim[0]) + lower_lim[0],
                        np.random.rand(1)*(-lower_lim[1] + upper_lim[1]) + lower_lim[1],
                        np.random.rand(1)*(-lower_lim[2] + upper_lim[2]) + lower_lim[2],
                        np.random.rand(1)*(-lower_lim[3] + upper_lim[3]) + lower_lim[3],
                        np.random.rand(1)*(-lower_lim[4] + upper_lim[4]) + lower_lim[4],
                        np.random.rand(1)*(-lower_lim[5] + upper_lim[5]) + lower_lim[5]]).ravel())

#generate experimental data for 30 experiments (10 configurations * 3 points)
def FK(q,params,delta,Tbase,Ttool):
       T = multi_dot([Tbase, R_z(q[0]), T_x(params[0]), T_y(params[1]), R_x(params[2]), R_y(q[1]+params[3]),\
                      T_x(params[4]),R_x(params[5]),R_z(params[6]), R_y(q[2]+params[7]),T_x(params[8]),T_z(params[9]),R_z(params[10]),\
       R_x(q[3]+params[11]),T_y(params[12]),T_z(params[13]),R_z(params[14]), R_y(q[4]+params[15]),T_z(params[16]),R_z(params[17]), R_x(q[5]),Ttool]) 
      
       T[:3,3] = T[:3,3] + delta*np.random.randn(1,3)
       return T

M1 = []
M2 = []
M3 = []
for i in range(len(qu)):
    
    M1.append(FK(qu[i],realParams, delta,TbaseR,Ttool1R))
    M2.append(FK(qu[i],realParams, delta,TbaseR,Ttool2R))
    M3.append(FK(qu[i],realParams, delta,TbaseR,Ttool3R))
# jacobian calculation
def Jacobian(q,params,Tbase,Ttool):

    Td1 = multi_dot([Tbase, R_z(q[0]), HT_x(l1), R_y(q[1]),T_x(params[4]), R_y(q[2]),\
                  T_x(params[8]),T_z(params[9]), R_x(q[3]), R_y(q[4]), R_x(q[5]), Ttool])
    J1= np.hstack((Td1[0,3], Td1[1,3], Td1[2,3]))
    
    
    Td2 = multi_dot([Tbase, R_z(q[0]), T_x(params[1]), HT_y(l1), R_y(q[1]),T_x(params[4]), R_y(q[2]),\
                    T_x(params[8]),T_z(params[9]), R_x(q[3]), R_y(q[4]), R_x(q[5]), Ttool])
    J2= np.hstack((Td2[0,3], Td2[1,3], Td2[2,3]))


    Td3 = multi_dot([Tbase, R_z(q[0]), T_x(params[0]),HR_x(q1).subs(q1,0), R_y(q[1]),T_x(params[4]), R_y(q[2]),\
                    T_x(params[8]),T_z(params[9]), R_x(q[3]), R_y(q[4]), R_x(q[5]), Ttool])
    J3= np.hstack((Td3[0,3], Td3[1,3], Td3[2,3]))

    
    Td4 = multi_dot([Tbase, R_z(q[0]), T_x(params[0]), R_y(q[1]), HR_y(q1).subs(q1,0), T_x(params[4]), R_y(q[2]), T_x(params[8]), T_z(params[9]), R_x(q[3]), R_y(q[4]), R_x(q[5]), Ttool])
    J4= np.hstack((Td4[0,3], Td4[1,3], Td4[2,3]))
    Td5 = multi_dot([Tbase, R_z(q[0]), T_x(params[0]), R_y(q[1]),HT_x(l1), R_y(q[2]), T_x(params[8]),\
                    T_z(params[9]), R_x(q[3]), R_y(q[4]), R_x(q[5]), Ttool])
    J5= np.hstack((Td5[0,3], Td5[1,3], Td5[2,3]))


    Td6 = multi_dot([Tbase, R_z(q[0]), T_x(params[0]), R_y(q[1]),T_x(params[4]), HR_x(0), R_y(q[2]),T_x(params[8]),T_z(params[9]), R_x(q[3]), R_y(q[4]), R_x(q[5]), Ttool])
    J6= np.hstack((Td6[0,3], Td6[1,3], Td6[2,3]))

    
    Td7 = multi_dot([Tbase, R_z(q[0]), T_x(params[0]), R_y(q[1]),T_x(params[4]),HR_z(q1).subs(q1,0), R_y(q[2]),\
                     T_x(params[8]), T_z(params[9]), R_x(q[3]), R_y(q[4]), R_x(q[5]), Ttool])
    J7= np.hstack((Td7[0,3], Td7[1,3], Td7[2,3]))

    
    Td8 = multi_dot([Tbase, R_z(q[0]), T_x(params[1]), R_y(q[1]),T_x(params[4]), R_y(q[2]),HR_y(q1).subs(q1,0),\
                     T_x(params(9)), T_z(params[9]), R_x(q[3]), R_y(q[4]), R_x(q[5]), Ttool])
    J8= np.hstack((Td8[0,3], Td8[1,3], Td8[2,3]))


    Td9 = multi_dot([Tbase, R_z(q[0]), T_x(params[0]), R_y(q[1]),T_x(params[4]), R_y(q[2]),HT_x(l1),\
                    T_x(params[8]),T_z(params[9]), R_x(q[3]), R_y(q[4]), R_x(q[5]), Ttool])
    J9= np.hstack((Td9[0,3], Td9[1,3], Td9[2,3]))
    

    Td10 = multi_dot([Tbase, R_z(q[0]),T_x(params[0]), R_y(q[1]),T_x(params[4]), R_y(q[2]),\
                 T_x(params[8]),HT_z(l1), T_z(params[9]), R_x(q[3]), R_y(q[4]), R_x(q[5]), Ttool])
    J10= np.hstack((Td10[0,3], Td10[1,3], Td10[2,3]))

    
    Td11 = multi_dot([Tbase, R_z(q[0]), T_x(params[0]), R_y(q[1]),T_x(params[4]), R_y(q[2]),\
                      T_x(params[8]),T_z(params[9]),HR_z(q1).subs(q1,0), R_x(q[3]), R_y(q[4]), R_x(q[5]), Ttool])
    J11= np.hstack((Td11[0,3], Td11[1,3], Td11[2,3]))


    Td12 = multi_dot([Tbase, R_z(q[0]), T_x(params[0]), R_y(q[1]),T_x(params[4]), R_y(q[2]),\
                    T_x(params[8]),T_z(params[9]), R_x(q[3]),HR_x(q1).subs(q1,0), R_y(q[4]), R_x(q[5]), Ttool])
    J12= np.hstack((Td12[0,3], Td12[1,3], Td12[2,3]))


    Td13 = multi_dot([Tbase, R_z(q[0]), T_x(params[0]), R_y(q[1]),T_x(params[4]), R_y(q[2]), T_x(params[8]),\
                    T_z(params[9]), R_x(q[3]),HT_y(l1), R_y(q[4]), R_x(q[5]), Ttool])
    J13= np.hstack((Td13[0,3], Td13[1,3], Td13[2,3]))


    Td14 = multi_dot([Tbase, R_z(q[0]), T_x(params[0]), R_y(q[1]),T_x(params[4]), R_y(q[2]), T_x(params[8]),\
                    T_z(params[9]), R_x(q[3]),HT_z(l1), R_y(q[4]), R_x(q[5]), Ttool])
    J14= np.hstack((Td14[0,3], Td14[1,3], Td14[2,3]))


    Td15 = multi_dot([Tbase, R_z(q[0]),T_x(params[0]), R_y(q[1]),T_x(params[4]), R_y(q[2]), T_x(params[8]),\
                    T_z(params[9]), R_x(q[3]),HR_z(q1).subs(q1,0), R_y(q[4]), R_x(q[5]), Ttool])
    J15= np.hstack((Td15[0,3], Td15[1,3], Td15[2,3]))

    
    Td16 = multi_dot([Tbase, R_z(q[0]), T_x(params[0]), R_y(q[1]),T_x(params[4]), R_y(q[2]), T_x(params[8]),\
                    T_z(params[9]), R_x(q[3]), R_y(q[4]),HR_y(q1).subs(q1,0), R_x(q[5]), Ttool])
    J16= np.hstack((Td16[0,3], Td16[1,3], Td16[2,3]))


    Td17 = multi_dot([Tbase, R_z(q[0]), T_x(params[0]), R_y(q[1]),T_x(params[4]), R_y(q[2]), T_x(params[8]),\
                    T_z(params[9]), R_x(q[3]), R_y(q[4]),HT_z(l1), R_x(q[5]), Ttool])
    J17= np.hstack((Td17[0,3], Td17[1,3], Td17[2,3]))


    Td18 = multi_dot([Tbase, R_z(q[0]), T_x(params[0]), R_y(q[1]),T_x(params[4]), R_y(q[2]), T_x(params[8]),\
                    T_z(params[9]), R_x(q[3]), R_y(q[4]),HR_z(q1).subs(q1,0), R_x(q[5]), Ttool])
    J18= np.hstack((Td18[0,3], Td18[1,3], Td18[2,3]))

    Jacob = np.vstack((J1.T, J2.T, J3.T, J4.T, J5.T, J6.T, J7.T, J8.T, J9.T, J10.T, J11.T, J12.T, J13.T, J14.T, J15.T, J16.T, J17.T, J18.T))
    return Jacob


# two step algorithm
while True:
    # calculate distance
    for i=1:size(q,1)
        M1i[i] = FK(q[i],idealParams + paramsDelta ,0,Tbase,Ttool1)
        M2i[i] = FK(q[i],idealParams + paramsDelta ,0,Tbase,Ttool1)
        M3i[i] = FK(q[i],idealParams + paramsDelta ,0,Tbase,Ttool1)
        M1dist[i]) = M1[i] - M1i[i]
        M2dist[i] = M2[i] - M2i[i]
        M3dist[i] = M3[i] - M3i[i]
        deltaDist[i]=[M1dist[1:3,4,i] M2dist[1:3,4,i] M3dist[1:3,4,i]]
    
    # 1st step find Tbase and Ttools
    B = np.eye(18)
    pbase = B[:3]
    phibase = B[3:5]
    rbase = np.array([[0, -phibase(3), phibase(2)], 
                      [phibase(3), 0, -phibase(1)], 
                      [-phibase(2), phibase(1), 0 ]]) + eye(3)
    Tbased = np.array([rbase, pbase, 0, 0, 0, 1])
    Ttool1d = np.array([[eye(3), multi_dot([rbase.T*B(7:9)])],
                        [0, 0, 0, 1]])
    Ttool2d = np.array([[eye(3), multi_dot([rbase.T*B(10:12)])],
                        [0, 0, 0, 1]])
    Ttool3d = np.array([[eye(3), multi_dot([rbase.T*B(13:15)])],
                        [0, 0, 0, 1]])
    
    
    # recalculate distance with new Tbase and Ttools
    for i=1:size(q,1)
        M1i[i] = FK(q[i],idealParams + paramsDelta ,0,Tbase*Tbased,Ttool1*Ttool1d)
        M2i[i] = FK(q[i],idealParams + paramsDelta ,0,Tbase*Tbased,Ttool2*Ttool2d)
        M3i[i] = FK(q[i],idealParams + paramsDelta ,0,Tbase*Tbased,Ttool3*Ttool3d)
        M1dist[i] = M1[i] - M1i[i]
        M2dist[i] = M2[i] - M2i[i]
        M3dist[i] = M3[i] - M3i[i]
        n[i] = mean([norm(deltaDist[1,i] norm(deltaDist[2,i] norm(deltaDist[3,i])])
    print("Real:", ideal)
    print("Calibrated:", params)
    #error distance average
    acctemp = acc;
    acc = mean(n)
    if (j>10000)||(abs(acctemp-acc)<1e-5)
        disp(j)
        acc
        break;
    # 2st step find robot parameters
    paramsDelta = paramsDelta + FindParams_Agilus(q, idealParams, deltaDist, Tbase*Tbased,Ttool1*Ttool1d,Ttool2*Ttool2d,Ttool3*Ttool3d)
    j = j+1
    print("Delta_before:", paramDelta)
    print("Delta_after:", paramsDelta)
    p = paramsDelta

from mpl_toolkits.mplot3d import Axes3D
x1 = np.arange(-np.pi,np.pi, 0.1)
y1 = np.arange(-np.pi,np.pi, 0.1)
xx, yy = np.meshgrid(x, y, sparse=True)
z1 = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(xx, yy, z1, 
                       cmap=plt.cm.coolwarm,linewidth=0,antialiased=False)

plt.show()

print(np.shape(z[1,1]))
print(z)

