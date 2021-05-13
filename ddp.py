import numpy as np
%matplotlib inline
import matplotlib.pylab as plt
from numpy import linalg


T = 1000
N = 1000 #N+1 points
x0 = np.array([[2.],[2.]]) #contrainte
xtarg = np.array([[0.],[0.]])
xweight = 1.
uweight = 1.
xweightT = 1.

u = np.zeros((2,N)) #commande en vitesse : vx,vy ; nu * N
Cx = xweight*np.eye(x0.shape[0])
Cu = uweight*np.eye(u.shape[0])
CxT = xweightT*np.eye(x0.shape[0])
dt = T/N
x=np.tile(x0,(1,N+1)) #nx * N+1 #attention :  dtype float...

assert(x.shape==(2,N+1))
assert(u.shape==(2,N))

def next_state(x,u):
    return x + u*dt

def cost(x,u):
    return 0.5*(x-xtarg).T@Cx@(x-xtarg) + 0.5*u.T@Cu@u

def Lx(x):
    return Cx@(x-xtarg)

def Lu(u):
    return Cu@u

def Lxx():
    return Cx

def Luu():
    return Cu

def finalcost(x):
    return 0.5*(x-xtarg).T@CxT@(x-xtarg)

def LxT(x):
    return CxT@(x-xtarg)

def LxxT():
    return CxT

def calcul_cout_et_affichage(x,u):
    sumcosts = 0.
    for t in range(0,N):
        xt = x[:,t:t+1]
        ut = u[:,t:t+1]
        sumcosts += cost(xt,ut)
        x[:,t+1:t+2] = next_state(xt,ut)
    sumcosts += finalcost(x[:,N:N+1])
    print("cout :")
    print(sumcosts)
    print("x=")
    print(x)
    print("u=")
    print(u)
    plt.scatter(x[0:1,:],x[1:2,:])

Fx = np.eye(2)
Fu = dt*np.eye(2)

def backwardpass(x):
    Vx = [] #termes d'ordre 1 de N a 0
    Vxx = [] #termes d'ordre 2 de N a 0
    k = [] #gains de N a 0
    K = [] #gains de N a 0
    
    #at time T : final cost
    Vx.append(LxT(x[:,-1:]))
    Vxx.append(LxxT())

    for t in range(N-1,-1,-1): #N-1 a 0
        #at time t < T
        Qx = Lx(x[:,t:t+1]) + Fx.T@Vx[-1]
        Qu = Lu(u[:,t:t+1]) + Fu.T@Vx[-1]
        Qxx = Lxx() + Fx.T@Vxx[-1]@Fx 
        Quu = Luu() + Fu.T@Vxx[-1]@Fu
        Qxu = Fx.T@Vxx[-1]@Fu #+Lxu...
        Qux = Fu.T@Vxx[-1]@Fx #+Lux...
        k.append(np.linalg.inv(Quu)@Qu)
        K.append(np.linalg.inv(Quu)@Qux)
        Vx.append(Qx-Qxu@k[-1])
        Vxx.append(Qxx-Qxu@K[-1])
    return k,K

def forwardpass(k,K,x):
    for t in range (N):
        u[:,t:t+1]=-k[-1]-K[-1]@x[:,t:t+1]
        k.pop()
        K.pop()
        x[:,t+1:t+2] = next_state(x[:,t:t+1],u[:,t:t+1])
    return x,u

k,K = backwardpass(x)
x,u = forwardpass(k,K,x)
calcul_cout_et_affichage(x,u)
