import numpy as np
%matplotlib inline
import matplotlib.pylab as plt
from numpy import linalg

T = 2 #T+1 points
m = 1.
s0 = np.array([[1.],[1.]])
starg = np.array([[0.],[0.]])
sweight = 1.
uweight = 1.
sweightT = 1.

u = np.zeros((2,T)) #commande en vitesse : vx,vy ; nu * T
#u[0:1,:]=1
#u[1:2,:]=1
Cs = sweight*np.eye(s0.shape[0])
Cu = uweight*np.eye(u.shape[0])
CsT = sweightT*np.eye(s0.shape[0])

dt = 1/T
sumcosts = 0.
s=np.tile(s0,(1,T+1)) #ns * T+1 #attention :  dtype float...
assert(s.shape==(2,T+1))
assert(u.shape==(2,T))

def next_state(s,u):
    return s + u*dt

def cost(s,u):
    return 0.5*(s-starg).T@Cs@(s-starg) + 0.5*u.T@Cu@u

def finalcost(s):
    return 0.5*(s-starg).T@CsT@(s-starg)

def grad_finalcost(s):
    return CsT@(s-starg)

def hess_finalcost():
    return CsT

for t in range(0,T):
    st = s[:,t:t+1]
    ut = u[:,t:t+1]
    sumcosts += cost(st,ut)
    s[:,t+1:t+2] = next_state(st,ut)
sumcosts += finalcost(s[:,T:T+1])

print("cout :")
print(sumcosts)
print("s=")
print(s)
print("u=")
print(u)
plt.scatter(s[0:1,:],s[1:2,:])

######






Fx = np.eye(2)
Fu = dt*np.eye(2)

def backwardpass():  
    Vx = [] #termes d'ordre 1 de T a 0
    Vxx = [] #termes d'ordre 2 de T a 0
    k = [] #gains de T a 0
    K = [] #gains de T a 0
    
    #at time T : final cost
    Vx.append(grad_finalcost(s[:,-1:]))
    Vxx.append(hess_finalcost())

    for t in range(T-1,-1,-1): #T-1 a 0
        #at time t < T
        Qx = Vx[-1].T@Fx + Cs@s[:,t:t+1]
        Qu = Vx[-1].T@Fu + Cu@u[:,t:t+1]
        Qxx = Fx.T@Vxx[0]@Fx + Cs
        Quu = Fu.T@Vxx[0]@Fu + Cu
        Qxu = Fx.T@Vxx[0]@Fu #+Lxu...
        Qux = Fu.T@Vxx[0]@Fx #+Lux...
        k.append(-np.linalg.inv(Quu)@Qu)
        K.append(-np.linalg.inv(Quu)@Qux)
        Vx.append(Qx-Qxu@k[-1])
        Vxx.append(Qxx-Qxu@K[-1])

def forwardpass():
    

def hess(Lxx,Luu,Lxu,Lux):
    Lxx = Cs
    Luu = Cu
    Lxu = np.zeros((s0.shape[0],u.shape[0]))
    Lux = np.zeros((u.shape[0],s0.shape[0]))
    Hess1 = np.concatenate((Lxx,Lxu),axis=1)
    Hess2 = np.concatenate((Lux,Luu),axis=1)
    return np.concatenate((Hess1,Hess2),axis=0)

backwardpass()
