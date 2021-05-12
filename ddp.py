import numpy as np
%matplotlib inline
import matplotlib.pylab as plt
from numpy import linalg

T = 10 #T+1 points
m = 1.
s0 = np.array([[10.],[-3.]])
starg = np.array([[0.],[0.]])
sweight = 1.
uweight = 1.
sweightT = 1.

u = np.zeros((2,T)) #commande en vitesse : vx,vy ; nu * T
Cs = sweight*np.eye(s0.shape[0])
Cu = uweight*np.eye(u.shape[0])
CsT = sweightT*np.eye(s0.shape[0])
dt = 1/T
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

def calcul_cout_et_affichage(s,u):
    sumcosts = 0.
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

def backwardpass(s):  
    Vx = [] #termes d'ordre 1 de T a 0
    Vxx = [] #termes d'ordre 2 de T a 0
    k = [] #gains de T a 0
    K = [] #gains de T a 0
    
    #at time T : final cost
    Vx.append(grad_finalcost(s[:,-1:]))
    Vxx.append(hess_finalcost())

    for t in range(T-1,-1,-1): #T-1 a 0
        #at time t < T
        Qx = Cs@s[:,t:t+1] + Fx.T@Vx[-1]
        Qu = Cu@u[:,t:t+1] + Fu.T@Vx[-1]
        Qxx = Cs + Fx.T@Vxx[-1]@Fx 
        Quu = Cu + Fu.T@Vxx[-1]@Fu
        Qxu = Fx.T@Vxx[-1]@Fu #+Lxu...
        Qux = Fu.T@Vxx[-1]@Fx #+Lux...
        k.append(np.linalg.inv(Quu)@Qu)
        K.append(np.linalg.inv(Quu)@Qux)
        Vx.append(Qx-Qxu@k[-1])
        Vxx.append(Qxx-Qxu@K[-1])
    return k,K

def forwardpass(k,K,u,s):
    for t in range (T):
        u[:,t:t+1]=-k[-1]-K[-1]@s[:,t:t+1]
        k.pop()
        K.pop()
        s[:,t+1:t+2] = next_state(s[:,t:t+1],u[:,t:t+1])
    return s,u

k,K = backwardpass(s)
s,u = forwardpass(k,K,u,s)

calcul_cout_et_affichage(s,u)
