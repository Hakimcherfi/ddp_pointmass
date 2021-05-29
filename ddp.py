import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg

T = 100 #T+1 points
dt = 0.01
x0 = np.array([[-1.],[-1.],[-1.],[0.],[0.],[0.]]) #contrainte
xtarg = np.array([[1.],[1.1],[1.2],[0.],[0.],[0.]])
xweight = 10.
uweight = .01
xweightT = 1000.
xsphere = np.array([[0.],[0.],[0.]]) #centre sphere
Rsphere = .5 #vrai rayon sphere
distsecu = .1 #ajout a variable precedente

#####
dimx = x0.shape[0]
dimspace = int(dimx/2)
dimu = dimspace
assert((dimspace,1)==xsphere.shape)
assert(x0.shape==xtarg.shape)
assert(distsecu>0.)
u = np.zeros((dimu,T)) 
x=np.tile(x0,(1,T+1)) #nx * N+1 #attention :  dtype float...
Fx = np.eye(dimx)
Fx[:dimspace,dimspace:]=dt*np.eye(dimspace)
Fu = np.concatenate([0.5*dt**2*np.eye(dimspace),dt*np.eye(dimspace)])

def next_state(x,u):
    return Fx@x + Fu@u

def costobstacle(x):
    distance = np.linalg.norm(x-xsphere)
    if (distance > Rsphere + distsecu):
        return 0.
    else:
        if (distance <= Rsphere):
            d = 0
        else :
            d = distance - Rsphere
        return (d-distsecu)**2
                
def costx(x):
    Cx = xweight*np.eye(dimx)
    return 0.5*(x-xtarg).T@Cx@(x-xtarg) + 10000*costobstacle(x[:dimspace,:])

def costu(u):
    Cu = uweight*np.eye(dimu)
    return 0.5*u.T@Cu@u

def cost(x,u):
    return costx(x)+costu(u)
    
def finalcost(x):
    CxT = xweightT*np.eye(dimx)
    return 0.5*(x-xtarg).T@CxT@(x-xtarg)

def gradient(f):
    def fbis(x,eps=0.0001):
        dim = x.shape[0]
        grad = np.zeros((dim,1))
        for n in range(dim):
            h = np.zeros((dim,1))
            h[n:n+1,:] = eps
            grad[n:n+1,:]=((f(x+h)-f(x-h))/(2*eps))
        return grad
    return fbis

def hessien(f):
    def fbis(x,eps=0.0001):
        dim = x.shape[0]
        hess = np.zeros((dim,dim))
        for n in range(dim):
            h = np.zeros((dim,1))
            h[n:n+1,:]=eps
            hess[n:n+1,n:n+1] = (f(x+h)+f(x-h)-2*f(x))/(eps**2)
        for n in range(dim):
            for m in range(n+1,dim):
                h = np.zeros((dim,1))
                h[n:n+1,:]=eps
                h[m:m+1,:]=eps
                hess[n:n+1,m:m+1]=0.5*((f(x+h)+f(x-h)-2*f(x))/(eps**2)-hess[n:n+1,n:n+1]-hess[m:m+1,m:m+1])
                hess[m:m+1,n:n+1]=hess[n:n+1,m:m+1]
        return hess
    return fbis

def Lx(x):
    return gradient(costx)(x)

def Lu(u):
    return gradient(costu)(u)

def Lxx(x):
    return hessien(costx)(x)

def Luu(u):
    return hessien(costu)(u)

def LxT(x):
    return gradient(finalcost)(x)

def LxxT(x):
    return hessien(finalcost)(x)

def backwardpass(x,u):
    Vx = [] #termes d'ordre 1 de T a 0
    Vxx = [] #termes d'ordre 2 de T a 0
    k = [] #gains de T a 0
    K = [] #gains de T a 0
    
    #at time T : final cost
    Vx.append(LxT(x[:,-1:]))
    Vxx.append(LxxT(x[:,-1:]))

    for t in range(T-1,-1,-1): #T-1 a 0
        #at time t < T
        Qx = Lx(x[:,t:t+1]) + Fx.T@Vx[-1]
        Qu = Lu(u[:,t:t+1]) + Fu.T@Vx[-1]
        Qxx = Lxx(x[:,t:t+1]) + Fx.T@Vxx[-1]@Fx 
        Quu = Luu(u[:,t:t+1]) + Fu.T@Vxx[-1]@Fu
        Qxu = Fx.T@Vxx[-1]@Fu #+Lxu...
        Qux = Fu.T@Vxx[-1]@Fx #+Lux...
        k.append(np.linalg.inv(Quu)@Qu)
        K.append(np.linalg.inv(Quu)@Qux)
        Vx.append(Qx-Qxu@k[-1])
        Vxx.append(Qxx-Qxu@K[-1])
    return k,K

def forwardpass(k,K,x,u):
    dx = np.zeros((dimx,T+1))
    du = np.zeros((dimu,T))
    for t in range (T):
        du[:,t:t+1]=-k[-1]-K[-1]@dx[:,t:t+1]
        dx[:,t+1:t+2] = next_state(x[:,t:t+1]+dx[:,t:t+1],u[:,t:t+1]+du[:,t:t+1])-x[:,t+1:t+2]
        k.pop()
        K.pop()
    x+=dx
    u+=du
    return x,u

def calcul_cout(x,u):
    sumcosts = 0.
    for t in range(0,T):
        sumcosts += cost(x[:,t:t+1],u[:,t:t+1])
    sumcosts += finalcost(x[:,T:T+1])
    return sumcosts

def scatter_x(x,cercle=False):
    plt.figure()
    figure, axes = plt.subplots()
    if (dimspace==2):
        if(cercle):
            draw_circle = plt.Circle((xsphere[0:1,:], xsphere[1:2,:]), Rsphere,fill=False)
            draw_circle2 = plt.Circle((xsphere[0:1,:], xsphere[1:2,:]), Rsphere+distsecu,fill=False)
            axes.add_artist(draw_circle)
            axes.add_artist(draw_circle2)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.scatter(x[0:1,:],x[1:2,:],s=3)
        plt.title("Trajectoire")
        plt.savefig("trajectoire.png")
    if(dimspace==1):
        plt.scatter(x[0:1,:],np.zeros((1,x[0:1,:].shape[1])))
        plt.title("Trajectoire")
    
def lines(x,u):
    t = np.linspace(0,T*dt,T+1)
    fig,(ax1,ax2,ax3)=plt.subplots(3,1,sharex=True)
    for k in range(dimspace):
        ax1.plot(t,x[k,:],label="x"+str(k+1))
        ax2.plot(t,x[dimspace+k,:],label="v"+str(k+1))
        ax3.plot(t[:-1],u[k,:],label="u"+str(k+1))
    ax1.legend()
    ax2.legend()
    ax3.legend()
    fig.suptitle("Courbes")
    plt.savefig("courbes.png",dpi=1000)

def calcul_cout_et_affichage(x,u):
    scatter_x(x,True)
    lines(x,u)

def ddp(x,u,k):
    for i in range(k):
        k,K = backwardpass(x,u)
        x,u = forwardpass(k,K,x,u)
        print(calcul_cout(x,u))
    return x,u

x,u=ddp(x,u,2)
calcul_cout_et_affichage(x,u)
