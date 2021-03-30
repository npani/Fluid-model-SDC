#from scipy.io import loadmat
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# coupled differential equations
def fluid_ode(x_n, t, params):
    w, T, T_flow, B, b, gamma,loss_model = params
    
    if(loss_model == 'QLB'):    
        p = get_link_congestion_prob_QLB(x_n,B,b,T)
    else:    
        p = get_link_congestion_prob_TB(x_n,B,b,T)
    P = get_slice_congestion_prob(p.reshape(no_of_links, 1), T_flow)   
    P = np.squeeze(P)
    result = gamma*(x_n*(((1-P)*w) - (P*x_n)))
    
    return result

# coupled differential equations with minimum and maximum bound
def fluid_ode_min_max(x_n, t, params):
    w, T, T_flow, B, b, gamma,loss_model,x_min, x_max = params            
    if(loss_model == 'QLB'):    
        p = get_link_congestion_prob_QLB(x_n,B,b,T)
    else:    
        p = get_link_congestion_prob_TB(x_n,B,b,T)
    P = get_slice_congestion_prob(p.reshape(no_of_links, 1), T_flow)   
    P = np.squeeze(P)
    result = gamma*(x_n*(((1-P)*w) - (P*x_n)))
    result_new = np.zeros(len(result))    
    for i in range(len(result)):
        result_new[i] = result[i]
        if(x_n[i]+result[i] < x_min[i]):
            result_new[i] = x_min[i]-x_n[i]
        if(x_n[i]+result[i] > x_max[i]):
            result_new[i] =  x_max[i]-x_n[i]
    
    return result_new

def get_slice_congestion_prob(p, T_flow):
    r = (1-p)*T_flow
    return 1 - np.prod(r, axis=0, where = r > 0, keepdims = True)

def get_link_congestion_prob_QLB(X,B,b,T):
    all_ones = np.transpose(0.99999*np.ones(len(B)))
    all_zeros = np.transpose(0.000001*np.ones(len(B)))
    result = np.minimum((np.sum(X*T,axis=1)/B)**(b*all_ones),all_ones)
    result = np.maximum(result,all_zeros)
    return result

def get_link_congestion_prob_TB(X,B,b,T):
    link_xs = np.sum(X*T,axis=1)
    for i in range(len(link_xs)):
        if (link_xs[i] < B[i]):
            link_xs[i] = 0
        else:
            link_xs[i] = (link_xs[i]-B[i])/link_xs[i]
            
    return link_xs

def plot_rates(t, xsol,loss_model,b, gamma,ids, topo):
    y1 = []
    y2 = []
    y3 = []
    for t1, x1 in zip(t, xsol):
        y1.append(x1[ids[0]])
        y2.append(x1[ids[1]])
        y3.append(x1[ids[2]])   
        
    plt.plot(t, y1, 'y', linewidth=2, label=labels[0])
    plt.plot(t, y2, 'r', linewidth=2, label=labels[1])
    plt.plot(t, y3, 'g', linewidth=2, label=labels[2])
    plt.grid('on')
    plt.xlabel('time',fontsize=18)
    plt.ylabel('x(t) (Mbps)',fontsize=18)
    if loss_model == 'QLB':
        plt.title(r'b = '+str(b)+', $\gamma$ = '+str(gamma),fontsize=18)
    else:
        plt.title(r'$\gamma$ = '+str(gamma),fontsize=18)

    plt.legend(loc='best')
    if topo == 'star':
        plt.ylim([0,32])
        plt.xlim([0,100])
    elif(topo == 'triangle'):    
        plt.ylim([0,12])
        plt.xlim([0,6000])
    else:
        plt.ylim([25,45])
        plt.xlim([0,100])
        
    plt.savefig('Price_model_'+loss_model+'.pdf')       
    plt.show()
    
    return x1


stoptime = 10000.0
numpoints = 20000

# Time instances at which the fluid model will be evaluated
 
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

abserr = 1.0e-6
relerr = 1.0e-6

# Queue length thereshold for queue based price function
b = 100
# Scaling parameter
gamma = 0.1
# Price function models: choose between 'QLB' and 'TB'
loss_model = 'TB'

# Link topologies
topo = 'star' # choose between 'star' and 'triangle'

# T = Traffic Matrix
# T_flow = Flow-to-link mapping; Note T is not equal to T_flow when a flow pass through the same link twice

if (topo == 'star'):   
    T = [[1, 0, 1],[1 ,1 ,0], [0,1,1]]
    T_flow = [[1, 0, 1],[1 ,1 ,0], [0,1,1]]
    labels = ['Slice 1', 'Slice 2', 'Slice 3']
    ids = [0,1,2]
elif (topo == 'triangle'):
    T = [[1, 1, 0, 0 ,1 ,1 ,1 ,2],[1 ,1 ,0 ,0 ,0 ,0, 1 ,0], [1, 1, 1 ,1 ,0 ,0, 0, 0], [0, 0 ,1 ,1 ,0, 0, 1, 0], [0, 0 ,1, 1 ,1 ,1 ,1, 0], [0 ,0 ,0 ,0, 1, 1, 0, 2]]
    T_flow = [[1, 1, 0, 0 ,1 ,1 ,1 ,1],[1 ,1 ,0 ,0 ,0 ,0, 1 ,0], [1, 1, 1 ,1 ,0 ,0, 0, 0], [0, 0 ,1 ,1 ,0, 0, 1, 0], [0, 0 ,1, 1 ,1 ,1 ,1, 0], [0 ,0 ,0 ,0, 1, 1, 0, 1]]
    labels = ['Slice 2', 'Slice 4', 'Slice 6']
    ids = [1,3,5]
else:    
    T = [[1],[1]]
    T_flow = [[1],[1]]
    labels = ['Slice 2']
    ids = [0]
            
        
# Total number of links    
no_of_links =len(T)
# Total number of flows
no_of_flows = len(T[0])
# Link capacities of each link
B = 34*np.transpose(np.ones(no_of_links))
# Optimal rates obtained from global controller
x_init = np.array([16.999859377747079, 16.999859377747022, 17.000140622252982])
# w values from global controller for each flow
w = np.array([1.0,1.0,1.0])

# Minimum rate requirement for the slices
x_min = np.array([0.0, 0.0, 0.0])

# Maximum rate requirement for the slices
x_max = np.array([float("inf"),float("inf"),float("inf")])


# Parameters for fluid model 
p = (w, np.array(T), np.array(T_flow), B, b, gamma, loss_model, x_min, x_max)
# Differential equation solution evaluated at each element of t 
xsol = odeint(fluid_ode_min_max, x_init, t, args=(p,),atol=abserr, rtol=relerr)

# Plot the rate as a function of time and return converged rate
x_converged = plot_rates(t, xsol,loss_model,b, gamma,ids, topo)    
print(x_converged)            
