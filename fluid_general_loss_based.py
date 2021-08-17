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
    w, T, multiplier, B, b, gamma,loss_model,x_min, x_max = params
            
    if(loss_model == 'QLB'):    
        p = get_link_congestion_prob_QLB(x_n,B,b,T*multiplier)
    else:    
        p = get_link_congestion_prob_TB(x_n,B,b,T*multiplier)
    P = get_slice_congestion_prob(p.reshape(no_of_links, 1), T)   
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

def get_goodput(t, xsol,params,topo):
    w, T, T_flow, B, b, gamma,loss_model,x_min, x_max = params
    xsol_goodput = []
    for t1, x1 in zip(t, xsol):        
        if(loss_model == 'QLB'):    
            p = get_link_congestion_prob_QLB(x1,B,b,T)
        else:    
            p = get_link_congestion_prob_TB(x1,B,b,T)
        P = get_slice_congestion_prob(p.reshape(no_of_links, 1), T_flow)   
        P = np.squeeze(P)
        
        goodput = (1-P)*x1
        xsol_goodput.append(goodput)      
        
    return xsol_goodput
        
def plot_rates(t, xsol,loss_model,b, gamma,ids):
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    for t1, x1 in zip(t, xsol):
        y1.append(x1[ids[0]])
        y2.append(x1[ids[1]])
        y3.append(x1[ids[2]])
        y4.append(x1[ids[3]])
        
    plt.plot(t, y1, 'y', linewidth=2, label=labels[0])
    plt.plot(t, y2, 'r', linewidth=2, label=labels[1])
    plt.plot(t, y3, 'g', linewidth=2, label=labels[2])
    plt.plot(t, y4, 'b', linewidth=2, label=labels[3])
    plt.grid('on')
    plt.xlabel('time',fontsize=18)
    plt.ylabel('x(t) (Mbps)',fontsize=18)
    if loss_model == 'QLB':
        plt.title(r'b = '+str(b)+', $\gamma$ = '+str(gamma),fontsize=18)
    else:
        plt.title(r'$\gamma$ = '+str(gamma),fontsize=18)

    plt.legend(loc='best')  
#    if topo == 'star':
#        plt.ylim([0,32])
#        plt.xlim([0,100])
#    elif(topo == 'triangle'):    
#        plt.ylim([0,12])
#        plt.xlim([0,6000])
#    else:
#        plt.ylim([25,45])
#        plt.xlim([0,100])
        
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
# Price function models: choose between 'QLB' and 'TB'; For now no physical interpretation for 'QLB'
loss_model = 'TB'

# Flow to resource mapping: 3 communication + 1 data base flow
T = [[1, 0, 1,1],[1 ,1 ,0,0], [0,1,1,0],[0,0,0,1]]

# Multiplier for each flow along each resource
multiplier = [[1, 0, 1,0.048],[1 ,1 ,0,0], [0,1,1,0],[0,0,0,0.001]]
        
# Total number of resources: 3 communication + 1 processing    
no_of_links =len(T)
# Total number of flows
no_of_flows = len(T[0])
# capacities of each resource: 3 communication + 1 processing   
B = [34,34,34,5]

# Rates from global controller: 3 communication + 1 data base flow rate
x_init = np.array([16.974917677377771, 8.974917677380578, 9.025082322620060, 100])
# w values from global controller for each flow
w = np.array([1, 1, 1, 7.999999999999885])

# Minimum rate requirement for the slices
x_min = np.array([0.0, 0.0, 0.0, 0.0])

# Maximum rate requirement for the slices
x_max = np.array([float("inf"),float("inf"),float("inf"),float("inf")])
   
# Parameters for fluid model 
p = (w, np.array(T), np.array(multiplier), B, b, gamma, loss_model, x_min, x_max)
# Differential equation solution evaluated at each element of t 
xsol = odeint(fluid_ode_min_max, x_init, t, args=(p,),atol=abserr, rtol=relerr)
ids = [0,1,2,3]
labels = ['Slice 1', 'Slice 2', 'Slice 3','db flow 1']

# Plot the rate as a function of time and return converged rate
x_converged = plot_rates(t, xsol,loss_model,b, gamma,ids)
print(x_converged)            
