#from scipy.io import loadmat
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# coupled differential equations with minimum and maximum bound
def fluid_ode_min_max(w, t, params):
    gamma,d,alpha, T, multiplier, B, w_min, w_max = params
    
    # Total number of links    
    no_of_links =len(T)
    # Total number of flows
    no_of_flows = len(T[0])
    gamma_queue = 1

    result = np.zeros(no_of_flows+no_of_links)
    result_new = np.zeros(no_of_flows+no_of_links)    
    
    p = np.array(w[no_of_flows:]).reshape(no_of_links, 1)
    mult_T_sum = np.sum(p*T,axis=0)
    mult_p_sum = np.sum(p*multiplier,axis=0)
        
    result[:no_of_flows] = gamma*(alpha - w[:no_of_flows]*mult_p_sum/(d+mult_T_sum))
    X = (w[:no_of_flows]/(d+mult_T_sum)).reshape(1,no_of_flows)
    result[no_of_flows:] = gamma_queue*((np.sum(X*multiplier,axis=1)-B)/B)
    
    for i in range(len(result)):
        result_new[i] = result[i]
        if(w[i]+result[i] < w_min[i]):
            result_new[i] = w_min[i]-w[i]
        if(w[i]+result[i] > w_max[i]):
            result_new[i] =  w_max[i]-w[i]
        
    return result_new
        
def get_rates(t, wsol,loss_model,gamma,d, labels):
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    for t1, w1 in zip(t, wsol):
        p = np.array(w1[no_of_flows:]).reshape(no_of_links, 1)
        mult_T_sum = np.sum(p*T,axis=0)
        X = w1[:no_of_flows]/(d+mult_T_sum)
        
        y1.append(X[0])
        y2.append(X[1])
        y3.append(X[2])
        y4.append(X[3])
    
    return [y1[-1], y2[-1], y3[-1], y4[-1]]


stoptime = 10000.0
numpoints = 20000

# Time instances at which the fluid model will be evaluated
 
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

abserr = 1.0e-6
relerr = 1.0e-6

# Scaling parameter
gamma = 0.1
# Price function models: choose between 'QLB' and 'TB'; For now no physical interpretation for 'QLB'
loss_model = 'DELAY'
    
T = np.array([[1, 0, 1,1],[1 ,1 ,0,0], [0,1,1,0],[0,0,0,1]])
multiplier = np.array([[1, 0, 1,0.048],[1 ,1 ,0,0], [0,1,1,0],[0,0,0,0.001]])

#T = np.array([[1, 0, 1,1],[1 ,1 ,0,0], [0,1,1,1]])
#multiplier = np.array([[1, 0, 1,0.02],[1 ,1 ,0,0], [0,1,1,0.025]])

#multiplier = np.array([[1, 0, 1,1],[1 ,1 ,0,0], [0,1,1,0],[0,0,0,1]])

#multiplier = np.array([[1, 0, 1,0.28],[1 ,1 ,0,0], [0,1,1,0],[0,0,0,0.6]])

#multiplier = np.array([[1, 0, 1,0.048],[1 ,1 ,0,0], [0,1,1,0],[0,0,0,0.5]])

#print("Printing traffic matrix")
#
#print(T*multiplier)        

        
# Total number of resources    
no_of_links =len(T)
# Total number of flows
no_of_flows = len(T[0])
# capacities of each resource
B = np.array([34,34,34,5])
#B = [34,34,34,200]
#B = [34,34,34,15]
#B = np.array([34,100,34])




#x_init = np.array([16.974917677377771, 8.974917677380578, 9.025082322620060, 100])
#x_init = np.array([2.72,31.2705,2.7295,594.6029])
d = np.array([0.1,0.1,0.1,0.1])
#p_init = np.array([0.02,0.01,0.01,0.01])
p_init = np.array([0.02,0.01,0.01,0.01])
#p_init = np.array([0.02,0.01,0.01])
#p_init = np.zeros(4)
T_init = d+p_init
#T_init = d

x_init = np.array([2.729531762836445,31.270442856530340,2.729531734476305,594.6028])
#x_init = np.array([17.99,16.113,8.499,375.481])

#x_init = np.array([10.36,23.63,10.36,13.273])
#x_init = np.array([2.729531762836445,31.270442856530340,2.729531734476305,594.6028])
#x_init = np.array([7.399998774625961,26.599979944758882,7.399998776009390,399.9999793981890])
#x_init = np.array([2.73,31.27,2.73,101.93])
#x_init = np.array([16.766,17.233,16.766,1.666])
w_init = x_init*T_init
w_init =np.array(list(w_init) + list(p_init))
#print(w_init)
# w values from global controller for each flow
alpha = np.array([1, 1, 1, 10])
#alpha = np.array([1, 1, 1, 1])


# Minimum rate requirement for the slices
w_min = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Maximum rate requirement for the slices
w_max = np.array([float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf")])
   
# Parameters for fluid model 
params = (gamma,d,alpha, T, multiplier,B, w_min, w_max)
# Differential equation solution evaluated at each element of t 
wsol = odeint(fluid_ode_min_max, w_init, t, args=(params,),atol=abserr, rtol=relerr)
#print(wsol)
#ids = [0,1,2,3]
labels = ['Flow 1', 'Flow 2', 'Flow 3','Flow 4 (Database)']

# Plot the rate as a function of time and return converged rate
x_converged = get_rates(t, wsol,loss_model,gamma,d, labels)
print("Printing Global Optimal Rates")
print(x_init)       
print("Printing Fluid model Rates (Delay)")     
print(x_converged)    