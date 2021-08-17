#from scipy.io import loadmat
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# coupled differential equations with minimum and maximum bound
def fluid_ode_min_max(w, t, params):
    gamma,d,alpha, T, multiplier, B, w_min, w_max = params
    result = np.ones(8)
    result_new = np.zeros(len(result))    
    p1 = w[4]
    p2 = w[5]
    p3 = w[6]
    p4 = w[7]
    
    flow_rate1 = w[0]/(d[0]+(p1+p2))
    flow_rate2 = w[1]/(d[1]+(p2+p3))
    flow_rate3 = w[2]/(d[2]+(p1+p3))
    flow_rate4 = w[3]/(d[3]+(2*p1+p4))
    
    # Window update rule
    result[0] = gamma*(alpha[0] - (flow_rate1*(p1+p2)))
    result[1] = gamma*(alpha[1] - (flow_rate2*(p2+p3)))
    result[2] = gamma*(alpha[2] - (flow_rate3*(p1+p3)))
    result[3] = gamma*(alpha[3] - (flow_rate4*(2*p1*multiplier[0][3]+p4*multiplier[3][3])))
    
    gamma_queue = 1
    # Queueing delay update rule
    
    arrival_rate1=(flow_rate1*multiplier[0][0])+(flow_rate3*multiplier[0][2])+(flow_rate4*multiplier[0][3])
    arrival_rate2=(flow_rate1*multiplier[1][0])+(flow_rate2*multiplier[1][1])
    arrival_rate3=(flow_rate2*multiplier[2][1])+(flow_rate3*multiplier[2][2])
    arrival_rate4=(flow_rate4*multiplier[3][3])
    
    result[4] =  gamma_queue*((arrival_rate1-B[0])/(B[0]))
    result[5] =  gamma_queue*((arrival_rate2-B[1])/(B[1]))
    result[6] =  gamma_queue*((arrival_rate3-B[2])/(B[2]))
    result[7] =  gamma_queue*((arrival_rate4-B[3])/(B[3]))
    
    for i in range(len(result)):
        result_new[i] = result[i]
        if(w[i]+result[i] < w_min[i]):
            result_new[i] = w_min[i]-w[i]
        if(w[i]+result[i] > w_max[i]):
            result_new[i] =  w_max[i]-w[i]
        
    return result_new
        
def plot_rates(t, wsol,loss_model,gamma,d, labels):
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []
    y6 = []
    y7 = []
    y8 = []
    
    arriv_rates = []
    for t1, w1 in zip(t, wsol):
        p1 = w1[4]
        p2 = w1[5]
        p3 = w1[6]
        p4 = w1[7]
        
        flow_rate1 = w1[0]/(d[0]+(p1+p2))
        flow_rate2 = w1[1]/(d[1]+(p2+p3))
        flow_rate3 = w1[2]/(d[2]+(p1+p3))
        flow_rate4 = w1[3]/(d[3]+(2*p1+p4))
        
        y1.append(flow_rate1)
        y2.append(flow_rate2)
        y3.append(flow_rate3)
        y4.append(flow_rate4)
        
        y5.append(p1)
        y6.append(p2)
        y7.append(p3)
        y8.append(p4)
    arriv_rates.append([flow_rate1+flow_rate3+(flow_rate4*0.048), flow_rate1+flow_rate2, flow_rate2+flow_rate3, (flow_rate4*0.001)])    
#        
    plt.plot(t, y1, 'y', linewidth=2, label=labels[0])
    plt.plot(t, y2, 'r', linewidth=2, label=labels[1])
    plt.plot(t, y3, 'g', linewidth=2, label=labels[2])
    plt.plot(t, y4, 'b', linewidth=2, label=labels[3])
    plt.grid('on')
    plt.xlabel('time',fontsize=18)
    plt.ylabel('x(t)',fontsize=18)
#    
    plt.title(r'$\gamma$ = '+str(gamma),fontsize=18)
    plt.legend(loc='best')  
    plt.xlim([0,200])
    
    plt.savefig('Price_model_'+loss_model+'_gamma_point1.pdf')       
#    plt.show()
    
    return [y1[-1], y2[-1], y3[-1], y4[-1]], [y5[-1], y6[-1], y7[-1], y8[-1]], arriv_rates


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
#multiplier = np.array([[1, 0, 1,0.048],[1 ,1 ,0,0], [0,1,1,0],[0,0,0,0.001]])
#multiplier = np.array([[1, 0, 1,0.28],[1 ,1 ,0,0], [0,1,1,0],[0,0,0,0.6]])

multiplier = np.array([[1, 0, 1,0.048],[1 ,1 ,0,0], [0,1,1,0],[0,0,0,0.5]])

#print("Printing traffic matrix")
#
#print(T*multiplier)        

        
# Total number of resources    
no_of_links =len(T)
# Total number of flows
no_of_flows = len(T[0])
# capacities of each resource
#B = [34,34,34,1]
B = [34,34,34,200]


#x_init = np.array([16.974917677377771, 8.974917677380578, 9.025082322620060, 100])
#x_init = np.array([2.72,31.2705,2.7295,594.6029])
d = np.array([0.1,0.1,0.1,0.1])
#p_init = np.array([0.02,0.01,0.01,0.01])
p_init = np.array([0.02,0.01,0.01,0.01])
#p_init = np.zeros(4)
T_init = d+p_init
#x_init = np.array([2.729531762836445,31.270442856530340,2.729531734476305,594.6028])
x_init = np.array([7.399998774625961,26.599979944758882,7.399998776009390,399.9999793981890])
#x_init = np.array([2.73,31.27,2.73,101.93])
#x_init = np.array([16.766,17.233,16.766,1.666])
w_init = x_init*T_init
w_init =np.array(list(w_init) + list(p_init))
#print(w_init)
# w values from global controller for each flow
alpha = np.array([1, 1, 1, 10])

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
x_converged, p_converged, arriv_rates = plot_rates(t, wsol,loss_model,gamma,d, labels)
print("Printing Global Optimal Rates")
print(x_init)       
print("Printing Fluid model Rates (Delay)")     
print(x_converged)   
print("Printing Arrival Rates")     
print(arriv_rates)         