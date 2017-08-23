from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


data = open('../exchanger/exchanger.dat')
lst = []
for line in data:
    lst += [line.split()]
time = [float(x[0]) for x in lst]
u = [float(x[1]) for x in lst]
y = [float(x[2]) for x in lst]
avg_y = np.mean(y)
avg_u = np.mean(u)

#divide identification set and validation set
ID_LENGHT = 3000
VAL_LENGHT = 1000
time_id = time[:ID_LENGHT]
u_id = u[:ID_LENGHT]
y_id = y[:ID_LENGHT]
avg_y_id = np.mean(y_id)
avg_u_id = np.mean(u_id)

time_val = time[ID_LENGHT:]
u_val = u[ID_LENGHT:]
y_val = y[ID_LENGHT:]
avg_y_val = np.mean(y_val)
avg_u_val = np.mean(u_val)


AR_deg = 4
X_deg = 10
hidden_max = 2
unit_max = 15
Y = np.array(y_id)
threshold_loss = 0.5

combinations = []
random_states = []
structures = []


for j in range(hidden_max):
    combinations = list(product(list(range(1,unit_max+1)), repeat = j+1))
    print(combinations)
    for comb in combinations:
        c = 0
        states = []
        for k in range(50):
            mlp = MLPRegressor(hidden_layer_sizes=comb,activation="logistic", solver="lbfgs",  \
                               verbose=False, random_state=k)
            reg_u = np.full(X_deg,avg_u_id)
            reg_y = np.full(AR_deg,avg_y_id)
            PHI = []
            for i in range(ID_LENGHT):
                if i!=0:
                    reg_y = np.append(reg_y, Y[i])[1:]
                    reg_u = np.append(reg_u, u_id[i])[1:]
                regressors = np.append(reg_u, reg_y)
                PHI.append(regressors)
            PHI = np.array(PHI)
            model = mlp.fit(PHI,Y)
            if model.loss_ < threshold_loss:
                print("Random_state: {}, Structure: {}, loss: {}".format(k, comb, model.loss_))
                states.append(k)
        random_states.append(states)
        structures.append(comb)
        c=c+1
#PHI identification

file_out = open('structures.txt', 'w')
for item in structures:
  file_out.write("{}\n".format(item))
file_out.close()

file_out = open('random_states.txt', 'w')
for item in random_states:
  file_out.write("{}\n".format(item))
file_out.close()

Y = np.array(y_id)
reg_u = np.full(X_deg,avg_u_id)
reg_y = np.full(AR_deg,avg_y_id)
PHI = []
for i in range(ID_LENGHT):
    if i!=0:
        reg_y = np.append(reg_y, Y[i])[1:]
        reg_u = np.append(reg_u, u_id[i])[1:]
    regressors = np.append(reg_u, reg_y)
    PHI.append(regressors)
PHI = np.array(PHI)

#PHI validation
Y_val = np.array(y_val)
reg_u = np.full(X_deg,avg_u_val)
reg_y = np.full(AR_deg,avg_y_val)
PHI_val = []
for i in tqdm(range(VAL_LENGHT)):
    if i!=0:
        reg_y = np.append(reg_y, Y_val[i])[1:]
        reg_u = np.append(reg_u, u_val[i])[1:]
    regressors = np.append(reg_u, reg_y)
    PHI_val.append(regressors)
PHI_val = np.array(PHI_val)

MSE_val = []
MSE_id = []
MSE_sim = []
for el in range(len(structures)):
    for state in random_states[el]:
        print("Structure: {}, random_state: {}".format(structures[el], state))
        mlp = MLPRegressor(hidden_layer_sizes=structures[el],activation="logistic", solver="lbfgs", verbose=True, random_state=state)
        model = mlp.fit(PHI,Y)
        y_hat = model.predict(PHI)

        MSE_id.append(mean_squared_error(y_id,y_hat))
        print("MSE on identification: ", MSE_id[-1])

        #MODEL VALIDATION
        y_hat_val = model.predict(PHI_val)
        MSE_val.append(mean_squared_error(y_val,y_hat_val))
        print("MSE on validation: ", MSE_val[-1])

        #SIMULATION
        # start from initial phi, then build step by step each ne element
        reg_y = np.full(AR_deg,0)
        reg_u = np.full(X_deg,0)
        reg = np.append(reg_u,reg_y)
        #simulate the process
        y_hat_sim  = []
        for i in range(VAL_LENGHT + ID_LENGHT):
            y_i = model.predict([reg]) #simulated
            y_hat_sim.append(y_i)
            reg_y = np.append(reg_y, y_hat_sim[i])[1:]
            reg_u = np.append(reg_u, u[i])[1:]  #append at beggining, then remove last one( [:-1])
            reg = np.append(reg_u,reg_y)
        MSE_sim.append(mean_squared_error(y,y_hat_sim))
        print("MSE on simulation: ", MSE_sim[-1])

MSE_id
file_out = open('mse_id.txt', 'w')
for item in MSE_id:
  file_out.write("{}\n".format(item))
file_out.close()

MSE_val
file_out = open('mse_val.txt', 'w')
for item in MSE_val:
  file_out.write("{}\n".format(item))
file_out.close()

MSE_sim
file_out = open('mse_sim.txt', 'w')
for item in MSE_sim:
  file_out.write("{}\n".format(item))
file_out.close()

MSE_sim_min = np.argmin(MSE_sim)
MSE_sim_max = np.argmax(MSE_sim)

i = 0
for el in range(len(structures)):
    for state in random_states[el]:
        if MSE_sim_min == i:
            st_min = structures[el]
            rs_min = state
        if MSE_sim_max == i:
            st_max = structures[el]
            rs_max = state
        i+=1

def analyze(st, rs, sim):
    mlp = MLPRegressor(hidden_layer_sizes=st,activation="logistic", solver="lbfgs",  \
                       verbose=True, random_state=rs)
    model = mlp.fit(PHI,Y)
    y_hat = model.predict(PHI)
    y_hat_val = model.predict(PHI_val)

    #SIMULATION
    # start from initial phi, then build step by step each ne element
    reg_y = np.full(AR_deg,0)
    reg_u = np.full(X_deg,0)
    reg = np.append(reg_u,reg_y)
    #simulate the process
    y_hat_sim  = []
    for i in range(VAL_LENGHT + ID_LENGHT):
        y_i = model.predict([reg]) #simulated
        y_hat_sim.append(y_i)
        reg_y = np.append(reg_y, y_hat_sim[i])[1:]
        reg_u = np.append(reg_u, u[i])[1:]  #append at beggining, then remove last one( [:-1])
        reg = np.append(reg_u,reg_y)

    #PLOT identification
    plt.figure(figsize=(15,8))
    plt.subplot(311)
    plt.plot(y_hat, color='blue')
    plt.subplot(312)
    plt.plot(y_id, color='red')
    plt.subplot(313)
    plt.plot(y_hat, color='blue')
    plt.plot(y_id, color='red')
    plt.savefig("plot_id_{}_{}.png".format(sim, st), transparent = False)
    #plt.show()

    #PLOT validation
    plt.figure(figsize=(15,8))
    plt.subplot(311)
    plt.plot(y_hat_val, color='blue')
    plt.subplot(312)
    plt.plot(y_val, color='red')
    plt.subplot(313)
    plt.plot(y_hat_val, color='blue')
    plt.plot(y_val, color='red')
    plt.savefig("plot_val_{}_{}.png".format(sim, st), transparent = False)
    #plt.show()
    #MODEL VALIDATION - CORRELATION FUNCTIONS
    from statsmodels.tsa.stattools import acf , ccf
    epsilon = np.array(y_val - y_hat_val)
    #Autocorrelation epsilon
    corr_ee = acf(epsilon)
    #Cross-correlation u-epsilon
    corr_ue = ccf(u_val, epsilon,unbiased=False)
    #Cross-correlation epsilon ( epsilon*u)
    corr_e_eu = ccf(epsilon,np.multiply(epsilon[1:],u_val[1:]),unbiased=False)
    #Cross-correlation delta(u^2)-epsilon
    corr_du2_e = ccf(np.power(u_val,2) - np.mean(np.power(u_val,2)),epsilon, unbiased=False)
    #Cross-correlation delta(u^2)-epsilon
    corr_du2_e2 = ccf(np.power(u_val,2) - np.mean(np.power(u_val,2)),np.power(epsilon,2), unbiased=False)
    #confidence interval   -95%
    conf_interval_sup = 1.96 / np.sqrt(VAL_LENGHT)
    conf_interval_inf = -1.96 / np.sqrt(VAL_LENGHT)
    #Diagrams plot
    plt.figure(figsize=(15,8))
    plt.subplot(231)
    plt.title(r'$\phi_{\xi\xi}(\tau)$', fontsize=30)
    plt.axhline(y=conf_interval_sup, color = "red")
    plt.axhline(y=conf_interval_inf, color = "red")
    plt.plot(corr_ee)
    plt.ylim((-1,1))
    plt.subplot(232)
    plt.title(r'$\phi_{\xi(\xi u)}(\tau)$', fontsize=30)
    plt.axhline(y=conf_interval_sup, color = "red")
    plt.axhline(y=conf_interval_inf, color = "red")
    plt.plot(corr_e_eu)
    plt.ylim((-1,1))
    plt.subplot(234)
    plt.title(r'$\phi_{u \xi}(\tau)$', fontsize=30)
    plt.axhline(y=conf_interval_sup, color = "red")
    plt.axhline(y=conf_interval_inf, color = "red")
    plt.plot(corr_ue)
    plt.ylim((-1,1))
    plt.subplot(235)
    plt.title(r'$\phi_{u^2\xi}(\tau)$', fontsize=30)
    plt.axhline(y=conf_interval_sup, color = "red")
    plt.axhline(y=conf_interval_inf, color = "red")
    plt.plot(corr_du2_e)
    plt.ylim((-1,1))
    plt.subplot(236)
    plt.title(r'$\phi_{u^2\xi^2}(\tau)$', fontsize=30)
    plt.axhline(y=conf_interval_sup, color = "red")
    plt.axhline(y=conf_interval_inf, color = "red")
    plt.plot(corr_du2_e2)
    plt.ylim((-1,1))
    plt.savefig("plot_correlation_tests_{}_{}.png".format(sim, st), transparent = False)
    #plt.show()


    plt.figure(figsize=(15,8))
    plt.subplot(311)
    plt.plot(y_hat_sim, color='blue')
    plt.subplot(312)
    plt.plot(y, color='red')
    plt.subplot(313)
    plt.plot(y_hat_sim, color='blue')
    plt.plot(y, color='red')
    plt.savefig("plot_sim_{}_{}.png".format(sim,st), transparent = False)
    #plt.show()

print("Lower MSE simulation:")
print("Structure: {}, Random_state: {}".format(st_min,rs_min))
print("Higher MSE simulation:")
print("Structure: {}, Random_state: {}".format(st_max,rs_max))

u_val = np.array(u_val)
analyze(st_min,rs_min,"min")
analyze(st_max,rs_max,"max")
