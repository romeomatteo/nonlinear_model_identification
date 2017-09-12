from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from itertools import product
import os
# y -> outlet liquid temperature
# q(t) -> liquid flow rate
# Narx -> yhat(t+1|t) = f(y(t) ... y(t-3) u(t) ... u(t-9))

#read data
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


#AR_deg = 1
#X_deg = 1
poly_degree = 3

#prod = product([1,2,3,4],repeat = 2)
prod = product([4],[5,6,7,8,9,10])
for (AR_deg,X_deg) in prod:
    features = []
    for i in reversed(range(X_deg)):
        if i==0:
            features.append("u(t)")
        else:
            features.append("u(t-{})".format(i))
    for i in reversed(range(AR_deg)):
        if i==0:
            features.append("y(t)")
        else:
            features.append("y(t-{})".format(i))
    print(features)


    poly = PolynomialFeatures(poly_degree)

    Y = np.array(y_id)
    reg_u = np.full(X_deg,avg_u_id)
    reg_y = np.full(AR_deg,avg_y_id)
    PHI = []
    for i in tqdm(range(ID_LENGHT)):
        if i!=0:
            reg_y = np.append(reg_y, Y[i])[1:]
            reg_u = np.append(reg_u, u_id[i])[1:]
        regressors = np.append(reg_u, reg_y)
        PHI.append(poly.fit_transform([regressors])[0])
    PHI = np.array(PHI)
    regressor_terms = poly.get_feature_names(features)
    print("Regressors: ", regressor_terms)

    #FROE 2
    poly = PolynomialFeatures(poly_degree)

    Y_val = np.array(y_val)
    reg_u = np.full(X_deg,avg_u_val)
    reg_y = np.full(AR_deg,avg_y_val)
    PHI_val = []
    for i in tqdm(range(VAL_LENGHT)):
        if i!=0:
            reg_y = np.append(reg_y, Y_val[i])[1:]
            reg_u = np.append(reg_u, u_val[i])[1:]
        regressors = np.append(reg_u, reg_y)
        PHI_val.append(poly.fit_transform([regressors])[0])
    PHI_val = np.array(PHI_val)
    regressor_terms = poly.get_feature_names(features)

    A = np.zeros((PHI.shape[1],PHI.shape[1]))
    W = np.zeros(PHI.shape)
    g_hat = np.array([])
    np.fill_diagonal(A,1)
    regressor_selected = np.array([], dtype=int)
    err_sum = 0

    previous_MSE = 0
    for k in range(PHI.shape[1]):
        srr = np.array([])
        MSSEs = np.array([])
        g = np.array([])
        if k == 0:

            # 1.A
            for i in range(PHI.shape[1]):

                # 1.B
                W[:,0] = PHI[:,i]

                # 1.C
                g_i = np.dot(W[:,0],Y)/np.power(np.linalg.norm(W[:,0]),2)
                g = np.append(g, g_i)

                # SIMULATE - srr
                g_hat_temp = np.append(g_hat, g_i)
                theta_temp = np.zeros(len(g_hat_temp))
                for m in reversed(range(len(g_hat_temp))):
                    # 4.A
                    if m == len(g_hat_temp):
                        theta_temp[m] = g_hat_temp[m]

                    # 4.B
                    else:
                        temp = 0
                        for q in range(m+1, len(g_hat_temp)):
                            temp += A[m,q] * theta_temp[q]
                        theta_temp[m] = g_hat_temp[m] - temp
                #Simulate
                poly = PolynomialFeatures(poly_degree)
                regressor_selected_temp = np.append(regressor_selected, i)
                print("Considered regressors:")
                for el in regressor_selected_temp:
                    print(el)
                reg_y = np.full(AR_deg,0)
                reg_u = u_val[:X_deg]
                reg = np.append(reg_u,reg_y)
                reg = poly.fit_transform([reg])[0]
                model_reg = reg[regressor_selected_temp]
                y_hat_sim  = []
                try:
                    for q in range(ID_LENGHT - X_deg):
                        y_i = np.dot(model_reg,theta_temp) #simulated
                        y_hat_sim.append(y_i)
                        reg_y = np.append(reg_y, y_hat_sim[q])[1:]
                        reg_u = np.append(reg_u, u_id[X_deg+q])[1:]  #append at the end, then remove the first one([:1])
                        reg = np.append(reg_u,reg_y)
                        reg = poly.fit_transform([reg])[0]
                        model_reg = reg[regressor_selected_temp]
                    MSE_sim = mean_squared_error(Y[X_deg:],y_hat_sim)
                    MSSEs = np.append(MSSEs, MSE_sim)
                    srr_i = MSE_sim / np.power(np.linalg.norm(Y[X_deg:]),2)
                    print("Regressor n° {}, testing regressor {}, srr = {}, MSSE = {} , previous = {}".format(k+1, regressor_terms[i],\
                                                                                                      srr_i, MSE_sim, previous_MSE))
                    srr = np.append(srr, srr_i)
                except ValueError:
                    print("VALUE ERROR!")
                    MSSEs = np.append(MSSEs, 100000)
                    srr = np.append(srr, -100)


            # 1.E
            j = np.argmin(srr)
            previous_MSE = MSSEs[j]
            print(previous_MSE, len(srr), len(MSSEs))
            print("Srr min: {}, associated with regressors {}".format(srr[j], regressor_terms[j]))

            # 3
            if (srr[j] < 0):
                print('Srr became negative: {}'.format(srr[j]))
                break;
            # 1.F
            W[:,0] = PHI[:,j]

            # 1.G
            g_hat = np.append(g_hat, g[j])
            regressor_selected = np.append(regressor_selected, j)
            print("Regressor selected ", regressor_selected, k)

        else:

            # 2.A
            for l in range(PHI.shape[1]):
                if l not in regressor_selected:
                    temp = np.zeros(PHI.shape[0])

                    # 2.C
                    for i in range(k):
                        A[i,k] = (np.dot(W[:,i],PHI[:,l]))/np.power(np.linalg.norm(W[:,i]),2)
                        temp += A[i,k] * W[:,i]

                    # 2.B
                    W[:,k] = PHI[:,l] - temp

                    # 2.D
                    g_i = np.dot(W[:,k],Y)/np.power(np.linalg.norm(W[:,k]),2)
                    g = np.append(g, g_i)

                    # 2.E
                    # SIMULATE - srr
                    g_hat_temp = np.append(g_hat, g_i)
                    theta_temp = np.zeros(len(g_hat_temp))
                    for m in reversed(range(len(g_hat_temp))):
                        if m == len(g_hat_temp):
                            theta_temp[m] = g_hat_temp[m]
                        else:
                            temp = 0
                            for q in range(m+1, len(g_hat_temp)):
                                temp += A[m,q] * theta_temp[q]
                            theta_temp[m] = g_hat_temp[m] - temp
                    #Simulate
                    poly = PolynomialFeatures(poly_degree)
                    regressor_selected_temp = np.append(regressor_selected, l)
                    print("Considered regressors:")
                    for el in regressor_selected_temp:
                        print(el)
                    reg_y = np.full(AR_deg,0)
                    reg_u = u_val[:X_deg]
                    reg = np.append(reg_u,reg_y)
                    reg = poly.fit_transform([reg])[0]
                    model_reg = reg[regressor_selected_temp]

                    y_hat_sim  = []
                    try:
                        for i in range(ID_LENGHT - X_deg):
                            y_i = np.dot(model_reg,theta_temp) #simulated
                            y_hat_sim.append(y_i)
                            reg_y = np.append(reg_y, y_hat_sim[i])[1:]
                            reg_u = np.append(reg_u, u_id[X_deg+i])[1:]  #append at the end, then remove the first one([:1])
                            reg = np.append(reg_u,reg_y)
                            reg = poly.fit_transform([reg])[0]
                            model_reg = reg[regressor_selected_temp]
                        MSE_sim = mean_squared_error(Y[X_deg:],y_hat_sim)
                        MSSEs = np.append(MSSEs, MSE_sim)
                        srr_i = (previous_MSE - MSE_sim) / np.power(np.linalg.norm(Y[X_deg:]),2)
                        print("Regressor n° {}, testing regressor {}, srr = {}, MSSE = {}, previous = {}".format(k+1, regressor_terms[l], \
                                                                                                        srr_i, MSE_sim, previous_MSE))
                        srr = np.append(srr, srr_i)
                    except ValueError:
                        print("VALUE ERROR!")
                        MSSEs = np.append(MSSEs, 100000)
                        srr = np.append(srr, -100)
                else:
                    srr = np.append(srr, -100)
                    g = np.append(g, 0)
                    MSSEs = np.append(MSSEs, 100000)

            # 2.F
            j = np.argmax(srr)
            previous_MSE = MSSEs[j]
            print(previous_MSE, len(srr), len(MSSEs))
            print("Srr max: {}, associated with regressors {}".format(srr[j], regressor_terms[j]))

            # 3
            if (srr[j] < 0):
                print('Srr became negative: {}'.format(srr[j]))
                break;
            # 2.G
            for i in range(k):
                A[i,k] = (np.dot(W[:,i],PHI[:,j]))/np.power(np.linalg.norm(W[:,i]),2)
                temp += A[i,k] * W[:,i]
            W[:,k] = PHI[:,j] - temp

            # 2.H
            g_hat = np.append(g_hat, g[j])
            regressor_selected = np.append(regressor_selected, j)


    theta = np.zeros(len(g_hat))
    for i in reversed(range(len(g_hat))):

        # 4.A
        if i == len(g_hat):
            theta[i] = g_hat[i]

        # 4.B
        else:
            temp = 0
            for k in range(i+1, len(g_hat)):
                temp += A[i,k] * theta[k]
            theta[i] = g_hat[i] - temp

    directory =  "./FROE_plots_SEM/AR_{}_X_{}/DEG_{}".format(AR_deg, X_deg, poly_degree)
    if not os.path.exists(directory):
        os.makedirs(directory)

    PHI_final = np.zeros((PHI.shape[0], len(regressor_selected)))
    for i in range(len(regressor_selected)):
        PHI_final[:,i] = PHI[:, regressor_selected[i]]

    file_out = open("./FROE_plots_SEM/AR_{}_X_{}/DEG_{}/model.txt".format(AR_deg, X_deg, poly_degree), 'w')
    file_out.write("y = \n")
    for i in range(len(regressor_selected)):
        if i == 0:
            file_out.write("{}*{}\n".format(theta[i],regressor_terms[regressor_selected[i]]))
        else:
            if theta[i] < 0:
                file_out.write("{}*{}\n".format(theta[i],regressor_terms[regressor_selected[i]]))
            else:
                file_out.write("+{}*{}\n".format(theta[i],regressor_terms[regressor_selected[i]]))
    file_out.close()

    y_hat = np.dot(PHI_final, theta)

    plt.figure(figsize=(15,8))
    plt.subplot(311)
    plt.title("Identification")
    plt.plot(y_hat, color='blue', label = "Prediction")
    plt.legend()
    plt.subplot(312)
    plt.plot(y_id, color='red', label = "Process")
    plt.legend()
    plt.subplot(313)
    plt.plot(y_hat, color='blue', label = "Prediction")
    plt.plot(y_id, color='red', label = "Process")
    plt.legend()
    plt.savefig("./FROE_plots_SEM/AR_{}_X_{}/DEG_{}/plot_id.png".format(AR_deg, X_deg, poly_degree), transparent = False)
    plt.close()
    MSE_id = mean_squared_error(y_id,y_hat)

    PHI_final_val = np.zeros((PHI_val.shape[0], len(regressor_selected)))
    for i in range(len(regressor_selected)):
        PHI_final_val[:,i] = PHI_val[:, regressor_selected[i]]


    y_hat_val = np.dot(PHI_final_val, theta)

    plt.figure(figsize=(15,8))
    plt.subplot(311)
    plt.title("Validation")
    plt.plot(y_hat_val, color='blue', label = "Prediction")
    plt.legend()
    plt.subplot(312)
    plt.plot(y_val, color='red', label = "Process")
    plt.legend()
    plt.subplot(313)
    plt.plot(y_hat_val, color='blue', label = "Prediction")
    plt.plot(y_val, color='red', label = "Process")
    plt.legend()
    plt.savefig("./FROE_plots_SEM/AR_{}_X_{}/DEG_{}/plot_val.png".format(AR_deg, X_deg, poly_degree), transparent = False)
    plt.close()

    MSE_val = mean_squared_error(y_val,y_hat_val)
    print("MSE on validation: ", MSE_val)

    ## Simulation
    # start from initial phi, then build step by step each ne element
    poly = PolynomialFeatures(poly_degree)
    reg_y = np.full(AR_deg,0)
    reg_u = u_val[:X_deg]
    reg = np.append(reg_u,reg_y)
    reg = poly.fit_transform([reg])[0]
    model_reg = reg[regressor_selected]  # initial values for the regression

    #simulate the process
    y_hat_sim  = []
    try:
        for i in range(VAL_LENGHT - X_deg):
            y_i = np.dot(model_reg,theta) #simulated
            y_hat_sim.append(y_i)
            reg_y = np.append(reg_y, y_hat_sim[i])[1:]
            reg_u = np.append(reg_u, u_val[X_deg+i])[1:]  #append at beggining, then remove last one( [:-1])
            reg = np.append(reg_u,reg_y)
            reg = poly.fit_transform([reg])[0]
            model_reg = reg[regressor_selected]
    except ValueError as error:
        print("An error occur: ", error)

    plt.figure(figsize=(15,8))
    plt.subplot(311)
    plt.title("Simulation")
    plt.plot(y_hat_sim, color='blue', label = "Simulation")
    plt.legend()
    plt.subplot(312)
    plt.plot(y_val, color='red' , label = "Process")
    plt.legend()
    plt.subplot(313)
    plt.plot(y_hat_sim, color='blue', label = "Simulation")
    plt.plot(y_val, color='red' , label = "Process")
    plt.legend()
    plt.savefig("./FROE_plots_SEM/AR_{}_X_{}/DEG_{}/plot_sim.png".format(AR_deg, X_deg, poly_degree), transparent = False)
    plt.close()

    MSE_sim = 0
    try:
        MSE_sim = mean_squared_error(y_val[X_deg:],y_hat_sim)
    except:
        MSE_sim = "overflow"
        print("Exception in MSE computation - simulation")
    print("MSE on simulation: {}".format(MSE_sim))  


    file_out = open("./FROE_plots_SEM/AR_{}_X_{}/DEG_{}/mse.txt".format(AR_deg, X_deg, poly_degree), 'w')
    file_out.write("MSE identification: {}\n".format(MSE_id))
    file_out.write("MSE validation: {}\n".format(MSE_val))
    file_out.write("MSE simulation: {}\n".format(MSE_sim))
    file_out.close()
