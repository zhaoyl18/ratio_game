import numpy as np
import matplotlib.pyplot as plt

from model import reward, stop_prob, value, grad_x, grad_y, primal_gap, primal_dual_gap
from model import grad_x_theta, grad_y_theta, softmax_param, fisher_matrix



def train(eta = 0.01, iterations = 1000, z_0 = [1, 1]):
    res = list()
    PrimalGaps = list()
    PrimalDualGaps = list()

    x = np.array([[z_0[0]], [1-z_0[0]]])
    y = np.array([[z_0[1]], [1-z_0[1]]])

    theta_1 = np.array([[5], [0]])
    theta_2 = np.array([[5], [0]])

    res.append(z_0)
    PrimalGaps.append(primal_gap(x, y))
    PrimalDualGaps.append(primal_dual_gap(x, y))


    for index in range(iterations):
        # print(index, theta_1)
        theta1_tmp = theta_1 - eta*np.dot(np.linalg.pinv(fisher_matrix(theta_1)), grad_x_theta(theta_1, theta_2))
        theta2_tmp = theta_2 + eta*np.dot(np.linalg.pinv(fisher_matrix(theta_2)), grad_y_theta(theta_1, theta_2))

        x_tmp = softmax_param(theta1_tmp)
        y_tmp = softmax_param(theta2_tmp)

        res.append([x_tmp[0][0], y_tmp[0][0]])
        PrimalGaps.append(primal_gap(x_tmp, y_tmp))
        PrimalDualGaps.append(primal_dual_gap(x_tmp, y_tmp))

        theta_1 = theta1_tmp
        theta_2 = theta2_tmp
    plt.cla()
    
    plt.plot(np.arange(len(PrimalDualGaps)), PrimalDualGaps, color = 'lightblue', label = 'PD gap')
    plt.plot(np.arange(len(PrimalGaps)), PrimalGaps, color = 'orange', label = 'x gap')
    

    plt.legend(['PD gap','x gap'])
    
    plt.yscale("log")
    plt.xscale("log")
    
    plt.xlabel("Iterations")
    plt.ylabel("Gap")
    # plt.show()
    plt.savefig("C:/Users/zhaoy/Desktop/natural_loglog/eta=%.2f.png" % eta)

for eta in np.linspace(0.0, 20.0, 401):
    print(eta)
    train(eta=eta)

# train(eta=1)

