import numpy as np
import matplotlib.pyplot as plt

from model import reward, stop_prob, value, grad_x, grad_y, primal_gap, primal_dual_gap
from archive.tools import project

res = list()
rewards = list()
# PrimalGaps = list()
# PrimalDualGaps = list()

def train(eta = 0.001, iterations = 2000, z_0 = [0.5, 0.5]):
    x = np.array([[z_0[0]], [1-z_0[0]]])
    y = np.array([[z_0[1]], [1-z_0[1]]])

    res.append(z_0)
    # PrimalGaps.append(primal_gap(x, y))
    # PrimalDualGaps.append(primal_dual_gap(x, y))


    for index in range(iterations):
        rewards.append(value(x, y))
        # print(index)
        x_tmp = project(x + eta*grad_x(x, y))
        y_tmp = project(y + eta*grad_y(x, y))
        if index == 1900:
            print(x_tmp)
            print(y_tmp)

        # res.append([x_tmp[0][0], y_tmp[0][0]])
        
        # PrimalGaps.append(primal_gap(x_tmp, y_tmp))
        # PrimalDualGaps.append(primal_dual_gap(x_tmp, y_tmp))

        x = x_tmp
        y = y_tmp

    plt.plot(np.arange(len(rewards)), rewards, color = 'lightblue', label = 'reward')
    # plt.plot(np.arange(len(PrimalGaps)), PrimalGaps, color = 'orange', label = 'x gap')
    

    # plt.legend(['PD gap','x gap'])
    
    # plt.yscale("log")
    
    plt.xlabel("Iterations")
    plt.ylabel("reward")
    plt.show()

train()


