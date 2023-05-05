import numpy as np
import matplotlib.pyplot as plt

from model import reward, stop_prob, value, grad_x, grad_y, primal_gap, primal_dual_gap
from model import grad_x_theta, grad_y_theta, softmax_param



def softmax_train(eta = 0.01, iterations = 20000):
    # res = list()
    # PrimalGaps = list()
    # PrimalDualGaps = list()
    rewards = list()

    theta_1 = np.array([[0.], [0.]])
    theta_2 = np.array([[0.], [5.]])

    # res.append(z_0)
    # PrimalGaps.append(primal_gap(x, y))
    # PrimalDualGaps.append(primal_dual_gap(x, y))


    for index in range(iterations):
        # print(index, theta_1)
        rewards.append(value(softmax_param(theta_1), softmax_param(theta_2)))
        theta1_tmp = theta_1 + eta*grad_x_theta(theta_1, theta_2)
        theta2_tmp = theta_2 + eta*grad_y_theta(theta_1, theta_2)

        # x_tmp = softmax_param(theta1_tmp)
        # y_tmp = softmax_param(theta2_tmp)

        # res.append([x_tmp[0][0], y_tmp[0][0]])
        # PrimalGaps.append(primal_gap(x_tmp, y_tmp))
        # PrimalDualGaps.append(primal_dual_gap(x_tmp, y_tmp))

        theta_1 = theta1_tmp
        theta_2 = theta2_tmp
    plt.cla()
    
    # plt.plot(np.arange(len(PrimalDualGaps)), PrimalDualGaps, color = 'lightblue', label = 'PD gap')
    # plt.plot(np.arange(len(PrimalGaps)), PrimalGaps, color = 'orange', label = 'x gap')
    plt.plot(np.arange(len(rewards)), rewards, color = 'lightblue', label = 'rewards')
    # plt.title("Independent PG")
    # plt.legend(["eta=%.3f" % eta,])
    
    # # plt.yscale("log")
    # # plt.xscale("log")
    
    # plt.xlabel("Iterations")
    # plt.ylabel("Value")
    # plt.show()
    # plt.savefig("C:/Users/zhaoy/Desktop/softmax/eta=%.2f.png" % eta)

# for eta in np.linspace(20.0, 50.0, 601):
#     print(eta)
#     train(eta=eta)



