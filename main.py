import numpy as np
import matplotlib.pyplot as plt
plt.figure(dpi=1200)

from model import reward, stop_prob, value, grad_x, grad_y, primal_gap, primal_dual_gap
from model import grad_x_theta, grad_y_theta, softmax_param

def policy_2(theta, z1, z2):
    pi1 = softmax_param(theta)
    pi2_a1_equals_1 = softmax_param(z1)
    pi2_a1_equals_0 = softmax_param(z2)
    pi2 = pi2_a1_equals_1*pi1[0][0] + pi2_a1_equals_0*pi1[1][0]
    return pi2

def train(eta = 0.05, iterations = 5000):
    # res = list()
    # PrimalGaps = list()
    # PrimalDualGaps = list()
    rewards = list()

    theta = np.array([[0.], [0.]])
    z1 = np.array([[0.], [0.]])
    z2 = np.array([[0.], [0.]])

    theta_tmp = np.array([[0.], [0.]])
    z1_tmp = np.array([[0.], [0.]])
    z2_tmp = np.array([[0.], [0.]])

    for index in range(iterations):
        # print(index, theta_1)
        policy_2(theta, z1, z2)
        pi1 = softmax_param(theta)
        pi2 = policy_2(theta, z1, z2)
        rewards.append(value(pi1, pi2))

        # sequantial update
        theta_tmp[0][0] = theta[0][0] + eta*value(np.array([[1], [0]]), pi2)
        theta_tmp[1][0] = theta[1][0] + eta*value(np.array([[0], [1]]), pi2)

        z1_tmp[0][0] = z1[0][0] + eta*value(np.array([[1], [0]]), np.array([[1], [0]]))
        z1_tmp[1][0] = z1[1][0] + eta*value(np.array([[1], [0]]), np.array([[0], [1]]))

        z2_tmp[0][0] = z2[0][0] + eta*value(np.array([[0], [1]]), np.array([[1], [0]]))
        z2_tmp[1][0] = z2[1][0] + eta*value(np.array([[0], [1]]), np.array([[0], [1]]))

        theta = theta_tmp
        z1 = z1_tmp
        z2 = z2_tmp
    plt.cla()
    
    # plt.plot(np.arange(len(PrimalDualGaps)), PrimalDualGaps, color = 'lightblue', label = 'PD gap')
    # plt.plot(np.arange(len(PrimalGaps)), PrimalGaps, color = 'orange', label = 'x gap')
    plt.plot(np.arange(len(rewards)), rewards, color = 'orange', linewidth=3)

    
    
    
    rewards = list()

    theta_1 = np.array([[0.], [0.]])
    theta_2 = np.array([[0.], [0.]])

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
    
    # plt.plot(np.arange(len(PrimalDualGaps)), PrimalDualGaps, color = 'lightblue', label = 'PD gap')
    # plt.plot(np.arange(len(PrimalGaps)), PrimalGaps, color = 'orange', label = 'x gap')
    plt.plot(np.arange(len(rewards)), rewards, color = 'lightblue', linewidth=3)
    plt.legend(["Sequential policy updates", "Independent PG"])
    plt.xlabel("Iterations")
    plt.ylabel("Value")
    plt.title("stepsize=%.2f" % eta)
    plt.show()
    # plt.savefig("stepsize= %.2f.png" % eta)
    

train(eta = 0.05, iterations = 5000)



