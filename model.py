import numpy as np
import os
import matplotlib.pyplot as plt

def reward(epsilon = 0.3): # defalut epsilon = 0.1
    R = np.array([[1, 0.5], [-0.5, 0]])
    # R = np.array([[1, epsilon], [-epsilon, 0]])
    return R

def stop_prob(s = 0.3): # defalut s = 0.3
    S = np.array([[1, 1], [0.1, 0.1]])
    # S = np.array([[s, s], [1, 1]])
    return S

def value(pi_1, pi_2):
    nume = np.dot(pi_1.T, np.dot(reward(), pi_2))[0][0]
    denom = np.dot(pi_1.T, np.dot(stop_prob(), pi_2))[0][0]
    value = nume/denom
    return value

def softmax_param(theta = np.array([[100],[0]])):
    theta_1 = theta[0][0]
    theta_2 = theta[1][0]
    policy = np.zeros((2,1))
    if theta_1 - theta_2 > 100:
        policy[0][0] = 1.0
        policy[1][0] = 0.0
        return policy
    if theta_2 - theta_1 > 100:
        policy[0][0] = 0.0
        policy[1][0] = 1.0
        return policy
    log_sum = np.logaddexp(theta_1, theta_2)
    policy[0][0] = np.exp(theta_1)/(np.exp(log_sum))
    policy[1][0] = 1.0 - policy[0][0]

    return policy

def grad_x(pi_1, pi_2):
    denom = np.square(np.dot(pi_1.T, np.dot(stop_prob(), pi_2))[0][0])
    nume = np.dot(reward(), pi_2)*np.dot(pi_1.T, np.dot(stop_prob(), pi_2))[0][0]\
            - np.dot(stop_prob(), pi_2)*np.dot(pi_1.T, np.dot(reward(), pi_2))[0][0]
    grad = nume/denom
    return grad

def grad_y(pi_1, pi_2):
    denom = np.square(np.dot(pi_1.T, np.dot(stop_prob(), pi_2))[0][0])
    nume = np.dot(reward().T, pi_1)*np.dot(pi_1.T, np.dot(stop_prob(), pi_2))[0][0]\
            - np.dot(stop_prob().T, pi_1)*np.dot(pi_1.T, np.dot(reward(), pi_2))[0][0]
    grad = nume/denom
    return grad

def derivative_theta_1(theta):
    theta_1 = theta[0][0]
    theta_2 = theta[1][0]

    dx_dtheta_1 = np.zeros((2,1))
    dx_dtheta_1[0][0] = (np.exp(theta_1) * np.exp(theta_2))/np.square(np.exp(theta_1)+np.exp(theta_2))
    dx_dtheta_1[1][0] = - dx_dtheta_1[0][0]
    return dx_dtheta_1

def derivative_theta_2(theta):
    theta_1 = theta[0][0]
    theta_2 = theta[1][0]

    dx_dtheta_2 = np.zeros((2,1))
    dx_dtheta_2[0][0] = -(np.exp(theta_1) * np.exp(theta_2))/np.square(np.exp(theta_1)+np.exp(theta_2))
    dx_dtheta_2[1][0] = -dx_dtheta_2[0][0]
    return dx_dtheta_2

def fisher_matrix(theta):
    pi_a1 = softmax_param(theta)[0][0]
    pi_a2 = softmax_param(theta)[1][0]

    fisher = np.zeros((2,2))

    #compute nabla log and fisher matrix
    nabla_1 = (1/pi_a1) * derivative_theta_1(theta)
    M1 = np.dot(nabla_1, nabla_1.T)
    nabla_2 = (1/pi_a2) * derivative_theta_2(theta)
    M2 = np.dot(nabla_2, nabla_2.T)

    fisher = pi_a1 * M1 + pi_a2 * M2
    return fisher

def grad_x_theta(theta_1, theta_2):
    pi_1 = softmax_param(theta_1)
    pi_2 = softmax_param(theta_2)

    par_Vx_theta = np.zeros((2,1))

    denom = np.square(np.dot(pi_1.T, np.dot(stop_prob(), pi_2))[0][0])
    nume_0 = np.dot(reward(), pi_2)*np.dot(pi_1.T, np.dot(stop_prob(), pi_2))[0][0]\
            - np.dot(stop_prob(), pi_2)*np.dot(pi_1.T, np.dot(reward(), pi_2))[0][0]
    nume = np.dot(derivative_theta_1(theta = theta_1).T, nume_0)[0][0]
    par_Vx_theta[0][0] = nume/denom

    nume = np.dot(derivative_theta_2(theta = theta_1).T, nume_0)[0][0]
    par_Vx_theta[1][0] = nume/denom

    return par_Vx_theta

def grad_y_theta(theta_1, theta_2):
    pi_1 = softmax_param(theta_1)
    pi_2 = softmax_param(theta_2)

    par_Vy_theta = np.zeros((2,1))

    denom = np.square(np.dot(pi_1.T, np.dot(stop_prob(), pi_2))[0][0])
    nume_0 = np.dot(reward().T, pi_1)*np.dot(pi_1.T, np.dot(stop_prob(), pi_2))[0][0]\
            - np.dot(stop_prob().T, pi_1)*np.dot(pi_1.T, np.dot(reward(), pi_2))[0][0]
    nume = np.dot(derivative_theta_1(theta = theta_2).T, nume_0)[0][0]
    par_Vy_theta[0][0] = nume/denom

    nume = np.dot(derivative_theta_2(theta = theta_2).T, nume_0)[0][0]
    par_Vy_theta[1][0] = nume/denom

    return par_Vy_theta

def primal_gap(pi_1, pi_2):
    # return \max_{y^\prime} V(x^i, y^\prime) - V(x^*, y^*)
    # in this special case, V((x,1-x), (y, 1-y)) = \frac{0.1x - 0.1y -xy}{1-0.7x}
    return (0.1*pi_1[0][0])/(1 - 0.7*pi_1[0][0])

def primal_dual_gap(pi_1, pi_2):
    # return \max_{y^\prime} V(x^i, y^\prime) - V(x^*, y^*) - \min_{x^\prime} V(x^\prime. y^i)
    # in this special case, V((x,1-x), (y, 1-y)) = \frac{0.1x - 0.1y -xy}{1-0.7x}

    max_y = (0.1*pi_1[0][0])/(1 - 0.7*pi_1[0][0])
    min_x = 0
    if pi_2[0][0] < 10/107:
        min_x = (-0.1*pi_1[0][0])/(1)
    elif pi_2[0][0] > 10/107:
        min_x = (0.1-1.1*pi_1[0][0])/(0.3)
    return  max_y - min_x 


def MVI():
    n = 256
    X = np.linspace(0, 1, n, endpoint=True)
    Y = np.true_divide(-0.1*X, -0.7*np.square(X) - 0.14*X + 0.1)
    Y2 = np.ones((len(X),))

    # fig = plt.figure(figsize=(1, 1))

    # plt.plot([0, 0, 1, 1],[0, 1, 1, 0])
    plt.fill_between(X, Y2, color='purple', alpha=.25)
    plt.fill_between(X, Y, Y2, where=(0<X) & (X<1) & (Y<1) & (Y>0), color='green', alpha=.25)
    
    plt.xlim( (0, 1) )
    plt.ylim( (0, 1) )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

if __name__ == "__main__":
    # print("reward matrix:", reward(0.1))
    # print("stop probability matrix:", stop_prob(0.3))
    # pi1_0 = np.array([[1], [0]])
    # pi2_0 = np.array([[1], [0]])
    # print(value(pi1_0, pi2_0))

    # MVI()
    # print(softmax_param())
    print(derivative_theta_2(np.array([[3], [4]])))