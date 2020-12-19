import numpy as np
import os
import matplotlib.pyplot as plt

def reward(epsilon = 0.1):
    R = np.array([[-1, epsilon], [-epsilon, 0]])
    return R

def stop_prob(s = 0.3):
    S = np.array([[s, s], [1, 1]])
    return S

def value(pi_1, pi_2):
    nume = np.dot(pi_1.T, np.dot(reward(), pi_2))[0][0]
    denom = np.dot(pi_1.T, np.dot(stop_prob(), pi_2))[0][0]
    value = nume/denom
    return value

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

    MVI()