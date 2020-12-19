import numpy as np

from model import reward, stop_prob, value


def project(pi):

    x = pi[0][0]
    y = pi[1][0]
    res_x = (1 + x - y)/2
    res_y = (1 + y - x)/2

    if res_x >= 1:
        return np.array([[1], [0]])
    elif res_x <= 0:
        return np.array([[0], [1]])
    else:
        return np.array([[res_x], [res_y]])


if __name__ == "__main__":
    print(project(np.array([[0], [0]])))
