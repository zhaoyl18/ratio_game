# ratio game

Implementations for the numerical simulation in paper [Local Optimization Achieves Global Optimality in Multi-Agent Reinforcement
Learning](https://openreview.net/forum?id=V4jD1KmnQz). We implement and compare: (1) our algorithm with sequential policy updates and (2) the[independent policy gradient method](https://papers.nips.cc/paper/2020/file/3b2acfe2e38102074656ed938abf4ac3-Paper.pdf). We study von Neumann's ratio game, a very simple stochastic game. Plots for the six settings are shown below.

## (a)

Policies are initialized close to the stationary point, stepsize is 0.001.
![image](https://github.com/zhaoyl18/ratio_game/blob/main/stationary/stepsize= 0.00.png)

## (b)

Policies are initialized close to the stationary point, stepsize is 0.02.
![image](https://github.com/zhaoyl18/ratio_game/blob/main/stationary/stepsize= 0.01.png)

## (c)

Policies are initialized close to the stationary point, stepsize is 0.05.
![image](https://github.com/zhaoyl18/ratio_game/blob/main/stationary/stepsize= 0.05.png)

## (d)

Both policies are uniformly initialized, stepsize is 0.001.
![image](https://github.com/zhaoyl18/ratio_game/blob/main/stepsize= 0.00.png)

## (e)

Both policies are uniformly initialized, stepsize is 0.01.
![image](https://github.com/zhaoyl18/ratio_game/blob/main/stepsize= 0.01.png)

### (f)

Both policies are uniformly initialized, stepsize is 0.005.
![image](https://github.com/zhaoyl18/ratio_game/blob/main/stepsize= 0.05.png)