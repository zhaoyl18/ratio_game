# ratio game

Coding implementation for paper: [Local Optimization Achieves Global Optimality in Multi-Agent Reinforcement
Learning](https://arxiv.org/abs/2305.04819) (ICML 2023)

We study von Neumann's ratio game, a very simple stochastic game. We implement and compare two algorithms:

(1) Our algorithm with sequential policy updates 

(2) Independent policy gradient algorithm, e.g. this [paper](https://papers.nips.cc/paper/2020/file/3b2acfe2e38102074656ed938abf4ac3-Paper.pdf). 

Results are shown below.

## (a)

Policies are initialized close to the stationary point, stepsize is 0.001.
![image](https://github.com/zhaoyl18/ratio_game/blob/main/stationary/stepsize=0.00.png)

## (b)

Policies are initialized close to the stationary point, stepsize is 0.02.
![image](https://github.com/zhaoyl18/ratio_game/blob/main/stationary/stepsize=0.01.png)

## (c)

Policies are initialized close to the stationary point, stepsize is 0.05.
![image](https://github.com/zhaoyl18/ratio_game/blob/main/stationary/stepsize=0.05.png)

## (d)

Both policies are uniformly initialized, stepsize is 0.001.
![image](https://github.com/zhaoyl18/ratio_game/blob/main/stepsize=0.00.png)

## (e)

Both policies are uniformly initialized, stepsize is 0.01.
![image](https://github.com/zhaoyl18/ratio_game/blob/main/stepsize=0.01.png)

### (f)

Both policies are uniformly initialized, stepsize is 0.005.
![image](https://github.com/zhaoyl18/ratio_game/blob/main/stepsize=0.05.png)
