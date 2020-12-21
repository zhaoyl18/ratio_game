# ratio_game
Code of policy gradient methods for von Neumann's *ratio* game, a very simple stochastic game.

## baseline
Reproduce EG(Extragradient methods) convergence results from [Independent policy gradient methods](https://papers.nips.cc/paper/2020/file/3b2acfe2e38102074656ed938abf4ac3-Paper.pdf)
![image](https://github.com/zhaoyl18/ratio_game/blob/main/results/EG_baseline.png)

## two-timescale PG
Policy gradient descent&ascent(GDA) with **Softmax** parameterization
![image](https://github.com/zhaoyl18/ratio_game/blob/main/results/softmax/eta=1.00.png)

## two-timescale NPG
Simultaneous natural policy GDA with **Softmax** parameterization
![image](https://github.com/zhaoyl18/ratio_game/blob/main/results/natural/eta=1.00.png)
