# reinforcement learning

* Model-free RL
* Model-Based RL
* Policy-Based RL
* Value-Based RL
* Monte-Carlo update（回合更新）
* Temporal-Difference update（单步更新）
* On-Policy
* Off-Policy

## 概率论基础

**Random Variable(随机变量)**

Random variable： a variable whose values depend on outcomes of a random event.

Uppercase letter X for **random variable**

Lowercase letter x for an observed value

**Probability Density Function(概率密度函数)**

PDF provides a relative likelihood that the value of the random variable would equal that sample

it is a continuous distribution(gaussion distribution)
$$
p(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
$$
其中：

- $ x$是随机变量的取值
- $\mu$是均值（mean）
- $\sigma$ 是标准差（standard deviation）
- $\sigma^2$ 是方差（variance）
- $\exp(\cdot)$ 表示指数函数
- $\pi$ 是圆周率，约等于 3.1416

For continuous distribution,
$$
\int_{-\infty}^{+\infty} p(x) \, dx = 1
$$
For discrete distribution,
$$
\sum_{i} P(X = x_i) = 1
$$

**Expectation(期望)**

For continuous distribution , the expectation of f(X) is:
$$
E[f(X)] = \int_{-\infty}^{+\infty} p(x) f(x) \, dx
$$
For discrete distribution, the expectation of f(X) is:
$$
E[f(X)] = \sum_{i} f(x_i)P(X = x_i)
$$

**Random Sampling(随机抽样)**

 A sampling method where $n$ individuals are randomly selected from a population

## Terminology（术语）

**state** &**action** &**policy**

对于给定的state，policy函数会给出不同action的概率，然后在这些action中进行随机抽样

Policy function:
$$
\pi(a|s) = P(A = a|S = s).
$$
给定状态$s$,做出动作的概率密度，$\pi$ 是一个概率密度函数，强化学习就是学习policy函数

## reward

agent做出一个 动作，系统就会给一个奖励，奖励通常需要我们来定义，奖励的好坏非常影响结果

## state transition

如果有一个系统，其状态空间为 $S = \{s_1, s_2, \dots, s_n\}$，那么在状态 $s_t$ 执行动作 $a_t$ 后，转移到下一个状态 $s_{t+1}$ 的概率通常表示为：
$$
P(s_{t+1}∣s_t,a_t)
$$

- $s_t$：当前状态。
- $a_t$：当前状态下采取的行动（动作）。
- $s_{t+1}$：下一个状态。
- $P(s_{t+1}∣s_t,a_t)$：在状态 $s_t$ 下采取动作 $a_t$ 后转移到状态 $s_{t+1}$ 的概率。

## agent environment interaction

在状态$S_t$ 下，agent做出动作$a_t$ ,环境更新状态成为$S_{t + 1}$ ,同时环境还会给agent一个奖励$r_t$,

## Randomness in Reinforcement Learning 

Actions have randomness.

* Given state $S$, the action can be random, The policy function give the probability of each function.

State transitions have randomness

* Given state $S = s$ and action $A = a$ ,the environment randomly generates a new state $S'$.

## Return(累计回报)

Definition：also known as cumulative future reward(未来的累计奖励).
$$
R_t + R_{t + 1} + R_{t + 2} + R_{t + 3} + ...
$$
Are $R_t$ and $R_{t + 1}$ equally important?

## Discounted return

* $\gamma$:discounted rate(tuning hyper-parameter).

$$
U_t = R_t + \gamma R_{t + 1} + \gamma^2 R_{t + 2} + \gamma^3 R_{t + 3} + ...
$$

$$ \gamma \in [0, 1]$$

Two sources of randomness:

Action are random：$P[A = a | S = s] = \pi(a|s)$.

State are random：$P[S' = s'|S = s, A = a] = p(s'|s, a)$.

## Action-Value Function for policy $\pi$

$$
Q_{\pi}(s_t, a_t) = E[U_t|S_t = s_t, A_t = a_t].
$$

对随机变量$U_t$ 求期望得到一个常数$Q_{\pi}$ 

求法：把$U_t$ 看成未来所有动作$\{A_t, A_{t + 1}, A_{t + 2}, ...\}$和未来所有状态$\{S_t, S_{t + 1}, S_{t + 2}, ...\}$ 的函数，$s_t$和$a_t$作为观测值，其他的看成变量并对其经行积分。



$Q_{\pi}(s_t, a_t)$ 的直观意义：如果用policy函数$\pi$ ，在$s_t$这个状态之下做动作$a_t$ 的好坏。

## Optimal action-value function

$$
Q^{*}(s_t, a_t) = \underset{\pi}{max}Q_{\pi}(s_t, a_t)
$$

