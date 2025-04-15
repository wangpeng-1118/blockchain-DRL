"""
Double DQN & Natural DQN comparison,
The Pendulum example.

View more on original author's tutorial page: https://morvanzhou.github.io/tutorials/
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow

The original implementation was done using TensorFlow, and I re-implemented it in PyTorch.

Using:
python: 3.8.0
Tensorflow: 1.0
gym: 0.26.2
"""


import gym
import numpy as np
import matplotlib.pyplot as plt
import torch

from RL_brain_torch import DoubleDQN  # DoubleDQN类已经转换为PyTorch版本

# 创建环境
env = gym.make('Pendulum-v1')

# 设置随机种子
np.random.seed(1)
env.action_space.seed(1)
env.reset(seed=1)

MEMORY_SIZE = 3000
ACTION_SPACE = 11

# 初始化两个DQN模型
natural_DQN = DoubleDQN(n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
                        e_greedy_increment=0.001, double_q=False)
double_DQN = DoubleDQN(n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
                       e_greedy_increment=0.001, double_q=True)

def train(RL):
    total_steps = 0
    observation = env.reset()[0]  # 获取初始状态，确保是数组
    q_history = []

    while True:
        # if total_steps - MEMORY_SIZE > 8000: env.render()

        action = RL.choose_action(observation)

        f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)  # convert to [-2 ~ 2] float actions
        observation_, reward, terminated, truncated, info = env.step(np.array([f_action]))

        reward /= 10  # normalize to a range of (-1, 0). r = 0 when get upright

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > MEMORY_SIZE:  # learning
            q = RL.learn()
            q_history.append(q)

        if total_steps - MEMORY_SIZE > 20000:  # stop game
            break

        observation = observation_
        total_steps += 1

    return q_history

# 训练两个模型
q_natural = train(natural_DQN)
q_double = train(double_DQN)

# 绘制结果
plt.plot(np.array(q_natural), c='r', label='natural')
plt.plot(np.array(q_double), c='b', label='double')
plt.legend(loc='best')
plt.ylabel('Q eval')
plt.xlabel('training steps')
plt.grid()
plt.show()