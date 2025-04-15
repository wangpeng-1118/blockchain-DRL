"""
The double DQN based on this paper: https://arxiv.org/abs/1509.06461

View more on original author's tutorial page: https://morvanzhou.github.io/tutorials/
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow

The original implementation was done using TensorFlow, and I re-implemented it in PyTorch.


Using:
Tensorflow: 2.2.1
gym: 0.26.2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# doubleDQN的作用在于减少 Q 值过估计，提高学习稳定性，更好的性能。
class DoubleDQN(nn.Module):
    def __init__(self, n_actions, n_features, learning_rate=0.005, reward_decay=0.9,
                 e_greedy=0.9, replace_target_iter=200, memory_size=3000, batch_size=32,
                 e_greedy_increment=None, double_q=True):
        super(DoubleDQN, self).__init__()
        self.n_actions = n_actions      # 11个动作
        self.n_features = n_features    # 3
        self.lr = learning_rate         # 学习率
        self.gamma = reward_decay       # 奖励衰减率
        self.epsilon_max = e_greedy     # 选择网络给出的最优动作的概率
        self.replace_target_iter = replace_target_iter # 网络重置的步数
        self.memory_size = memory_size  # 记录的动作的数量
        self.batch_size = batch_size    # 批量大小
        self.epsilon_increment = e_greedy_increment  # 选择最优动作的概率的增量
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.double_q = double_q        # 是否使用double DQN

        self.learn_step_counter = 0     # 学习步骤计数器
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2)) # 记录动作的数组
        self._build_net()
        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

        self.cost_his = []

    def _build_net(self):
        # 输入state输出action
        self.eval_net = nn.Sequential(
            nn.Linear(self.n_features, 20),
            nn.ReLU(),
            nn.Linear(20, self.n_actions)
        )
        self.target_net = nn.Sequential(
            nn.Linear(self.n_features, 20),
            nn.ReLU(),
            nn.Linear(20, self.n_actions)
        )
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def store_transition(self, s, a, r, s_):
        # 存储历史动作，如果动作超过memory_size则对memory_size进行取余
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = torch.tensor(observation[np.newaxis, :], dtype=torch.float32)
        if np.random.uniform() > self.epsilon:  # choosing action
            action = np.random.randint(0, self.n_actions)
        else:
            actions_value = self.eval_net(observation)
            action = torch.argmax(actions_value).item()
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 在memory中，第二维的前两个数表示state，第三个表示action，第四个表示reward，最后两个表示下一个状态
        s = torch.tensor(batch_memory[:, :self.n_features], dtype=torch.float32)
        a = torch.tensor(batch_memory[:, self.n_features].astype(int), dtype=torch.long)
        r = torch.tensor(batch_memory[:, self.n_features + 1], dtype=torch.float32)
        s_ = torch.tensor(batch_memory[:, -self.n_features:], dtype=torch.float32)

        # a的shape为[batch_size],a.unsqueeze(1)的shape为[batch_size, 1]
        # gather是在相对应的维度上面寻找index中索引对应的数值，如果在二维的数据中寻找，则dim需要设置为1，三维中寻找，dim许哟啊设置为2
        q_eval = self.eval_net(s).gather(1, a.unsqueeze(1)).squeeze(1)# q估计
        # 当你调用detach()方法时，它会返回一个新的张量，该张量与原始张量共享数据，但不会出现在计算图中。
        q_next = self.target_net(s_).detach()
        if self.double_q:
            q_eval4next = self.eval_net(s_).detach()
            max_act4next = torch.argmax(q_eval4next, dim=1)
            selected_q_next = q_next[range(self.batch_size), max_act4next]
        else:
            selected_q_next = q_next.max(1)[0] # max(1)返回在dim=1的维度上面的最大值以及索引

        q_target = r + self.gamma * selected_q_next
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cost = loss.item()
        self.cost_his.append(self.cost)  # Append cost to cost_his

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        return q_eval.max().item()