"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using PyTorch to build the neural network.

View more on original author's tutorial page: https://morvanzhou.github.io/tutorials/
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow

The original implementation was done using TensorFlow, and I re-implemented it in PyTorch.

Using:
PyTorch: 2.2.1
gym: 0.7.3
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

np.random.seed(1)
torch.manual_seed(1)


# Deep Q Network off-policy

class DeepQNetwork(nn.Module):
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=300, memory_size=500, batch_size=32, e_greedy_increment=None):
        super(DeepQNetwork, self).__init__()
        self.n_actions = n_actions  # n个行动
        self.n_features = n_features # 用于表示某种状态的特征数量，值为2
        self.lr = learning_rate      # 学习率
        self.gamma = reward_decay    # 奖励衰减因子
        self.epsilon_max = e_greedy  # 最优动作的选择概率
        self.replace_target_iter = replace_target_iter   # 目标网络的更新频率
        self.memory_size = memory_size                   # 经验回访缓存的大小
        self.batch_size = batch_size                     # 批量大小，每次从经验回访缓存中采样的样本数量
        self.epsilon_increment = e_greedy_increment      # 贪婪策略的增量
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        # 记录学习频率的计数器，用于控制目标网络的更新频率
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        # eval_net用于得到q预测
        self.eval_net = nn.Sequential(
            nn.Linear(self.n_features, 10),
            nn.ReLU(),
            nn.Linear(10, self.n_actions)
        )
        # target_net用于得到q现实
        self.target_net = nn.Sequential(
            nn.Linear(self.n_features, 10),
            nn.ReLU(),
            nn.Linear(10, self.n_actions)
        )

        # RMSprop是梯度下降的改进版本，它可以调整每个参数的学习率
        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=self.lr)
        # 均方误差损失函数
        self.loss_func = nn.MSELoss()

        # Initialize cost history list
        self.cost_his = []  # Add this line to initialize cost_his

    def store_transition(self, s, a, r, s_):
        # 检查当前对象是否已经具有了相对应的属性，没有则赋初值为0
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # np.hstack 将多个数组在水平方向上拼接在一起，形成一个新的数组。
        # 这些数组必须要有相同的行数
        '''
        [[1, 2, 3], [3, 4, 5]]和[[5, 6], [7, 8]]拼接起来得到[[1, 2, 3, 5, 6], [3, 4, 5, 7, 8]]
        [1, 2]和[3, 4]拼接为[1, 2, 3 ,4]
        '''
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        observation = torch.tensor(observation[np.newaxis, :], dtype=torch.float32)

        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net(observation)
            action = torch.argmax(actions_value).item()
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        batch_memory = torch.tensor(batch_memory, dtype=torch.float32)

        q_next = self.target_net(batch_memory[:, -self.n_features:])
        q_eval = self.eval_net(batch_memory[:, :self.n_features])

        # change q_target w.r.t q_eval's action
        q_target = q_eval.clone()

        batch_index = torch.arange(self.batch_size, dtype=torch.int32)
        eval_act_index = batch_memory[:, self.n_features].long()
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * torch.max(q_next, dim=1)[0]

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad() # 清除梯度
        loss.backward() # 反向传播
        self.optimizer.step() # 更新参数

        self.cost = loss.item()
        self.cost_his.append(self.cost)  # Append cost to cost_his

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()