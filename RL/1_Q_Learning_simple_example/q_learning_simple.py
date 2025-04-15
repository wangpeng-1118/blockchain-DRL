import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATE = 6 # the length of the 1 dimensional world
ACTIONS = ['left', 'right']  # available action
EPSILON = 0.9 # greedy police 选择最优动作的概率
ALPHA = 0.1 # learn rate
LAMBDA = 0.9 # discount factor
MAX_EPISODES = 13 # maximum episodes
FRESH_TIME = 0.1 # fresh time for one move

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # q_table initial values
        columns=actions                 # action's name
    )
    # print(table)
    return table

def choose_action(state, q_table):
    state_action = q_table.iloc[state, :] # 将q_table 中的state那一行赋值到state_action中

    if(np.random.uniform() > EPSILON) or (state_action.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        if(q_table.iloc[state].loc['left'] > q_table.iloc[state].loc['left']):
            action_name = 'left'
        else:
            action_name = 'right'
    return action_name

def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATE - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = 0
        else:
            S_ = S - 1
    return S_, R

def update_env(S, episode, step_counter):
    env_list = ['-']*(N_STATE - 1) + ['T']  # 创建一个一维的环境
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                         ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def rl():
    q_table = build_q_table(N_STATE, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_action(S, q_table)
            # print(A)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.iloc[S].loc[A]   # 索引第S行，标签为A的值
            if S_ != 'terminal':
                q_target = R + LAMBDA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True

            q_table.iloc[S].loc[A] += ALPHA * (q_target - q_predict) # update
            S = S_

            update_env(S, episode, step_counter + 1)
            step_counter += 1
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ_table:\n')
    print(q_table)





