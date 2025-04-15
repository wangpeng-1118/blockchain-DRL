"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using PyTorch to build the neural network.

View more on original author's tutorial page: https://morvanzhou.github.io/tutorials/
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow

The original implementation was done using TensorFlow, and I re-implemented it in PyTorch.

Using:
python: 3.8.0
PyTorch: 2.2.1
gym: 0.26.2
"""

from maze_env import Maze
from RL_brain_torch import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()