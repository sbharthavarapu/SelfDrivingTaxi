import gym
from IPython.display import clear_output
from time import sleep
import numpy as np
import math

env = gym.make("Taxi-v3").env

print("Action space: ", env.action_space)
print("State space: ", env.observation_space)


env.s = 328
env.render()

# epochs = 0
# penalties, rewards = 0, 0
#
# frames = []
#
# done = False
#
# while not done:
#     action = env.action_space.sample()
#     state, reward, done, info = env.step(action)
#
#     if reward == -10:
#         penalties += 1
#
#     frames.append({'frame': env.render(mode = "ansi"), 'state': state, "action": action, "reward": reward})
#     epochs += 1
#
# print("Time steps taken: ", epochs)
# print("Penalties incurred: ", penalties)
#
#
# def print_frames(frames):
#     for i, frame in enumerate(frames):
#         clear_output(wait=True)
#         print(frame['frame'].getvalue())
#         print(f"Timestep: {i + 1}")
#         print(f"State: {frame['state']}")
#         print(f"Action: {frame['action']}")
#         print(f"Reward: {frame['reward']}")
#         sleep(.1)
#
# print_frames(frames)

q_table = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.1
gamma = 0.6
epsilon = 0.1

all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1
        state = next_state
        epochs += 1

    if i%100 == 0:
        clear_output(wait = True)
        print(f"Episode: {i}")

print("Training finished")