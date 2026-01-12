import os
import random
from collections import deque

import cv2
import flappy_bird_gymnasium
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

SAVE_PATH = "flappy_checkpoint.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 100000
TARGET_UPDATE = 1000
LEARNING_RATE = 1e-4
MEMORY_SIZE = 50000
IMAGE_SIZE = 84


class ImagePreprocessWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(4, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8
        )
        self.frame_buffer = deque(maxlen=4)

    def observation(self, obs):
        img = self.env.render()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        if len(self.frame_buffer) == 0:
            for _ in range(4):
                self.frame_buffer.append(img)
        else:
            self.frame_buffer.append(img)
        return np.array(self.frame_buffer)


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)


env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
env = ImagePreprocessWrapper(env)
policy_net = DQN((4, IMAGE_SIZE, IMAGE_SIZE), env.action_space.n).to(DEVICE)
target_net = DQN((4, IMAGE_SIZE, IMAGE_SIZE), env.action_space.n).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer(MEMORY_SIZE)

if os.path.exists(SAVE_PATH):
    print(f"S-a găsit un checkpoint: {SAVE_PATH}. Se încarcă...")
    policy_net.load_state_dict(torch.load(SAVE_PATH))
    target_net.load_state_dict(policy_net.state_dict())
else:
    print("No checkpoint found, starting over.")

steps_done = 0
scores = []


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(
        -1.0 * steps_done / EPSILON_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            state_t = (
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE) / 255.0
            )
            return policy_net(state_t).max(1)[1].item()
    else:
        return env.action_space.sample()


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    state, action, reward, next_state, done = memory.sample(BATCH_SIZE)

    state = torch.tensor(state, dtype=torch.float32).to(DEVICE) / 255.0
    next_state = torch.tensor(next_state, dtype=torch.float32).to(DEVICE) / 255.0
    action = torch.tensor(action, dtype=torch.int64).unsqueeze(1).to(DEVICE)
    reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    done = torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    curr_Q = policy_net(state).gather(1, action)
    with torch.no_grad():
        max_next_Q = target_net(next_state).max(1)[0].unsqueeze(1)
        expected_Q = reward + (GAMMA * max_next_Q * (1 - done))

    loss = nn.MSELoss()(curr_Q, expected_Q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


print("Start training (CTRL+C to stop and save)")
try:
    for i_episode in range(5000):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            reward_adj = -1.0 if terminated else 0.1
            if reward > 0:
                reward_adj = 1.0

            memory.push(state, action, reward_adj, next_state, done)
            state = next_state
            total_reward += reward

            optimize_model()

            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                scores.append(total_reward)
                break

        if i_episode % 10 == 0:
            current_eps = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(
                -1.0 * steps_done / EPSILON_DECAY
            )
            print(
                f"Episod {i_episode} | Scor: {total_reward:.2f} | Epsilon: {current_eps:.4f}"
            )

        if i_episode % 50 == 0:
            torch.save(policy_net.state_dict(), SAVE_PATH)
            print(" -> Checkpoint saved.")

except KeyboardInterrupt:
    print("\nTraining stopped.")

finally:
    print("Saving...")
    torch.save(policy_net.state_dict(), SAVE_PATH)
    env.close()

    plt.plot(scores)
    plt.title("Score History")
    plt.savefig("scores.png")
    print("Graph saved in scores.png")
