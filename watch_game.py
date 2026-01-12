import time
from collections import deque

import cv2
import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

MODEL_PATH = "flappy_checkpoint.pth"
IMAGE_SIZE = 84
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImagePreprocessWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(4, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8
        )
        self.frame_buffer = deque(maxlen=4)

    def observation(self, obs):
        img = self.env.render()

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_resized = cv2.resize(img_gray, (IMAGE_SIZE, IMAGE_SIZE))

        if len(self.frame_buffer) == 0:
            for _ in range(4):
                self.frame_buffer.append(img_resized)
        else:
            self.frame_buffer.append(img_resized)

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


def main():
    env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
    env = ImagePreprocessWrapper(env)

    model = DQN((4, IMAGE_SIZE, IMAGE_SIZE), env.action_space.n).to(DEVICE)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print("ERROR: No checkpoint file found.")
        return

    for episode in range(5):
        obs, info = env.reset()
        state = obs
        total_reward = 0
        done = False

        print(f"Start episode: {episode + 1}...")

        while not done:
            img_to_show = env.render()
            img_bgr = cv2.cvtColor(img_to_show, cv2.COLOR_RGB2BGR)
            cv2.imshow("AI Flappy Bird", img_bgr)

            with torch.no_grad():
                state_t = (
                    torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    / 255.0
                )
                action = model(state_t).max(1)[1].item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            state = next_obs
            total_reward += reward
            done = terminated or truncated

            if cv2.waitKey(20) & 0xFF == ord("q"):
                done = True
                episode = 5

        print(f"Episode {episode + 1} score: {total_reward:.2f}")
        time.sleep(1)

    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()
