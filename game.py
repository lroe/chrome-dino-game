import gym
import time
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import cv2
from PIL import Image
import random

class DinoGameEnv(gym.Env):
    def __init__(self):
        self.driver = webdriver.Chrome(executable_path='/path/to/chromedriver')
        self.driver.get("chrome://dino")
        self.driver.find_element_by_tag_name('body').send_keys(Keys.SPACE)
        time.sleep(1)
        self.game_speed = 0.1
        self.max_score = 1000
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
    
    def reset(self):
        self.driver.get("chrome://dino")
        self.driver.find_element_by_tag_name('body').send_keys(Keys.SPACE)
        time.sleep(1)
        return self.get_observation()
    
    def get_observation(self):
        screenshot = self.driver.get_screenshot_as_png()
        image = Image.open(io.BytesIO(screenshot))
        image = np.array(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (84, 84))
        return resized_image
    
    def step(self, action):
        body = self.driver.find_element_by_tag_name('body')
        if action == 1:
            body.send_keys(Keys.SPACE)
        time.sleep(self.game_speed)
        observation = self.get_observation()
        done = self.check_game_over()
        reward = -1 if done else 1
        return observation, reward, done, {}
    
    def check_game_over(self):
        screenshot = self.driver.get_screenshot_as_png()
        image = Image.open(io.BytesIO(screenshot))
        image = np.array(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.sum(gray_image < 50) > 1000
    
    def close(self):
        self.driver.quit()

from stable_baselines3 import DQN
from stable_baselines3.common.envs import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import gym

def make_env():
    return DinoGameEnv()

env = DummyVecEnv([make_env])

model = DQN('CnnPolicy', env, verbose=1)

model.learn(total_timesteps=100000)

model.save("dino_dqn_model")

observation = env.reset()
for _ in range(1000):
    action, _states = model.predict(observation)
    observation, reward, done, info = env.step(action)
    if done:
        break

env.close()
