This project automates the Chrome Dinosaur Game using reinforcement learning (RL). Built with Python, OpenAI's Gym, and the Stable-Baselines3 library, the custom environment (DinoGameEnv) controls the dino character within the browser, using Selenium to interact with the game in real-time.

# Key features include:

Environment Setup: A custom Gym environment for the Chrome Dino Game that uses Selenium WebDriver to interact with the browser.
State Representation: Screenshots from the game are processed and resized to provide grayscale observations.
Reinforcement Learning: A Deep Q-Network (DQN) model is used to train the agent to play the game.
Training and Evaluation: The model trains with the DQN algorithm for a specified number of timesteps, learning to avoid obstacles

# How to Use
## 1. Prerequisites
Make sure you have the following installed:

Python 3.7+
Selenium (pip install selenium)
OpenAI Gym (pip install gym)
Stable-Baselines3 (pip install stable-baselines3)
OpenCV (pip install opencv-python)
Pillow (pip install pillow)
A ChromeDriver matching your Chrome browser version (download from here)

## 2. Clone the Repository
bash
Copy code
git clone https://github.com/yourusername/chrome-dino-rl.git
cd chrome-dino-rl
3. Setup ChromeDriver Path
Replace '/path/to/chromedriver' in the code with the actual path to your ChromeDriver executable.

## 4. Run the Code
Run the following code to start training the agent:

python
```
python dino_train.py```
This will:

Create a custom Gym environment for the Dino game.
Train a Deep Q-Network (DQN) model using Stable-Baselines3.
Save the trained model as dino_dqn_model.
5. Test the Trained Model
To test the model, run the following code:

python
```
python dino_test.py```
The agent will play the game based on the trained model and display its performance.

Notes:

Make sure the game window remains open during training/testing.
Adjust parameters (like total_timesteps) in the code if you want to run longer training sessions.
