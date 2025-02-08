import time
import datetime

import torch.nn
import torch.optim
import tqdm
import wandb

import random
import numpy as np
import torch

from tensordict.nn import TensorDictSequential
# from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.modules import EGreedyModule
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate
from torchrl.data.replay_buffers.samplers import RandomSampler, PrioritizedSampler

# from torchrl.record.loggers import generate_exp_name, get_logger
from utils_dqn import (
    eval_model,
    make_dqn_model,
    make_env,
    print_hyperparameters)

import tempfile

from collections import deque

import numpy as np
np.float_ = np.float64
import yaml

from gymnasium.envs.registration import register

from utils import AttrDict
import os

from torchrl.envs.utils import RandomPolicy
# from torchrl.collectors import SyncDataCollector

register(
    id='HangmanEnv-v0',
    entry_point='custom_env.hangman_env:HangmanEnv',
    max_episode_steps=32, # NOTE: maximum word length normally around 24 + 6 lives
)

# Load config.yaml
with open("config_dqn.yaml", "r") as file:
    config = yaml.safe_load(file)


# Access hyperparameters
cfg = AttrDict(config)

# Set seeds for reproducibility
seed = cfg.env.seed
random.seed(seed)
np.random.seed(seed)
# TODO: maybe better to set before the loop???
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set variables
total_frames = cfg.collector.num_iterations * cfg.collector.training_steps 
frames_per_batch = cfg.collector.frames_per_batch
init_random_frames = cfg.collector.init_random_frames
training_steps = cfg.collector.training_steps

grad_clipping = True if cfg.optim.max_grad_norm is not None else False
max_grad = cfg.optim.max_grad_norm
batch_size = cfg.buffer.batch_size
prioritized_replay = cfg.buffer.prioritized_replay

# Evaualation
enable_evaluation = cfg.logger.evaluation.enable
eval_freq = cfg.logger.evaluation.eval_freq

device = cfg.device
if device in ("", None):
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
device = torch.device(device)

# Print the current seed and group
print(f"Running with Seed: {seed} on Device: {device}")

# Make the components
# Policy
cfg.policy.load_pretrained_bert = False
model = make_dqn_model(cfg.policy).to(device)
model.eval()

# NOTE: evaluate also with the e-greedy
checkpoint_path = "models/dqn/2025_01_22-15_04_27/DQN_hangman_checkpoint_49.pth"

# Load checkpoint
checkpoint = torch.load(checkpoint_path)

# Restore model and optimizer state
model.load_state_dict(checkpoint['model_state_dict'])


# Eval Env
with open("data/practice_words.txt", "r") as file:
    eval_word_list1 = file.read().splitlines()
test_env1 = make_env(device,
                    seed=cfg.env.seed,
                    word_list=eval_word_list1,
                    pretrained_bert=False)
test_env1.eval()

row_info = [checkpoint_path]

# Print test reward
test_reward, accuracy = eval_model(model, test_env1, 0)
print(f"DataEval1: Test Reward: {test_reward} | Test Accuracy: {accuracy}")
row_info.append(test_reward.item())
row_info.append(accuracy)

# groups from nltk of 1000 words
with open(f"data/test_sets_1000words/group_{1}.txt", "r") as file:
    eval_word_list2 = file.read().splitlines()

test_env2 = make_env(device,
                    seed=cfg.env.seed, 
                    word_list=eval_word_list2,
                    pretrained_bert=False)
test_env2.eval()

test_reward, accuracy = eval_model(model, test_env2, 0)
print(f"DataEval: Test Reward: {test_reward} | Test Accuracy: {accuracy}")
row_info.append(test_reward.item())
row_info.append(accuracy)

# groups from train words in groups of 1000 words
with open(f"data/train_sets_1000words/group_{1}.txt", "r") as file:
    eval_word_list2 = file.read().splitlines()

test_env2 = make_env(device,
                    seed=cfg.env.seed, 
                    word_list=eval_word_list2,
                    pretrained_bert=False)
test_env2.eval()

test_reward, accuracy = eval_model(model, test_env2, 0)
print(f"Train: Test Reward: {test_reward} | Train Accuracy: {accuracy}")
row_info.append(test_reward.item())
row_info.append(accuracy)

# Append to csv called "models/results.csv"
with open("models/results.csv", "a") as file:
    file.write(",".join(map(str, row_info)))
    file.write("\n")