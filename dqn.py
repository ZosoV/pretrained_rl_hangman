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

register(
    id='HangmanEnv-v0',
    entry_point='custom_env.hangman_env:HangmanEnv',
    max_episode_steps=32, # NOTE: maximum word length normally around 24 + 6 lives
)

# Load config.yaml
with open("config_dqn.yaml", "r") as file:
    config = yaml.safe_load(file)

# Set the name with the data of the experiment
current_date = datetime.datetime.now()
date_str = current_date.strftime("%Y_%m_%d-%H_%M_%S")  # Includes date and time
run_name = f"{config['run_name']}_{date_str}"

# Initialize W&B run with config
wandb.init(
    name=run_name,
    project=config["project_name"], 
    group=config["group_name"], 
    mode=config["mode"], 
    config=config  # Pass the entire config for hyperparameter tracking
)

# Print config
# print("Config:")
# for key, value in config.items():
#     print(f"{key}: {value}")

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
model = make_dqn_model(cfg.policy).to(device)

#TODO: Set correctly the input with the ids according to the tokenizer

# NOTE: annealing_num_steps: number of steps 
# it will take for epsilon to reach the eps_end value.
# the decay is linear
greedy_module = EGreedyModule(
    annealing_num_steps=cfg.collector.epsilon_decay_period,
    eps_init=cfg.collector.epsilon_start,
    eps_end=cfg.collector.epsilon_end,
    spec=model.spec,
).to(device)
model_explore = TensorDictSequential(
    model,
    greedy_module,
).to(device)

# Create the collector
# NOTE: init_random_frames: Number of frames 
# for which the policy is ignored before it is called.
collector = SyncDataCollector(
    create_env_fn=make_env( device = device, 
                            seed = cfg.env.seed,
                            pretrained_bert = cfg.policy.load_pretrained_bert,),
    policy=model_explore,
    frames_per_batch=frames_per_batch,
    # total_frames=total_frames,
    exploration_type=ExplorationType.RANDOM,
    device=device,
    storing_device=device,
    split_trajs=False,
    # max_frames_per_traj=-1,
    init_random_frames=init_random_frames,
)

# Create the replay buffer
if cfg.buffer.prioritized_replay:
    print("Using Prioritized Replay Buffer")
    sampler = PrioritizedSampler(
        max_capacity=cfg.buffer.buffer_size, 
        alpha=cfg.buffer.alpha, 
        beta=cfg.buffer.beta)
else:
    sampler = RandomSampler()

# if cfg.buffer.scratch_dir is None:
#     tempdir = tempfile.TemporaryDirectory()
#     scratch_dir = tempdir.name
# else:
#     scratch_dir = cfg.buffer.scratch_dir

storage = LazyTensorStorage(max_size=cfg.buffer.buffer_size, device=device)
# storage = LazyMemmapStorage( # NOTE: additional line
#         max_size=cfg.buffer.buffer_size,
#         scratch_dir=scratch_dir,
#     ),

replay_buffer = TensorDictReplayBuffer(
    pin_memory=False,
    prefetch=20,
    storage=storage,
    batch_size=cfg.buffer.batch_size,
    sampler = sampler,
    priority_key="total_priority"
)

# Create the loss module
loss_module = DQNLoss(
    value_network=model,
    loss_function="l2", 
    delay_value=True, # delay_value=True means we will use a target network
)

loss_module.make_value_estimator(gamma=cfg.loss.gamma) # only to change the gamma value
loss_module = loss_module.to(device) # NOTE: check if need adding

target_net_updater = HardUpdate(
    loss_module, 
    value_network_update_interval=cfg.loss.target_update_period
)

# Create the optimizer
optimizer = torch.optim.Adam(loss_module.parameters(), 
                                lr=cfg.optim.lr, #
                                eps=cfg.optim.eps)


# Eval Env
with open("data/practice_words.txt", "r") as file:
    eval_word_list1 = file.read().splitlines()

test_env1 = make_env(device,
                    seed=cfg.env.seed,
                    word_list=eval_word_list1,
                    pretrained_bert=cfg.policy.load_pretrained_bert)
test_env1.eval()

# with open("data/test_words_nltk.txt", "r") as file:
#     eval_word_list2 = file.read().splitlines()

# chose 500 words randomly from eval_word_list2
# eval_word_list2 = random.sample(eval_word_list2, 500)

# test_env2 = make_env(device,
#                     seed=cfg.env.seed, 
#                     word_list=eval_word_list2,
#                     pretrained_bert=cfg.policy.load_pretrained_bert)
# test_env2.eval()

# iteration = 0
# print(f"Saving checkpoint at iteration {iteration}")

# # Create the directory
# path = f"models/dqn/{date_str}"
# os.makedirs(path, exist_ok=True)

# # Save checkpoint
# checkpoint = {
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'iteration': iteration,
#     'collected_frames': 0,
#     'total_episodes': 0,
#     'replay_buffer': replay_buffer.state_dict(),  # Save replay buffer state if supported
#     'greedy_module_state_dict': greedy_module.state_dict(),
# }
# torch.save(checkpoint, f"{path}/{cfg.run_name}_checkpoint_{iteration}.pth")

# Main loop
collected_frames = 0 # Also corresponds to the current step
total_episodes = 0

if cfg.load_checkpoint:
    # Load checkpoint
    checkpoint = torch.load(cfg.checkpoint_path)

    # Restore model and optimizer state
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore other states
    # start_iteration = checkpoint['iteration'] + 1
    # collected_frames = checkpoint['collected_frames']
    # total_episodes = checkpoint['total_episodes']

    # Restore replay buffer if applicable
    # if 'replay_buffer' in checkpoint:
    #     replay_buffer.load_state_dict(checkpoint['replay_buffer'])

    # # Restore exploration module
    # if 'greedy_module_state_dict' in checkpoint:
    #     greedy_module.load_state_dict(checkpoint['greedy_module_state_dict'])

start_time = time.time()
c_iter = iter(collector)
for iteration in range(cfg.collector.num_iterations):

    i = 0

    sum_return = 0
    number_of_episodes = 0
    num_steps = 0
    
    steps_in_batch = training_steps // frames_per_batch
    data_iter = tqdm.tqdm(
            desc="IT_%s:%d" % ("train", iteration),
            total= steps_in_batch * frames_per_batch,
            bar_format="{l_bar}{r_bar}"
        )
    
    training_start = time.time()
    for i in range(steps_in_batch):
        data = next(c_iter)
    
        data_iter.update(data.numel())

        # NOTE: This reshape must be for frame data (maybe)
        data = data.reshape(-1)
        collected_frames += frames_per_batch
        greedy_module.step(frames_per_batch)

        replay_buffer.extend(data)

        # Warmup phase (due to the continue statement)
        if collected_frames < init_random_frames:
            continue

        # optimization steps            
        sampled_tensordict = replay_buffer.sample(batch_size).to(device)

        # Also the loss module will use the current and target model to get the q-values
        loss = loss_module(sampled_tensordict)
        q_loss = loss["loss"]
        optimizer.zero_grad()
        q_loss.backward()
        if grad_clipping:
            torch.nn.utils.clip_grad_norm_(
                list(loss_module.parameters()), max_norm=max_grad)
        optimizer.step()

        # Update the priorities
        if prioritized_replay:
            replay_buffer.update_priority(index=sampled_tensordict['index'], priority = sampled_tensordict["td_error"])

        # NOTE: This is only one step (after n-updated steps defined before)
        # the target will update
        target_net_updater.step()

        # update weights of the inference policy
        # NOTE: Updates the policy weights if the policy of the data 
        # collector and the trained policy live on different devices.
        collector.update_policy_weights_()

        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        # When there are at least one done trajectory in the data batch
        if len(episode_rewards) > 0:
            sum_return += episode_rewards.sum().item()
            number_of_episodes += len(episode_rewards)
            
            # Get episode num_steps
            episode_num_steps = data["next", "step_count"][data["next", "done"]]
            num_steps += episode_num_steps.sum().detach().item()

    training_time = time.time() - training_start
    average_steps_per_second =  num_steps / training_time

    if collected_frames >= init_random_frames:
        info2flush = {
            "train/epsilon": greedy_module.eps,
            "train/average_q_value": torch.gather(data["action_value"], 1, data["action"].unsqueeze(1)).detach().mean().item(),
            "train/average_steps_per_second": collected_frames / training_time,
            "train/average_td_loss": loss["loss"].detach().mean().item(),
        }

        if number_of_episodes > 0:
            total_episodes += number_of_episodes
            info2flush["train/average_return"] = sum_return / number_of_episodes
            info2flush["train/average_episode_length"] = num_steps / number_of_episodes

        # Evaluation
        if enable_evaluation:
            if (iteration + 1) % eval_freq == 0:
                test_reward, accuracy = eval_model(model, test_env1, iteration)
                info2flush["eval1/average_return"] = test_reward
                info2flush["eval1/accuracy"] = accuracy

                # test_reward = eval_model(model, test_env2, iteration)
                # info2flush["eval2/average_return"] = test_reward

        if cfg.logger.save_checkpoint:
            if (iteration + 1) % cfg.logger.save_checkpoint_freq == 0:
                print(f"Saving checkpoint at iteration {iteration}")
                
                # Create the directory
                path = f"models/dqn/{date_str}"
                os.makedirs(path, exist_ok=True)
                
                # Save checkpoint
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': iteration,
                    'collected_frames': collected_frames,
                    'total_episodes': total_episodes,
                    'replay_buffer': replay_buffer.state_dict(),  # Save replay buffer state if supported
                    'greedy_module_state_dict': greedy_module.state_dict(),
                }
                torch.save(checkpoint, f"{path}/{cfg.run_name}_checkpoint_{iteration}.pth")

        # Flush the information to wandb
        wandb.log(info2flush, step=collected_frames)

collector.shutdown()
end_time = time.time()
execution_time = end_time - start_time
formatted_time = str(datetime.timedelta(seconds=int(execution_time)))
print(f"Collected Frames: {collected_frames}, Total Episodes: {total_episodes}")
print(f"Training took {formatted_time} (HH:MM:SS) to finish")
print("Hyperparameters used:")
print_hyperparameters(cfg)

# TODO: Saved the model. Check how to save the model and load
if cfg.logger.save_checkpoint:
    print("Saving final model")
    path = f"models/dqn/{date_str}"
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), f"{path}/{cfg.run_name}_final_model.pt")


wandb.finish()

