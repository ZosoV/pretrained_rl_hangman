# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn
import torch.nn.functional as F
import torch.optim
from torchrl.data import CompositeSpec
from torchrl.envs.libs.gym import GymEnv
from torchrl.record import VideoRecorder
from tensordict.nn import TensorDictModule
from torchrl.modules import QValueActor

from torchrl.envs import (
    CatFrames,
    DoubleToFloat,
    EndOfLifeTransform,
    GrayScale,
    CenterCrop,
    GymEnv,
    NoopResetEnv,
    Resize,
    RewardSum,
    SignTransform,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    VecNorm,
)

import numpy as np
np.float_ = np.float64

from utils_model import CustomBERT, freezing_layers_and_LoRA

# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_env(device="cpu", seed = 0, word_list=None, pretrained_bert=False):
    
    # Load word list
    if word_list is None:
        with open("data/filtered_words.txt", "r") as file:
            word_list = file.read().splitlines()

    # TODO: Notice probably will be worthed to set a game for all the possible words

    env = GymEnv(
        "HangmanEnv-v0",
        device=device,
        word_list=word_list,
        categorical_action_encoding=True,
        pretrained_bert=pretrained_bert)

    env = TransformedEnv(env)
    env.append_transform(RewardSum())
    env.append_transform(StepCounter()) # NOTE: Cartpole-v1 has a max of 500 steps
    env.set_seed(seed)
    return env

# ====================================================================
# Model utils
# --------------------------------------------------------------------

def make_dqn_modules_pixels(proof_environment, policy_cfg, test_model=False):

    load_pretrained_bert = policy_cfg.load_pretrained_bert
    load_pretrained_bert_path = policy_cfg.load_pretrained_bert_path

    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape
    env_specs = proof_environment.specs

    # NOTE: I think I can change the next two lines by
    # num_outputs = proof_environment.action_spec.shape[-1]
    # action_spec = proof_environment.action_spec.space
    num_actions = env_specs["input_spec", "full_action_spec", "action"].space.n
    action_spec = env_specs["input_spec", "full_action_spec", "action"]

    print(f"Using {policy_cfg.type} for q-net architecture")
    
    # Define Q-Value Module
    q_net = CustomBERT(
        vocab_size=policy_cfg.vocab_size,
        hidden_size=policy_cfg.hidden_size,
        num_hidden_layers=policy_cfg.num_hidden_layers,
        num_attention_heads=policy_cfg.num_attention_heads,
        max_position_embeddings=policy_cfg.max_position_embeddings,
        intermediate_size=policy_cfg.intermediate_size,
        dqn_head=True
    )

    if load_pretrained_bert:
        print(f"Loading pretrained BERT from {load_pretrained_bert_path}")
        
        load_pretrained_bert_path = torch.load(load_pretrained_bert_path)
        q_net.load_state_dict(load_pretrained_bert_path)

        # pretrained_state_dict = torch.load(load_pretrained_bert_path)
        # pretrained_state_dict = pretrained_state_dict["model_state_dict"]

        # # Loading only the backbone
        # backbone_keys = {k: v for k, v in pretrained_state_dict.items() if "head" not in k}

        # # Update the state_dict of the new model's backbone
        # dqn_model_state_dict = q_net.state_dict()
        # dqn_model_state_dict.update(backbone_keys)
        # q_net.load_state_dict(dqn_model_state_dict)

        # Freeze the BERT model
    # q_net = freezing_layers_and_LoRA(q_net)

    q_net = TensorDictModule(q_net,
        in_keys=["observation", "tried_letters"], 
        out_keys=["action_value"])

    # NOTE: Do I need CompositeSpec here?
    # I think I only need proof_environment.action_spec
    qvalue_module = QValueActor(
        module=q_net,
        spec=CompositeSpec(action=action_spec), 
        in_keys=["observation","tried_letters"],
    )
    return qvalue_module


def make_dqn_model(policy_cfg):
    proof_environment = make_env(device="cpu", pretrained_bert=policy_cfg.load_pretrained_bert)
    qvalue_module = make_dqn_modules_pixels(proof_environment, policy_cfg)
    del proof_environment
    return qvalue_module

# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------
import tqdm

def eval_model(actor, test_env, iteration):
    num_words = len(test_env.unwrapped.word_list)
    test_env.eval()

    test_rewards = torch.zeros(num_words, dtype=torch.float32)

    data_iter = tqdm.tqdm(range(num_words), desc="IT_%s:%d" % ("val", iteration))
    
    decipher_words = 0
    # Wrap the range in tqdm to add a progress bar
    for i in data_iter:
        # test_env.reset(options={"word_id": i})
        # print(f"Testing word {i}: {test_env.unwrapped.target_word}")
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        last_reward = td_test["next", "reward"][td_test["next", "done"]]
        if last_reward.item() == 50: # wining the game
            decipher_words += 1
        test_rewards[i] = reward.sum()
    del td_test
    return test_rewards.mean(), decipher_words / num_words

def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()

def print_hyperparameters(cfg):
    keys = ["env",
            "collector",
            "buffer",
            "policy",
            "optim",
            "loss"]
    
    for key in keys:
        if key in cfg:
            print(f"{key}:")
            for k, v in cfg[key].items():
                print(f"  {k}: {v}")


def load_pretrained_BERT(model, checkpoint_path):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Restore model and optimizer state
    model.load_state_dict(checkpoint['model_state_dict'])
    return model