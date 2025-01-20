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

# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_env(frame_skip = 4, device="cpu", seed = 0, 
             cropping = False, is_test=False):

    env = GymEnv(
        env_name,
        frame_skip=frame_skip,
        from_pixels=True,
        pixels_only=False,
        device=device,
        categorical_action_encoding=True,
    )

    env = TransformedEnv(env)
    env.append_transform(NoopResetEnv(noops=30, random=True)) # NOTE: Cartpole with no noops will fall into reset in the begining
                                                                # I could use an small noop reset to avoid this, but I think is not necesary
                                                                # in this case. Analyze this later
    if not is_test:
        env.append_transform(EndOfLifeTransform()) # NOTE: Check my environment is not based on lives (so not important)
        env.append_transform(SignTransform(in_keys=["reward"])) #NOTE: cartpole has no negative rewards
    env.append_transform(ToTensorImage()) 
    env.append_transform(GrayScale())
    env.append_transform(Resize(84, 84))
    env.append_transform(CatFrames(N=frame_skip, dim=-3))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter(max_steps=4500)) # NOTE: Cartpole-v1 has a max of 500 steps
    env.append_transform(DoubleToFloat())
    env.append_transform(VecNorm(in_keys=["pixels"]))
    env.set_seed(seed)

    # NOTE: a rollout will be take a trajectory of frames and group the frames by N=4 sequentially
    # so that the output will be 7x4x84x84 because with a rollout of 10 steps we will have 7 groups of
    # 4 frames each
    return env

# ====================================================================
# Model utils
# --------------------------------------------------------------------

def make_dqn_modules_pixels(proof_environment, policy_cfg, enable_mico = False):

    # Define input shape
    input_shape = proof_environment.observation_spec["pixels"].shape
    env_specs = proof_environment.specs

    # NOTE: I think I can change the next two lines by
    # num_outputs = proof_environment.action_spec.shape[-1]
    # action_spec = proof_environment.action_spec.space
    num_actions = env_specs["input_spec", "full_action_spec", "action"].space.n
    action_spec = env_specs["input_spec", "full_action_spec", "action"]

    print(f"Using {policy_cfg.type} for q-net architecture")
    
    # Define Q-Value Module
    activation_class = getattr(torch.nn, policy_cfg.activation)

    q_net = MICODQNNetwork(
        input_shape=input_shape,
        num_outputs=num_actions,
        num_cells_cnn=list(policy_cfg.cnn_net.num_cells),
        kernel_sizes=list(policy_cfg.cnn_net.kernel_sizes),
        strides=list(policy_cfg.cnn_net.strides),
        num_cells_mlp=list(policy_cfg.mlp_net.num_cells),
        activation_class=activation_class,
        use_batch_norm=policy_cfg.use_batch_norm,
        enable_mico = enable_mico,
    )

    if enable_mico:
        q_net = TensorDictModule(q_net,
            in_keys=["pixels"], 
            out_keys=["action_value", "representation"])
    else:
        q_net = TensorDictModule(q_net,
            in_keys=["pixels"], 
            out_keys=["action_value"])

    # NOTE: Do I need CompositeSpec here?
    # I think I only need proof_environment.action_spec
    qvalue_module = QValueActor(
        module=q_net,
        spec=CompositeSpec(action=action_spec), 
        in_keys=["pixels"],
    )
    return qvalue_module


def make_dqn_model(env_name, policy_cfg, frame_skip, cropping = False, enable_mico = False):
    proof_environment = make_env(env_name, frame_skip = frame_skip, device="cpu", cropping = cropping)
    qvalue_module = make_dqn_modules_pixels(proof_environment, policy_cfg, enable_mico = enable_mico)
    del proof_environment
    return qvalue_module

# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def eval_model(actor, test_env, num_episodes=3):
    eval_seeds = [919409, 711872, 442081, 189061, 117840, 378457, 574025]

    test_rewards = torch.zeros(num_episodes, dtype=torch.float32)
    for i in range(num_episodes):
        test_env.set_seed(eval_seeds[i])
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        test_env.apply(dump_video)
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards[i] = reward.sum()
    del td_test
    return test_rewards.mean()


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


