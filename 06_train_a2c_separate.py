#!/usr/bin/env python3
import os
import time
import ptan
import gym
import pybullet_envs
import argparse
from tensorboardX import SummaryWriter

import a2c_model_separate, common

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

# ENV_NAME = "MinitaurBulletEnv-v0"
ENV_NAME = "LunarLanderContinuous-v2" ; STOP_REWARD = 200
VAL_SCALE = 1

GAMMA = 1.0
REWARD_STEPS = 2
BATCH_SIZE = 64
LR = 0.001
LR_RATIO = 2 # crt_lr / act_lr 
ENTROPY_BETA = 0.005
TEST_ITERS = 20_000
MAX_STEPS = 1_000_000
NUM_ENVS = 50
CLIP_GRAD = 0 # no clipping if 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action="store_true", help="Enable CUDA"
    )
    # parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    save_path = os.path.join("saves", "a2c-sep-" + f"{ENV_NAME}")
    os.makedirs(save_path, exist_ok=True)

    #env = gym.make(ENV_NAME)
    envs = [gym.make(ENV_NAME) for _ in range(NUM_ENVS)]
    test_env = gym.make(ENV_NAME)

    obs_size = test_env.observation_space.shape[0]
    act_size = test_env.action_space.shape[0]
    net_act = a2c_model_separate.ModelActor(obs_size, act_size).to(device)
    net_crt = a2c_model_separate.ModelCritic(obs_size, val_scale=VAL_SCALE).to(device)

    agent = a2c_model_separate.AgentA2C(net_act, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        envs, agent, GAMMA, REWARD_STEPS
    )

    opt_crt = optim.Adam(net_crt.parameters(), lr=LR)
    opt_act = optim.Adam(net_act.parameters(), lr=LR/LR_RATIO)

    writer = SummaryWriter(
        comment="-a2c-sep_" + f"{ENV_NAME}-L{LR}R{LR_RATIO}_" +
                f"B{BATCH_SIZE}_V{VAL_SCALE}_Et{ENTROPY_BETA}_C{CLIP_GRAD}"
    )
    batch = []
    best_reward = None

    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                if step_idx > MAX_STEPS:
                    print(f"Training Stopped after {MAX_STEPS}!")
                    break
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], step_idx)
                    tracker.reward(rewards[0], step_idx)

                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    with torch.no_grad():
                        rewards, steps = common.test_net(
                            net_act, test_env, device=device, act_only=True
                            )
                    print(
                        "Test done is %.2f sec, reward %.2f, steps %d"
                        % (time.time() - ts, rewards, steps)
                    )
                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print(
                                "Best reward updated: %.2f -> %.2f"
                                % (best_reward, rewards)
                            )
                            name = "best_%+.2f_%d.dat" % (rewards, step_idx)
                            fname = os.path.join(save_path, name)
                            #torch.save(net_act.state_dict(), fname)
                        best_reward = rewards
                        if best_reward > STOP_REWARD:
                            print("Solved!")
                            break

                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, vals_ref_v = common.unpack_batch_a2c(
                    batch, net_crt, GAMMA ** REWARD_STEPS, device, seperate_act_crt=True
                )
                batch.clear()

                opt_crt.zero_grad()
                value_v = net_crt(states_v)
                loss_value = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                loss_value.backward()
                if CLIP_GRAD > 0:
                    torch.nn.utils.clip_grad_norm_(net_crt.parameters(), CLIP_GRAD)
                opt_crt.step()

                opt_act.zero_grad()
                mu_v = net_act(states_v)
                adv = vals_ref_v.unsqueeze(-1) - value_v.detach()
                var_v = torch.exp(net_act.logstd) ** 2
                log_prob = adv * common.calc_logprob(mu_v, var_v, actions_v)
                loss_policy = -log_prob.mean()

                # entropy for normal distribution
                neg_entropy = -((torch.log(2 * np.pi * var_v) + 1) / 2).mean()
                loss_entropy = ENTROPY_BETA * neg_entropy

                loss_act = loss_policy + loss_entropy
                loss_act.backward()
                if CLIP_GRAD > 0:
                    torch.nn.utils.clip_grad_norm_(net_act.parameters(), CLIP_GRAD)
                opt_act.step()

                # if var_v.mean() > 5:
                #     print("variance too large!")
                #     break
                tb_tracker.track("loss_value", loss_value, step_idx)
                tb_tracker.track("loss_entropy", loss_entropy, step_idx)
                tb_tracker.track("loss_policy", loss_policy, step_idx)
                tb_tracker.track("values", value_v.mean(), step_idx)
                tb_tracker.track("vals_ref_v", vals_ref_v.mean(), step_idx)
                tb_tracker.track("mean_var", var_v.mean(), step_idx)

