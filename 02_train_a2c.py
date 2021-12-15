#!/usr/bin/env python3
import os
import time
import ptan
import gym
import pybullet_envs
import argparse
from tensorboardX import SummaryWriter

import a2c_model, common

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

ENV_NAME = "MinitaurBulletEnv-v0"
GAMMA = 0.999
REWARD_STEPS = 4
BATCH_SIZE = 32
LEARNING_RATE = 1e-4  # 5e-5
ENTROPY_BETA = 1e-3

TEST_ITERS = 10000
MAX_STEPS = 500_000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action="store_true", help="Enable CUDA"
    )
    #parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    save_path = os.path.join("saves", "a2c-" + f"{ENV_NAME}")
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(ENV_NAME)
    test_env = gym.make(ENV_NAME)
    net = a2c_model.ModelA2C(env.observation_space.shape[0], env.action_space.shape[0])
    net = net.to(device)
    agent = a2c_model.AgentA2C(net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, REWARD_STEPS)
    
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(comment="-a2c_" + f"{ENV_NAME}-L{LEARNING_RATE}")
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
                    rewards, steps = common.test_net(net, test_env, device=device)
                    print("Test done is %.2f sec, reward %.2f, steps %d" % (
                            time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.2f -> %.2f" % (best_reward, rewards))
                            name = "best_%+.2f_%d.dat" % (rewards, step_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(net.state_dict(), fname)
                        best_reward = rewards

                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, vals_ref_v = common.unpack_batch_a2c(
                    batch, net, GAMMA ** REWARD_STEPS, device)
                batch.clear()

                optimizer.zero_grad()
                mu_v, var_v, value_v = net(states_v)

                loss_value = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                adv = vals_ref_v.unsqueeze(-1) - value_v.detach()
                log_prob = adv * common.calc_logprob(mu_v, var_v, actions_v)
                loss_policy = -log_prob.mean()

                # entropy for normal distribution
                neg_entropy = - ( (torch.log(2*np.pi*var_v) + 1) / 2).mean()
                loss_entropy = ENTROPY_BETA * neg_entropy

                loss = loss_policy + loss_value + loss_entropy
                loss.backward()
                optimizer.step()

                tb_tracker.track("loss_entropy", loss_entropy, step_idx)
                tb_tracker.track("loss_policy", loss_policy, step_idx)
                tb_tracker.track("loss_value", loss_value, step_idx)
                tb_tracker.track("loss_total", loss, step_idx)



                
