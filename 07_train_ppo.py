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
ENV_NAME = "LunarLanderContinuous-v2"
STOP_REWARD = 200
VAL_SCALE = 1
GAMMA = 1.0
LR = 0.0005
LR_RATIO = 5  # crt_lr / act_lr
TEST_ITERS = 20_000
MAX_STEPS = 1_000_000

GAE_LAMBDA = 0.95
TRAJECTORY_SIZE = 1 + 256  # 2048
BATCH_SIZE = 64
PPO_EPS = 0.2
PPO_EPOCHES = 2

# CLIP_GRAD = 0 # no clipping if 0
# NUM_ENVS = 50
# ENTROPY_BETA = 0.005
# REWARD_STEPS is always 1


def calc_adv_ref(trajectory, net_crt, states_v, device="cpu"):
    """
    By trajectory calculate advantage and 1-step ref value
    :param trajectory: trajectory list
    :param net_crt: critic network
    :param states_v: states tensor
    :return: tuple with advantage numpy array and reference values
    """
    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    # generalized advantage estimator: smoothed version of the advantage
    # gae_t = delta_t + gamma*lambda*gat_{t+1}
    # delta_t = r_t + gamma*next_val - val
    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, (exp,) in zip(
        reversed(values[:-1]), reversed(values[1:]), reversed(trajectory[:-1])
    ):
        if exp.done:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + GAMMA * next_val - val
            last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)  # advantage + value = q_value

    adv_v = torch.FloatTensor(list(reversed(result_adv)))
    ref_v = torch.FloatTensor(list(reversed(result_ref)))
    return adv_v.to(device), ref_v.to(device)


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

    env = gym.make(ENV_NAME)
    # envs = [gym.make(ENV_NAME) for _ in range(NUM_ENVS)]
    test_env = gym.make(ENV_NAME)

    obs_size = test_env.observation_space.shape[0]
    act_size = test_env.action_space.shape[0]
    net_act = a2c_model_separate.ModelActor(obs_size, act_size).to(device)
    net_crt = a2c_model_separate.ModelCritic(obs_size, val_scale=VAL_SCALE).to(device)

    agent = a2c_model_separate.AgentA2C(net_act, device=device)
    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)

    opt_crt = optim.Adam(net_crt.parameters(), lr=LR)
    opt_act = optim.Adam(net_act.parameters(), lr=LR / LR_RATIO)

    writer = SummaryWriter(
        comment="-ppo_"
        + f"{ENV_NAME}-L{LR}R{LR_RATIO}_T{TRAJECTORY_SIZE}"
        + f"B{BATCH_SIZE}_E{PPO_EPOCHES}_V{VAL_SCALE}"  # +f"_Et{ENTROPY_BETA}_C{CLIP_GRAD}"
    )

    trajectory = []
    best_reward = None

    with ptan.common.utils.RewardTracker(writer) as tracker:
        for step_idx, exp in enumerate(exp_source):
            if step_idx > MAX_STEPS:
                print(f"Training Stopped after {MAX_STEPS}!")
                break
            rewards_steps = exp_source.pop_rewards_steps()
            if rewards_steps:
                rewards, steps = zip(*rewards_steps)
                writer.add_scalar("episode_steps", np.mean(steps), step_idx)
                tracker.reward(np.mean(rewards), step_idx)

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
                            "Best reward updated: %.2f -> %.2f" % (best_reward, rewards)
                        )
                        name = "best_%+.2f_%d.dat" % (rewards, step_idx)
                        fname = os.path.join(save_path, name)
                        # torch.save(net_act.state_dict(), fname)
                    best_reward = rewards
                    if best_reward > STOP_REWARD:
                        print("Solved!")
                        break

            trajectory.append(exp)
            if len(trajectory) < TRAJECTORY_SIZE:
                continue

            traj_states = [t[0].state for t in trajectory]
            traj_actions = [t[0].action for t in trajectory]
            traj_states_v = torch.FloatTensor(traj_states).to(device)
            traj_actions_v = torch.FloatTensor(traj_actions).to(device)

            traj_adv_v, traj_ref_v = calc_adv_ref(
                trajectory, net_crt, traj_states_v, device=device
            )

            mu_v = net_act(traj_states_v)
            var_v = torch.exp(net_act.logstd) ** 2
            old_logprob_v = common.calc_logprob(mu_v, var_v, traj_actions_v)
            # print(old_logprob_v.shape)  # [T, 2]

            # normalize advantages (mean should be zero)
            traj_adv_v = traj_adv_v - torch.mean(traj_adv_v)
            traj_adv_v /= torch.std(traj_adv_v)

            # drop last entry from the trajectory, as our adv and ref value calculated without it
            trajectory = trajectory[:-1]
            old_logprob_v = old_logprob_v[:-1].detach()

            sum_loss_value = 0.0
            sum_loss_policy = 0.0
            count_steps = 0

            for epoch in range(PPO_EPOCHES):
                for batch_ofs in range(0, len(trajectory), BATCH_SIZE):
                    batch_l = batch_ofs + BATCH_SIZE
                    states_v = traj_states_v[batch_ofs:batch_l]
                    actions_v = traj_actions_v[batch_ofs:batch_l]
                    batch_adv_v = traj_adv_v[batch_ofs:batch_l]
                    # print(batch_adv_v.shape) #[batch_size]
                    batch_adv_v = batch_adv_v.unsqueeze(-1)
                    batch_ref_v = traj_ref_v[batch_ofs:batch_l]
                    batch_old_logprob_v = old_logprob_v[batch_ofs:batch_l]

                    opt_crt.zero_grad()
                    value_v = net_crt(states_v)
                    loss_value = F.mse_loss(value_v.squeeze(-1), batch_ref_v)
                    loss_value.backward()
                    # if CLIP_GRAD > 0:
                    #     torch.nn.utils.clip_grad_norm_(net_crt.parameters(), CLIP_GRAD)
                    opt_crt.step()

                    opt_act.zero_grad()
                    mu_v = net_act(states_v)
                    batch_var_v = torch.exp(net_act.logstd) ** 2
                    logprob_pi_v = common.calc_logprob(mu_v, batch_var_v, actions_v)
                    ratio_v = torch.exp(logprob_pi_v - batch_old_logprob_v)
                    surr_obj_v = batch_adv_v * ratio_v
                    c_ratio_v = torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS)
                    clipped_surr_v = batch_adv_v * c_ratio_v
                    loss_policy = -torch.min(surr_obj_v, clipped_surr_v).mean()
                    loss_policy.backward()
                    opt_act.step()

                    sum_loss_value += loss_value.item()
                    sum_loss_policy += loss_policy.item()
                    count_steps += 1

            trajectory.clear()
            writer.add_scalar("advantage", traj_adv_v.mean().item(), step_idx)
            writer.add_scalar("values", traj_ref_v.mean().item(), step_idx)
            writer.add_scalar("loss_policy", sum_loss_policy / count_steps, step_idx)
            writer.add_scalar("loss_value", sum_loss_value / count_steps, step_idx)
            writer.add_scalar("mean_var", var_v.mean(), step_idx)
