#!/usr/bin/env python3
import os
import ptan
import time
import gym

# import pybullet_envs
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.nn.functional as F

import common, ddpg_model

ENV_NAME = "LunarLanderContinuous-v2"
STOP_REWARD = 220
# VAL_SCALE = 200

GAMMA = 0.999
REWARD_STEPS = 2

BATCH_SIZE = 32
LEARNING_RATE = 0.0005

REPLAY_SIZE = 100_000
REPLAY_INITIAL = 10_000

TEST_ITERS = 10_000
MAX_STEPS = 1_000_000

#CLIP_GRAD = 0.1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action="store_true", help="Enable CUDA"
    )
    # parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    save_path = os.path.join("saves", "ddpg-" + f"{ENV_NAME}")
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(ENV_NAME)
    test_env = gym.make(ENV_NAME)
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]
    act_net = ddpg_model.DDPGActor(obs_size, act_size).to(device)
    crt_net = ddpg_model.DDPGCritic(obs_size, act_size).to(device)
    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    agent = ddpg_model.AgentDDPG(act_net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, GAMMA, REWARD_STEPS
    )
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter(
        comment="-ddpg_" + f"{ENV_NAME}-L{LEARNING_RATE}_B{BATCH_SIZE}"
    )

    step_idx = 0
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                if step_idx > MAX_STEPS:
                    print(f"Training Stopped after {MAX_STEPS}!")
                    break

                step_idx += 1
                buffer.populate(1)
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], step_idx)
                    tracker.reward(rewards[0], step_idx)

                if len(buffer) < REPLAY_INITIAL:
                    continue

                batch = buffer.sample(BATCH_SIZE)
                (
                    states_v,
                    actions_v,
                    rewards_v,
                    dones_mask,
                    last_states_v,
                ) = common.unpack_batch_ddqn(batch, device)

                # train critic
                crt_opt.zero_grad()
                q_v = crt_net(states_v, actions_v)
                last_act_v = tgt_act_net.target_model(last_states_v)
                q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
                q_last_v[dones_mask] = 0.0
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA ** REWARD_STEPS
                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_loss_v.backward()
                crt_opt.step()
                tb_tracker.track("loss_critic", critic_loss_v, step_idx)
                tb_tracker.track("critic_ref", q_ref_v.mean(), step_idx)

                # train actor
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                actor_loss_v = -crt_net(states_v, cur_actions_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()
                tb_tracker.track("loss_actor", actor_loss_v, step_idx)

                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = common.test_net(
                        act_net, test_env, device=device, act_only=True
                    )
                    print(
                        "Test done in %.2f sec, reward %.3f, steps %d"
                        % (time.time() - ts, rewards, steps)
                    )
                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print(
                                "Best reward updated: %.3f -> %.3f"
                                % (best_reward, rewards)
                            )
                            name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(act_net.state_dict(), fname)
                        best_reward = rewards
                        if best_reward > STOP_REWARD:
                            print("Solved!")
                            break                        

