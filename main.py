#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from collections import deque

import gym
import numpy as np
import torch
import torch.optim as optim
from gym.wrappers import Monitor
from gym_minigrid.wrappers import ImgObsWrapper

from misc_utils import Memory, Transition, one_hot, LinearSchedule
from model import QNetwork

torch.set_default_tensor_type(torch.FloatTensor)


def main(args):
    env = gym.make(args.env)
    if 'MiniGrid' in args.env:
        env = ImgObsWrapper(env)
    path = args.base_path + args.env
    os.makedirs(path, exist_ok=True)
    # obs_shape = np.prod(env.observation_space.shape).astype(int)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.n

    q = QNetwork(obs_shape, act_shape)
    q_target = QNetwork(obs_shape, act_shape)
    opt = optim.Adam(lr=args.lr, params=q.parameters())
    memory = Memory(capacity=args.memory)
    scheduler = LinearSchedule(schedule_timesteps=int(args.max_steps * 0.1), final_p=0.01)

    avg_rw = deque(maxlen=40)
    avg_len = deque(maxlen=40)

    def get_action(s, t):

        s = torch.Tensor(s[None,:])
        _q = q(s)
        if np.random.sample() > scheduler.value:
            best_action = np.argmax(_q.detach(), axis=-1).item()
        else:
            best_action = np.random.randint(0, act_shape)
            scheduler.update(t)
        return best_action

    def train(batch):
        batch = Transition(*zip(*batch))
        s = torch.Tensor(batch.state)
        a = torch.Tensor(one_hot(np.array(batch.action), num_classes=act_shape))
        r = torch.Tensor(batch.reward)
        d = torch.Tensor(batch.done)
        s1 = torch.Tensor(batch.next_state)

        value = (q(s) * a).sum(dim=-1)
        next_value = r + args.gamma * (1. - d) * torch.max(q_target(s1), dim=-1)[0]
        loss = (.5 * (next_value - value) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    state = env.reset()

    q_target.load_state_dict(q.state_dict())

    ep_rw = 0
    ep_len = 0
    ep = 0
    for t in range(args.max_steps):
        action = get_action(state, t)
        next_state, reward, done, _ = env.step(action)
        memory.push(state, action, next_state, reward, done)
        ep_rw += reward
        ep_len += 1

        state = next_state.copy()
        if done:
            ep += 1
            avg_rw.append(ep_rw)
            avg_len.append(ep_len)
            ep_rw = 0
            ep_len = 0
            state = env.reset()

        if t % args.train_every == 0 and len(memory) > args.batch_size:
            batch = memory.sample(batch_size=args.batch_size)
            train(batch)

        if t % args.update_every == 0:
            q_target.load_state_dict(q.state_dict())
            print(f't:{t}\tep:{ep}\tavg_rw:{np.mean(avg_rw)}\tavg_len:{np.mean(avg_len)}\teps:{scheduler.value}')

    env = Monitor(env, directory=path)

    for ep in range(4):
        s = env.reset()
        while True:
            a = get_action(s, t=0)
            s1, r, d, _ = env.step(a)
            s = s1.copy()
            if d:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", default="MiniGrid-DoorKey-5x5-v0")
    parser.add_argument("--max_steps", default=int(1e6))
    parser.add_argument("--gamma", default=.99)
    parser.add_argument("--base_path", default='logs/')
    parser.add_argument("--memory", default=int(1e4))
    parser.add_argument("--train_every", default=4)
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--update_every", default=int(1e4))
    parser.add_argument("--lr", default=1e-3)
    args = parser.parse_args()

    main(args)
