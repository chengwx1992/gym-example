#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import __init__

import os
import random
import numpy as np
import gym
from gym import spaces
import ray
from ray.rllib.agents import ppo
import alphartc_gym
from utils.packet_info import PacketInfo
from utils.packet_record import PacketRecord

UNIT_M = 1000000
MAX_BANDWIDTH_MBPS = 8
MIN_BANDWIDTH_MBPS = 0.01
LOG_MAX_BANDWIDTH_MBPS = np.log(MAX_BANDWIDTH_MBPS)
LOG_MIN_BANDWIDTH_MBPS = np.log(MIN_BANDWIDTH_MBPS)

def liner_to_log(value):
    # from 10kbps~8Mbps to 0~1
    value = np.clip(value / UNIT_M, MIN_BANDWIDTH_MBPS, MAX_BANDWIDTH_MBPS)
    log_value = np.log(value)
    return (log_value - LOG_MIN_BANDWIDTH_MBPS) / (LOG_MAX_BANDWIDTH_MBPS - LOG_MIN_BANDWIDTH_MBPS)

def log_to_linear(value):
    # from 0~1 to 10kbps to 8Mbps
    value = np.clip(value, 0, 1)
    log_bwe = value * (LOG_MAX_BANDWIDTH_MBPS - LOG_MIN_BANDWIDTH_MBPS) + LOG_MIN_BANDWIDTH_MBPS
    return np.exp(log_bwe) * UNIT_M

class GymEnv:
    def __init__(self, step_time=60):
        self.gym_env = None     
        self.step_time = step_time
        self.trace_set = os.path.join(os.path.dirname(__file__), "traces")
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float64)

    def reset(self):
        self.gym_env = alphartc_gym.Gym("test_gym")
        self.gym_env.reset(trace_path=random.choice(self.trace_set),
            report_interval_ms=self.step_time,
            duration_time_ms=0)
        packet_record = PacketRecord()
        packet_record.reset()
        return [0.0, 0.0, 0.0, 0.0]

    def step(self, action):
        # action: log to linear
        bandwidth_prediction = log_to_linear(action)

        # run the action
        packet_list, done = self.gym_env.step(bandwidth_prediction)
        for pkt in packet_list:
            packet_info = PacketInfo()
            packet_info.payload_type = pkt["payload_type"]
            packet_info.ssrc = pkt["ssrc"]
            packet_info.sequence_number = pkt["sequence_number"]
            packet_info.send_timestamp = pkt["send_time_ms"]
            packet_info.receive_timestamp = pkt["arrival_time_ms"]
            packet_info.padding_length = pkt["padding_length"]
            packet_info.header_length = pkt["header_length"]
            packet_info.payload_size = pkt["payload_size"]
            packet_info.bandwidth_prediction = bandwidth_prediction
            packet_record.on_receive(packet_info)

        # calculate state
        states = []
        receiving_rate = packet_record.calculate_receiving_rate(interval=self.step_time)
        states.append(liner_to_log(receiving_rate))
        delay = packet_record.calculate_average_delay(interval=self.step_time)
        states.append(min(delay/1000, 1))
        loss_ratio = packet_record.calculate_loss_ratio(interval=self.step_time)
        states.append(loss_ratio)
        latest_prediction = packet_record.calculate_latest_prediction()
        states.append(liner_to_log(latest_prediction))

        # calculate reward
        reward = states[0] - states[1] - states[2]

        return states, reward, done, {}

ray.init()
trainer = ppo.PPOTrainer(env=GymEnv)

while True:
    print(trainer.train())
