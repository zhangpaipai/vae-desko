import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import gym
from cartpole_env import CartPoleEnv_adv
from PIL import Image
import matplotlib.pyplot as plt
import cv2
#import stable_baselines3
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from variant import *
from network import *
import os
import numpy as np
# 数据集：随机采样动作，记录动作序列action_sequence['pred_horizon'-1]，cost序列['pred_horizon'-1]，图像['pred_horizon']。一共收集N2个图像、动作、cost序列。


# 数据集的序列步数n，总轮数N2
n = 16
N2 = 10000

def collect_dataset(args):

    os.makedirs('./dataset', exist_ok=True)
    x_input = torch.zeros([ args['N2'], args['pred_horizon'], 3, 400, 600 ]) 
    a_input = torch.zeros([ args['N2'], args['pred_horizon']-1, args['act_dim'] ]) 
    cost_input = torch.zeros([ args['N2'], args['pred_horizon']-1, 1 ]) 
    env = CartPoleEnv_adv()
    env = env.unwrapped
    state = env.reset()
    for i in range(args['N2']):
        screen = np.array(env.render(mode='rgb_array'))
        cv2.imwrite("./dataset/{}_0.jpg".format(i), cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        # 转为tensor，且维度(3,400,600)
        screen = ( torch.as_tensor(screen, dtype=torch.float32) ).permute(2,0,1)
        x_input[i][0] = screen
        for s in range(args['pred_horizon']-1):
            
            action = env.action_space.sample()
            state, cost, done, info = env.step(action)
            # 不满16步就done了怎么办
            if done:
                state = env.reset()
            screen = np.array(env.render(mode='rgb_array'))
            cv2.imwrite("./dataset/{}_{}.jpg".format(i,s+1), cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
            # 转为tensor，且维度(3,400,600)
            screen = ( torch.as_tensor(screen, dtype=torch.float32) ).permute(2,0,1)
            x_input[i][s+1] = screen
            a_input[i][s] = torch.as_tensor(action, dtype=torch.float32)
            cost_input[i][s] = cost
        
    env.close()
    # 返回的是tensor
    return x_input, a_input, cost_input

