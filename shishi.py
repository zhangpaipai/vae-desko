import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import gym
import torch
import cv2
import numpy as np
import os
from variant import *

a = torch.tensor([2,4], dtype = torch.float32)
print(a)






""" env = gym.make('CartPole-v0')
state = env.reset()

screen = env.render(mode='rgb_array')
#print(screen.size) #400*600*3
#print(screen.shape) #(400,600,3)
a = screen.reshape(-1)
print(screen.shape[0])

env.close()
os.makedirs('./dataset_1', exist_ok=True)
cv2.imwrite("./dataset_1/1.jpg", screen) """


""" args = VARIANT
os.makedirs('./log', exist_ok=True)
A_result = np.array(np.zeros([args['latent_dim'], args['latent_dim']]))
np.savetxt('./log/A.txt', A_result)

B_result = np.array(np.zeros([args['latent_dim'], args['act_dim']]))
np.savetxt('./log/B.txt', B_result) """