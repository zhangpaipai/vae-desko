from variant import *
import os
import gym
from robustness_eval import *
from desko import Desko 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from generate_data import *
from network import *
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter


class DealDataset(Dataset):
    
    def __init__(self, args):
        x_input, a_input, cost_input = collect_dataset(args)
        #self.x_data = torch.as_tensor(x_input, dtype=torch.float32)
        #self.a_data = torch.as_tensor(a_input, dtype=torch.float32)
        self.x_data = x_input/255
        self.a_data = a_input
        self.cost_data = cost_input
        self.len = x_input.shape[0] # N2
        print(x_input.shape)
    
    def __getitem__(self, index):
        return self.x_data[index], self.a_data[index], self.cost_data[index]

    def __len__(self):
        return self.len



def loss_func(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x,  size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE+KLD


def main():
    args = VARIANT
    root_dir = args['log_path']
    env = get_env_from_name(args)
    #args['state_dim'] = env.observation_space.shape[0]
    args['act_dim'] = env.action_space.shape[0]
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs("./construction_image", exist_ok=True)
    
    model = train_desko(args)

    print('logging to ' + args['log_path'])
    # 存variant
    if args['store_hyperparameter']:
        store_hyperparameters(root_dir, args)
    # 选择哪种MPC控制
    controller = get_controller(model, args)
    controller._build_controller()
    controller.check_controllability()

    if args['evaluation_form'] == 'dynamic':
        dynamic(controller, env, args, args)
    elif args['evaluation_form'] == 'constant_impulse':
        constant_impulse(controller, env, args)
    
# 测试pred_horizon内预测是否准确
def test():
    args = VARIANT
    env = get_env_from_name(args)
    #args['state_dim'] = env.observation_space.shape[0]
    args['act_dim'] = env.action_space.n

    os.makedirs('./dataset_test', exist_ok=True)
    
    model = Desko(args)
    model.load(args['log_path'])
    model.restore_Koopman_operator()
    
    state = env.reset()
    for i in range(2):
        screen = np.array(env.render(mode='rgb_array'))
        cv2.imwrite("./dataset_test/true_{}_0.jpg".format(i), cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        # 转为tensor，且维度(3,400,600)
        screen = ( torch.as_tensor(screen, dtype=torch.float32) ).permute(2,0,1)
        x_input = screen.unsqueeze(0)
        for s in range(args['pred_horizon']-1):
            
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            a_input = (torch.tensor(action, dtype=torch.float32)).unsqueeze(0)
            # 不满16步就done了怎么办
            if done:
                state = env.reset()
            screen = np.array(env.render(mode='rgb_array'))
            cv2.imwrite("./dataset_test/true_{}_{}.jpg".format(i,s+1), cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
            
            phi, log_prob = model.mlp.get_phi(x_input)
            phi = phi.unsqueeze(-1)  # 保证矩阵能相乘
            phi = torch.matmul(model.A, phi) + torch.matmul(model.B, a_input.unsqueeze(-1))  # (批量,隐空间维度,1)
            x_input = torch.matmul(model.C, phi)
            x_input = x_input.squeeze(-1)
            predict = model.decoder(x_input)
            save_image(predict[0], "./dataset_test/predict_{}_{}.jpg".format(i,s+1))
    env.close()


def train_desko(args):
    
    # Generate training data
    data = DealDataset(args) 
    train_loader = DataLoader(data, batch_size=args['batch_size'], shuffle=True, drop_last=True)
    model = Desko(args)
    # tensorboard可视化
    writer = SummaryWriter(args['tensorboad_path'])
    for e in range(args['num_epochs']):
        for x_input, a_input, cost_input in train_loader:  
            # 训练更新desko模型并返回 alpha_loss和loss
            alpha_loss, loss, forward_pred_loss, reconstruct_loss, KL_loss = model.learn(args, x_input, a_input, cost_input, e)  
        # tensorboard可视化
        writer.add_scalar('desko_alpha_loss_loss', alpha_loss, e)
        writer.add_scalar('desko_loss_loss', loss, e)
        writer.add_scalar('desko_forward_pred_loss', forward_pred_loss, e)
        writer.add_scalar('desko_reconstruct_loss', reconstruct_loss, e)
        writer.add_scalar('desko_KL_loss', KL_loss, e)
        
        if e % args['save_frequency'] == 0:
            model.save(args['log_path'])
            model.store_Koopman_operator()
            print('episode: ', e)
    writer.close()
    return model



if __name__ == '__main__':
    main()
    #test()