import numpy as np
import torch
from network import *
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import os
from torchvision.utils import save_image

class Desko():
    

    def __init__(self, args):

        if args['target_entropy'] is None:
            self.target_entropy = -args['latent_dim']  # lower bound of the entropy
        else:
            self.target_entropy = args['target_entropy']
        
        self.log_alpha = torch.log( torch.tensor(args['alpha']) )  # Entropy Temperature
        self.log_alpha.requires_grad = True
        self._create_encoder(args)
        self._create_koopman_operator(args)
        self.alpha_train = optim.Adam([self.log_alpha], lr = args['learning_rate']) 
        self.train = optim.Adam([self.A, self.B]+list(self.vae.parameters()), lr = args['learning_rate']) 
    

    def _create_encoder(self, args):
        
        self.vae = Vae(args)


    def _create_koopman_operator(self, args):
        self.A = torch.randn([args['latent_dim'], args['latent_dim']], requires_grad=True)
        self.B = torch.randn([args['latent_dim'], args['act_dim']], requires_grad=True)
        
        
    def create_forward_pred(self, args, x_input, a_input, e):
        """
        Iteratively predict future state with the Koopman operator
        :param args(list):
        #x_input = torch.zeros([ args['batch_size'], args['pred_horizon'], 3, 400, 600 ]))
        #a_input = torch.zeros([ args['batch_size'], args['pred_horizon']-1, args['act_dim'] ]))
        :return: forward_pred(Tensor): forward predictions
        """
        forward_pred = []
        x_mean_forward_pred = []
        mean_forward_pred = []
        var_forward_pred = []
        # 取出初始时间戳（步）的隐空间
        phi_t = self.phi[:, 0].unsqueeze(-1)  # 为了保证矩阵相乘，在最后的维度增加一个维度(batch_size, latent_dim, 1)
        mean_t = self.mean[:, 0].unsqueeze(-1)
        var_t = self.var[:, 0].unsqueeze(-1)

        for t in range(args['pred_horizon']-1):
            phi_t = torch.matmul(self.A, phi_t) + torch.matmul(self.B, a_input[:, t].unsqueeze(-1))  # (batch_size, latent_dim, 1)
            x_t = self.vae.decoder(phi_t.squeeze(-1))
            mean_t = torch.matmul(self.A, mean_t) + torch.matmul(self.B, a_input[:, t].unsqueeze(-1))
            var_t = torch.matmul(self.A, var_t) + torch.matmul(self.B, a_input[:, t].unsqueeze(-1))
            x_mean_t = self.vae.decoder(phi_t.squeeze(-1))
            #（预测步数-1，批量，隐空间维度）
            forward_pred.append(x_t)
            mean_forward_pred.append(mean_t.squeeze(-1))
            x_mean_forward_pred.append(x_mean_t)
            var_forward_pred.append(var_t.squeeze(-1))
        # 再转为(批量，预测步数-1，隐空间维度)（相当于把list变成张量）
        self.forward_pred = torch.stack(forward_pred, axis=1)
        if e % 10 == 0:
            save_image(self.forward_pred[0], "./construction_image/{}.jpg".format(e))
        self.x_mean_forward_pred = torch.stack(x_mean_forward_pred, axis=1)
        self.mean_forward_pred = torch.stack(mean_forward_pred, axis=1)
        self.var_forward_pred = torch.stack(var_forward_pred, axis=1)
        return

    def create_optimizer(self, args, x_input, a_input):
        
        self.entropy = - torch.mean(self.log_prob)

        self.alpha_loss = alpha_loss = self.log_alpha * (self.entropy - self.target_entropy).detach() 

        self.mean_criterion = nn.MSELoss()
        self.var_criterion = nn.MSELoss()
        # 训练矩阵A、B + vae.encoder的loss
        self.forward_pred_loss = self.mean_criterion(self.mean_forward_pred, (self.mean[:, 1:, :]).detach())\
                               + self.var_criterion(self.var_forward_pred, (self.var[:, 1:, :]).detach())
        self.forward_pred_loss = self.forward_pred_loss * 1e-20
        # 训练矩阵A、B + vae.encoder + vae.decoder的reconstruct loss
        self.reconstruct_loss = nn.BCELoss(reduction='mean')(self.forward_pred, x_input[:, 1:])
        # 最后一步的reconstuct loss
        self.reconstruct_pred_T_loss = nn.BCELoss(reduction='mean')(self.forward_pred[:, -1], x_input[:, -1])
        
        # 训练vae.encoder的KL divergence loss(with N(0,1))
        self.KL_loss = -0.5 * ( torch.sum(1 + self.log_var - self.var - self.mean**2, dim=(1,2)) ).mean()
        

        self.loss = 1 * self.forward_pred_loss + 10 * self.reconstruct_loss + 0 * self.reconstruct_pred_T_loss \
                  + self.KL_loss - (torch.exp(self.log_alpha)).detach() * self.entropy #+ cost_loss


    def learn(self, args, x_input, a_input, cost_input, e):
        """
        传入的是tensor
        x_input = torch.zeros([ args['batch_size'], args['pred_horizon'], args['state_dim'] ]))
        a_input = torch.zeros([ args['batch_size'], args['pred_horizon']-1, args['act_dim'] ]))
        """
        # 转为(批量*预测步数，图片空间维度)
        x_input = x_input.reshape([-1, *args['state_dim']])

        self.phi, self.log_prob, self.mean, self.log_var= self.vae.encoder(x_input)
        self.var = torch.exp(self.log_var)
        # 转回(批量，预测步数，隐空间维度)
        self.mean = self.mean.reshape([ args['batch_size'], args['pred_horizon'], args['latent_dim'] ])
        self.log_var = self.log_var.reshape([ args['batch_size'], args['pred_horizon'], args['latent_dim'] ])
        self.var = self.var.reshape([ args['batch_size'], args['pred_horizon'], args['latent_dim'] ])
        self.phi = self.phi.reshape([ args['batch_size'], args['pred_horizon'], args['latent_dim'] ])
        self.log_prob = self.log_prob.reshape([ args['batch_size'], args['pred_horizon'], args['latent_dim'] ])
        x_input = x_input.reshape([ args['batch_size'], args['pred_horizon'], *args['state_dim']])
        # 预测传播
        self.create_forward_pred(args, x_input, a_input, e)
        # 优化器创建
        self.create_optimizer(args, x_input, a_input)

        # 梯度更新
        self.alpha_train.zero_grad()
        self.alpha_loss.backward()
        self.alpha_train.step()

        self.train.zero_grad()
        self.loss.backward()
        self.train.step()

        return self.alpha_loss, self.loss, self.forward_pred_loss, self.reconstruct_loss, self.KL_loss
        

    def store_Koopman_operator(self):
        os.makedirs('./log', exist_ok=True)
        self.A_result = self.A.detach().numpy()
        self.B_result = self.B.detach().numpy()
        np.savetxt('./log/A.txt', self.A_result)
        np.savetxt('./log/B.txt', self.B_result)

    def restore_Koopman_operator(self):
        self.A = torch.as_tensor(np.loadtxt('./log/A.txt'), dtype=torch.float32)
        self.B = torch.as_tensor(np.loadtxt('./log/B.txt', ndmin=2), dtype=torch.float32)
    
    def save(self, path):
        torch.save(self.vae.state_dict(), path+'/'+'vae_checkpoint.pth')

    def load(self, path):
        self.vae.load_state_dict(torch.load(path+'/'+'vae_checkpoint.pth'))