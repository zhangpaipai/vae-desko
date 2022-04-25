import numpy as np

VARIANT = {

    'env_name': 'CartPole-v0',

    'alg_name': 'DeSKO',   # (Deep Stochastic Koopman Operator)
    
    'log_path': './log',   # A, B, vae

    'tensorboad_path': './tensorboard',   # tensorboard可视化

    'learning_rate': 1e-3,
        
    'encoder_struct': [100, 80],  # state空间到latent空间的encoder(MLP)每层神经元个数

    'init_w': 3e-3,  # state空间到latent空间的encoder(MLP)最后一个连接层的权重范围
    
    'activation': 'relu',  # state空间到latent空间的encoder(MLP)的激活函数

    'state_dim': (3,400,600),  # 图片维数
    
    'act_dim': 1,  # 动作空间维数
        
    'latent_dim': 20,  # 隐空间维数

    'pred_horizon': 16,  # 多步更新的步长(包含初始状态，所以其实是15)

    'controller_name': 'Stochastic_MPC_with_observation_v2',   # 哪种MPC

    'alpha': .1,  # 熵（H）的拉格朗日乘子

    'target_entropy': -20.,  # 熵（H）的下界
        
    'batch_size': 16,  #128
    
    'num_epochs': 5000,       # 训练desko的轮数
    
    'total_data_size': 40000,

    'save_frequency': 1,    # desko模型的保存频率(A,B,vae)
    
    'N2': 500,  # 数据集总轮数N2

    ### MPC params
    'Q': np.diag([1., .1, 10., 0.01]),
    'R': np.diag([0.1]),
    'control_horizon': 6,
    'controller_name': 'Stochastic_MPC_with_observation_v2',
    # dynamic eval
    'max_ep_steps': 250,

    'eval_render': False,

    'evaluation_form': 'normal_step',   # env.step的类型('impulse'和'constant_impulse'和'various_disturbance'和'normal_step')

    'directly_show': True,   # 画图
    
    'plot_average': False   # 画图
}


def get_env_from_name(args):
    
    from cartpole_env import CartPoleEnv_adv as dreamer
    env = dreamer()
    env = env.unwrapped
    #env.seed(SEED)
    return env


# 一共9种MPC
def get_controller(model, args):
    if args['controller_name'] == 'MPC':
        from lqr_controller import MPC as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'MPC_with_motion_planning':
        from lqr_controller import MPC_with_motion_planning as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Time_varying_MPC':
        from lqr_controller import Time_varying_MPC as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Stochastic_MPC':
        from lqr_controller import Stochastic_MPC as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Stochastic_MPC_v2':
        from lqr_controller import Stochastic_MPC_v2 as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Stochastic_MPC_v3':
        from lqr_controller import Stochastic_MPC_v3 as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Stochastic_MPC_with_observation':
        from lqr_controller import Stochastic_MPC_with_observation as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Stochastic_MPC_with_observation_v2':
        from lqr_controller import Stochastic_MPC_with_observation_v2 as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Stochastic_MPC_with_motion_planning':
        from lqr_controller import Stochastic_MPC_with_motion_planning as build_func
        controller = build_func(model, args)

    else:
        print('controller does not exist')
        raise NotImplementedError
    return controller

def store_hyperparameters(path, args):
    np.save(path + "/hyperparameters.npy", args)