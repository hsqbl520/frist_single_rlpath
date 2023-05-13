import os
import time

import argparse
import numpy as np
from model import Model
from simple_model import MAModel
from maddpg import MADDPG
from simple_agent import MAAgent
from agent import Agent
#from ddpg import DDPG


from make_env import make_env
import paddle

#调用的是parl模型的API
from parl.utils import logger, summary
from gym import spaces

CRITIC_LR = 0.01 #
ACTOR_LR = 0.01  #
GAMMA = 0.95  #初始0.95#预测229步
TAU = 0.01  # soft update

MAX_STEP_PER_EPISODE = 25 # maximum step per episode
EVAL_EPISODES = 3

# # Runs policy and returns episodes' rewards and steps for evaluation
# def run_evaluate_episodes(env, agent, eval_episodes):
#     eval_episode_rewards = []
#     eval_episode_steps = []
#     while len(eval_episode_rewards) < eval_episodes:
#         obs = env.reset()
#         done = False
#         total_reward = 0
#         steps = 0
#         while not done and steps < MAX_STEP_PER_EPISODE:
#             steps += 1
#             action = agent.predict(obs)
#             obs, reward, done,  = env.step(action)
#
#             total_reward += reward
#             # show animation
#             if args.show:
#                 time.sleep(0.1)
#                 env.render()
#
#         eval_episode_rewards.append(total_reward)
#         eval_episode_steps.append(steps)
#     return eval_episode_rewards, eval_episode_steps
#
# def evaluate_main():
#     logger.set_dir('{}/evaluate_log/{}'.format(args.data_dir,args.date))
#     if args.continuous_actions:
#         assert isinstance(env.action_space, spaces.Box)
#     # build agents
#     # 1.建立神经网络模型
#     model = Model(env.obs_dim, env.act_dim)
#     # 2.算法
#     algorithm = DDPG(
#         model,
#         gamma=GAMMA,
#         tau=TAU,
#         actor_lr = ACTOR_LR,
#         critic_lr=CRITIC_LR)
#     # 3.创建 agent
#     agent = Agent(
#         algorithm,
#         obs_dim=env.obs_dim,
#         act_dim=env.act_dim
#         )
#
#     #2.载入训练完成的模型
#     model_file = args.data_dir+"/models/" +args.restore_model_dir + '/agent_' + 'ddpg'
#     if not os.path.exists(model_file):
#         raise Exception(
#             'model file {} does not exits'.format(model_file))
#     agent.restore(model_file)
#
#     #3开始一个episodes
#     eval_episode_rewards, eval_episode_steps = run_evaluate_episodes(env, agent, EVAL_EPISODES)
#
#     # summary.add_scalar('eval/episode_reward',np.mean(eval_episode_rewards), eval_episode_steps)
#     # logger.info('Evaluation over: {} episodes, Reward: {}'.format(EVAL_EPISODES, np.mean(eval_episode_rewards)))

def run_episode(env, agent):

    obs = env.reset()
    done = False
    episode_reward, episode_steps = 0, 0
    while not done and episode_steps < MAX_STEP_PER_EPISODE:
        episode_steps += 1

        action = agent.sample(obs)
        next_obs, reward, done = env.step(action)
        terminal = float(done) if episode_steps < MAX_STEP_PER_EPISODE else 0

        agent.add_experience(obs, action, reward, next_obs, terminal)

        # compute reward of every agent
        obs = next_obs
        episode_reward += reward

        # show model effect without training
        if args.restore and args.show:
            continue

        agent.learn([agent])
    return episode_reward, episode_steps

def train_main():
    #基于paddle
    paddle.seed(args.seed)
    #np.random.seed(args.seed)
    #日志存储修改
    logger.set_dir('{}/train_log/{}'.format(args.data_dir,args.date))

    if args.continuous_actions:
        assert isinstance(env.action_space, spaces.Box)

    # build agent
    #1.建立神经网络模型
    model = MAModel(env.obs_dim, env.act_dim)
    #2.算法
    algorithm = MADDPG(
        model,
        gamma=GAMMA,
        tau=TAU,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR)
    #3.创建 agent
    agent = MAAgent(
        algorithm,
        obs_dim_n=env.obs_dim,
        act_dim_n=env.act_dim)


    #可选4.如果有模型用了，将权重直接放到神经网络里边去
    if args.restore:
        # restore modle
        model_file = args.data_dir+"/models/"+args.restore_model_dir

        if not os.path.exists(model_file):
            raise Exception(
                'model file {} does not exits'.format(model_file))
        agent.restore(model_file)

    #5更新神经网络参数
    total_steps = 0
    total_episodes = 0
    #在最大episodes之下就继续进行训练
    while total_episodes <= args.max_episodes:
        # run an episode
        episode_reward, episode_steps = run_episode(env, agent)

        summary.add_scalar('train/episode_reward_wrt_episode', episode_reward,total_episodes)
        #summary.add_scalar('train/episode_reward_wrt_step', episode_reward, total_steps)
        #这个是日志打印保持的地方
        # if total_episodes%1000==0:
        logger.info(
            'total_steps {}, episode {}, reward {},  episode steps {}'
            .format(total_steps, total_episodes, episode_reward, episode_steps))

        total_steps += episode_steps
        total_episodes += 1

        # # evaluste agents
        # if total_episodes % args.test_every_episodes == 0:
        #     eval_episode_rewards, eval_episode_steps = run_evaluate_episodes(env, agent, EVAL_EPISODES)
        #     summary.add_scalar('eval/episode_reward',np.mean(eval_episode_rewards), total_episodes)
        #     logger.info('Evaluation over: {} episodes, Reward: {}'.format(EVAL_EPISODES, np.mean(eval_episode_rewards)))
        #
        # # save model
        # if total_episodes % args.auto_model_save_frequency== 0:
        #     if args.store_model:
        #         episodes_son_dir="/"+str(total_episodes)
        #         model_dir =args.data_dir+ "/models/"+args.date +args.model_dir+episodes_son_dir
        #         os.makedirs(os.path.dirname(model_dir), exist_ok=True)
        #         model_name = '/agent_' + 'ddpg'
        #         agent.save(model_dir + model_name)
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument('--env',type=str,default='simple_spread',help='scenario of MultiAgentEnv')
    # auto save model, optional restore model
    parser.add_argument('--show', action='store_true', default=False, help='display or not')
    #save_path
    parser.add_argument('--data_dir',type=str,default='./data',help='date')
    #经常需要修改的参数
    parser.add_argument('--seed',type=int,default=1,help="It's seed!")

    parser.add_argument('--date',type=str,default='2023_5_11',help='date')

    # 测试
    parser.add_argument('--test_every_episodes',type=int,default=int(10000), help='the episode interval between two consecutive evaluations')

    #载入模型及路径
    #载入模型自己写事件
    parser.add_argument('--restore',action='store_true',default=False,help='restore model or not, must have model_dir')

    parser.add_argument('--restore_model_dir',type=str,default='2023_5_11/processCpu_4_ChangeRatioAs7/8000',help='directory for saving model')
    
    #保存模型及主文件夹路径及自动保存间隔次数    
    parser.add_argument('--store_model',action='store_true',default=False,help='store model or not, must have model_dir')

    parser.add_argument('--model_dir',type=str,default='/processCpu_4_ChangeRatioAs7',help='directory for saving model')

    parser.add_argument('--auto_model_save_frequency',type=int,default=int(1e3),help='the episode(frequency) to auto sace model file ')

    #其他参数
    parser.add_argument('--continuous_actions',action='store_true',default=True,help='use continuous action mode or not')
    parser.add_argument('--max_episodes',type=int,default=10000,help='stop condition: number of episodes')
    parser.add_argument('--use_target_model', type=bool, default=True, help='use_target_model')
    args = parser.parse_args()
    env = make_env(args.env, args.continuous_actions)
    train_main()
    #evaluate_main()
