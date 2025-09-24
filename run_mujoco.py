# !/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import FlattenObservation
from gym import spaces
from gymnasium.wrappers import TimeLimit
from gymnasium.core import Wrapper
import logging
from baselines import logger
from half_cheetah import *
from walker2d import *
import panda_gym
import matplotlib as mpl
import matplotlib.pyplot as plt

class CustomFlattenObservation(gym.Wrapper):
    def __init__(self, env, objcoeff, itschrew):
        super().__init__(env)
        self.old_position = None
        self.new = True
        self.auxEsperaInteracao = False
        self.cont = 0
        self.initial_position = np.array([0, 0, 0])
        self.phase = 0 # Fase 0 -> Ensina interagir com o objeto, Fase 1 -> Ensina a levar o objeto ao alvo
        self.objcoeff = objcoeff
        self.itschrew = itschrew
        obs_dim = 6
        # Definindo o espaço de observação para incluir observation, desired_goal e achieved_goal
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(#25 +
                   env.observation_space['observation'].shape[0] +
                   #obs_dim +
                   env.observation_space['desired_goal'].shape[0] +
                   env.observation_space['achieved_goal'].shape[0],),# +
                   #3,), #Apenas para estabelecer as dimensões do espaço adicionado
            dtype=np.float32
        )

    def reset(self, iters_so_far=0, **kwargs):
        obs, info = self.env.reset(**kwargs)
        #print('\n\n')
        #indices = np.r_[0:25] #[0:9, 20:23] para push / [0:11, 20:25] para pick and place / [0:9, 14:25] para obs_20 / Testar tambem 0:11 para slide e push
        #obs['observation'] = obs['observation'][indices]
        #obs['desired_gripper_position'] = obs['observation'][3:6]
        '''
        if iters_so_far <= 200:
            fixed_goal = np.array([1.21268572, 0.88490783, 0.42469975])
        else:
            fixed_goal = np.array([1.4173941,  0.63262762, 0.42469975])
        '''
        #fixed_goal = np.array([1.21268572,  0.88490783, 0.42469975])
        #obs["desired_goal"] = fixed_goal
        #if (iters_so_far<=300):
        #    obs["desired_goal"] = np.array([1.4173941,  0.85262762, 0.42469975])
        #else:
        #    obs["desired_goal"] = np.array([1.4173941,  0.63262762, 0.42469975])
        #obs['observation'][6:9] = [round(obs['observation'][3]-obs['observation'][0],8), round(obs['observation'][4]-obs['observation'][1],8), round(obs['observation'][5]-obs['observation'][2],8)]
        #obs['observation'][9:12] = [round(obs['desired_goal'][0]-obs['achieved_goal'][0],8), round(obs['desired_goal'][1]-obs['achieved_goal'][1],8), round(obs['desired_goal'][2]-obs['achieved_goal'][2],8)]
        #obs["observation"] = obs["observation"][:6]
        return self.flatten_obs_reset(obs)[0], obs

    def step(self, action, iters_so_far):
        obs, reward, done, truncated, info = self.env.step(action)
        
        #indices = np.r_[0:25]
        #obs['observation'] = obs['observation'][indices]
        #print(obs['observation'])
        
        #obs['desired_gripper_position'] = obs['observation'][3:6]
        goal_reward = self.env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info={})
        object_reward = self.env.compute_reward(obs["observation"][0:3], obs['observation'][3:6], info={})
        
        #reward = 1*object_reward + 1*goal_reward
        
        if iters_so_far <= self.itschrew:
            reward = self.objcoeff*object_reward + (1-self.objcoeff)*goal_reward
        else:
            reward = goal_reward
        
        if goal_reward == 0:
            reward = 0
        
        return self.flatten_obs(obs), reward, done, truncated, info, obs, self.phase #retorna também o obs original, apenas no step, para implementação do HER

    def flatten_obs(self, obs):
        # Concatenando observation, desired_goal e achieved_goal
        #obs['desired_gripper_position'] = obs['observation'][3:6]
        return np.concatenate((
            obs['observation'],
            obs['desired_goal'],
            obs['achieved_goal'],
        #    obs['desired_gripper_position']
        ))
    
    def flatten_obs_reset(self, obs):
        # Concatenando observation, desired_goal e achieved_goal
        #obs['desired_gripper_position'] = obs['observation'][3:6]
        return np.concatenate((
            obs['observation'],
            obs['desired_goal'],
            obs['achieved_goal'],
        #    obs['desired_gripper_position']
        )), []


def is_goal_env(env):
    """
    Verifica se o ambiente é um Goal Environment.
    """
    obs_space = env.observation_space
    return isinstance(obs_space, gym.spaces.Dict) and \
           all(key in obs_space.spaces for key in ["observation", "desired_goal", "achieved_goal"])

def train(env_id,num_timesteps,seed,num_options,app,saves,wsaves,epoch,w_intfc,switch,mainlr,intlr,piolr,multi,eta,render,optimsize,entcoeff,kher,hermvobj,randomact,objcoeff,itschrew,itsheroff):
    import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    goal_env = False
    
    if env_id=="AntWalls":
        from antwalls import AntWallsEnv
        env=AntWallsEnv()
        #env.seed(seed) 
    else:

        if render:
            env = gym.make(env_id, render_mode='human', max_episode_steps=50)
        else:
            #env = gym.make(env_id, max_episode_steps=200)
            #env = MountainCarGoalWrapper(env)
            env = gym.make(env_id, max_episode_steps=50)

        # Verifica se o ambiente é um goal environment
        if is_goal_env(env):
            goal_env = True
            print(f"{env_id} is a Goal Environment. Applying FlattenObservation.")
            #env = FlattenObservation(env)  # Usar somente para goal environments
            env = CustomFlattenObservation(env, objcoeff, itschrew) # Com função própria, mesmo resultado do FlattenObservation, porém mais fácil de personalizar
        #else:
        #    print(f"{env_id} não é um Goal Environment.")

        # Reinicia o ambiente para obter a observação achatada
        #env = FlattenObservation(env) #Utilizar apenas com o robotics
        
        #obs, obs_org = env.reset(seed=seed)
        obs, obs_org = env.reset() #Por algum motivo inserir seed aqui impede o sistema de convergir
        print("Observação:", obs)
        print("Obs original:", obs_org)

        #env.seed(seed)


    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2, num_options=num_options, w_intfc=w_intfc)

    gym.logger.setLevel(logging.WARN)

    print(f'num_options: {num_options}')
    print(f'seed: {seed}')

    '''
    if not multi:
        if num_options ==1:
            optimsize=64
        elif num_options ==2:
            optimsize=32
        else:
            optimsize=int(64/num_options)
    else:
        optimsize=64
    '''
    #optimsize=64 #Testar com outros seeds além do 28. Rodando 1 opt 1000 e 2 opt 512

    num_timesteps = num_timesteps
    tperbatch = 2001 #if not epoch else int(1e4)#2048 if not epoch else int(1e4) # original é 2048
    pposgd_simple.learn(env, policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=tperbatch,
            clip_param=0.2, entcoeff=entcoeff,
            optim_epochs=10, optim_stepsize=mainlr, optim_batchsize=optimsize,
            gamma=0.99, lam=0.95, schedule='constant', num_options=num_options,
            app=app, saves=saves, wsaves=wsaves, epoch=epoch, seed=seed,
            w_intfc=w_intfc,switch=switch,intlr=intlr,piolr=piolr,multi=multi,
            eta=eta,render=render,is_goal_env=goal_env, kher=kher, hermvobj=hermvobj, 
            randomact=randomact, objcoeff=objcoeff, itschrew=itschrew, itsheroff=itsheroff
        )
    '''
    original
    pposgd_simple.learn(env, policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=tperbatch,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=mainlr, optim_batchsize=optimsize,
            gamma=0.99, lam=0.95, schedule='constant', num_options=num_options,
            app=app, saves=saves, wsaves=wsaves, epoch=epoch, seed=seed,
            w_intfc=w_intfc,switch=switch,intlr=intlr,piolr=piolr,multi=multi,eta=eta,render=render,is_goal_env=goal_env
        )
    '''
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='AntWalls')
    #parser.add_argument('--timesteps', help='number of timesteps', type=int, default=2e7)#2e8) #Numero máximo de timesteps
    parser.add_argument('--iters', help='number of timesteps', type=int, default=1000)#2e8) #Numero máximo de timesteps
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--opt', help='number of options', type=int, default=2) 
    parser.add_argument('--app', help='Append to folder name', type=str, default='')        
    parser.add_argument('--saves', help='Save the returns at each iteration', dest='saves', action='store_true', default=False)
    parser.add_argument('--wsaves', help='Save the weights',dest='wsaves', action='store_true', default=False)    
    parser.add_argument('--switch', help='Switch task after 150 iterations', dest='switch', action='store_true', default=False)    
    parser.add_argument('--nointfc', help='Disables interet functions', dest='w_intfc', action='store_false', default=True)    
    parser.add_argument('--epoch', help='Load weights from a certain epoch', type=int, default=0) 
    parser.add_argument('--mainlr', type=float, default=5e-5)
    parser.add_argument('--intlr', type=float, default=5e-5)
    parser.add_argument('--piolr', type=float, default=5e-5)
    parser.add_argument('--optimsize', type=int, default=64)
    parser.add_argument('--entcoeff', type=float, default=0.00)
    parser.add_argument('--kher', type=int, default=4) #Valor e k para HER
    parser.add_argument('--hermvobj', type=int, default=1) #HER considera apenas trajetórias em que houve deslocamento o objeto
    parser.add_argument('--randomact', type=int, default=0) #Habilita random actions / 0->disabled, 1->enables, 2->with decay
    parser.add_argument('--multi', help='Multi updates', dest='multi', action='store_true', default=False)  
    parser.add_argument('--eta', type=float, default=0.1, help='trade off updates') #Probabilidade de atualizar todas as options -> procurar aumentar, o artigo usa 0.7 e 0.9
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--objcoeff', type=float, help='coefficient of object_reward (0-1)', default=1)
    parser.add_argument('--itschrew', type=int, help='iters to disable object_reward', default=150)
    parser.add_argument('--itsheroff', type=int, help='iters to disble her', default=300)

    args = parser.parse_args()

    print('\n**************************************************\n', 
        args, '\n**************************************************\n')

    train(args.env, num_timesteps=args.iters*2001, seed=args.seed, num_options=args.opt, app=args.app,
     saves=args.saves, wsaves=args.wsaves, epoch=args.epoch,w_intfc=args.w_intfc,
     switch=args.switch,mainlr=args.mainlr,intlr=args.intlr,piolr=args.piolr,multi=args.multi, eta=args.eta, 
     render=args.render, optimsize=args.optimsize, entcoeff=args.entcoeff, kher=args.kher, hermvobj=args.hermvobj, 
     randomact=args.randomact, objcoeff=args.objcoeff, itschrew=args.itschrew, itsheroff=args.itsheroff)


if __name__ == '__main__':
    main()
