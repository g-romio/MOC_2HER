from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
import math

from typing import Any, Optional, Union, Dict
from baselines.common.mpi_adam import MpiAdam

from enum import Enum

MPI=None
# from mpi4py import MPI
from collections import deque
import os
import shutil
from scipy import spatial
import gymnasium as gym
import matplotlib.pyplot as plt
import random

from baselines.her import her #Após salvar as pastas HER, comando "pip install -e ." dentro de CONTINUOUS_CONTROL_HER

her_enabled = 0

from tensorflow.keras.layers import Input, Reshape, Dense, concatenate, Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam


# ===== Implementação do ICM =====
class ICM:
    def __init__(self, env, training_games=100,
                 game_steps=1000, goal_steps=201, beta=0.01, lmd=0.99):
        
        self.env=env #import env
        self.state_shape=env.observation_space.shape[0] # the state space
        self.action_shape=env.action_space.shape[0] # the action space
        
        self.lmd=lmd # ratio of the external loss against the intrinsic reward
        self.beta=beta # ratio of the inverse loss against the forward reward 

        self.training_games=training_games #N training games
        self.goal_steps=goal_steps # N training steps
        self.batch_size=1000 # batch size for training the model

        self.model=self.build_icm_model() #build ICM
        self.model.compile(optimizer=Adam(), loss="mse") #Complies ICM
        
        self.positions=np.zeros((self.training_games,2)) #record learning process
        self.rewards=np.zeros(self.training_games) #record learning process

    '''
    def one_hot_encode_action(self, action):
        #from int to one hot encode
        action_encoded=np.zeros(self.action_shape, np.float32)
        action_encoded[action]=1
        return action_encoded
    '''

    ## BUILD ICM ##
    def inverse_model(self, output_dim=4):
        """
        Predict the action (a_t)
        via the current and next states (s_t, s_t+1)
        """
        def func(f_t, f_t1): 
            #f_t, f_t1 describe the feature vector of s_t and s_t+1, respectively
            inverse_net=concatenate([f_t, f_t1])
            inverse_net=Dense(24, activation='relu')(inverse_net)
            inverse_net=Dense(output_dim, activation='linear')(inverse_net)
            return inverse_net
        return func

    def forward_model(self, output_dim=2):
        """
        Predict the next state (s_t+1)
        via the current state and sction  (s_t, a_t)
        """
        def func(f_t, a_t):
            #f_t describe the feature vector of s_t
            forward_net=concatenate([f_t, a_t])
            forward_net=Dense(24, activation='relu')(forward_net)
            forward_net=Dense(output_dim, activation='linear')(forward_net)
            return forward_net
        return func

    def create_feature_vector(self, input_shape):

        model=Sequential()
        model.add(Dense(24, input_shape=input_shape, activation="relu"))
        model.add(Dense(12, activation="relu"))
        model.add(Dense(2, activation='linear', name='feature'))

        return model

    def build_icm_model(self, state_shape=(16,), action_shape=(4,)):
        ## Main ICM network
        s_t=Input(shape=state_shape, name="state_t") # (2,)
        s_t1=Input(shape=state_shape, name="state_t1") # (2,)
        a_t=Input(shape=action_shape, name="action") # (3,)

        #reshape=Reshape(target_shape= (2,))

        feature_vector_map=self.create_feature_vector((16,))
        #fv_t=feature_vector_map(reshape(s_t))
        #fv_t1=feature_vector_map(reshape(s_t1))
        fv_t = feature_vector_map(s_t)  # Não é necessário reshape
        fv_t1 = feature_vector_map(s_t1)  # Não é necessário reshape

        a_t_hat=self.inverse_model()(fv_t, fv_t1)
        s_t1_hat=self.forward_model()(fv_t, a_t)

        # the intrinsic reward refelcts the diffrence between
        # the next state versus the predicted next state
        # $r^i_t = \frac{\nu}{2}\abs{\hat{s}_{t+1}-s_{t+1})}^2$
        int_reward=Lambda(lambda x: 0.5 * K.sum(K.square(x[0] - x[1]), axis=-1),
                     output_shape=(1,),
                     name="reward_intrinsic")([fv_t1, s_t1_hat])

        #inverse model loss
        #inv_loss=Lambda(lambda x: -K.sum(x[0] * K.log(x[1] + K.epsilon()), 
        #                                 axis=-1),
        #            output_shape=(1,))([a_t, a_t_hat])
        inv_loss = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=-1), output_shape=(1,))([a_t, a_t_hat])

        # combined model loss - beta weighs the inverse loss against the
        # rwd (generate from the forward model)
        loss=Lambda(lambda x: self.beta * x[0] + (1.0 - self.beta) * x[1],
                    output_shape=(1,))([int_reward, inv_loss])
        #
        # lmd is lambda, the param the weights the importance of the policy
        # gradient loss against the intrinsic reward
        rwd=Input(shape=(1,))
        loss=Lambda(lambda x: (-self.lmd * x[0] + x[1]), 
                    output_shape=(1,))([rwd, loss])

        return Model([s_t, s_t1, a_t, rwd], loss)
    '''
    def learn(self, prev_states, states, actions, rewards):
        #batch train the network
        s_t=prev_states
        s_t1=states
    
        actions=np.array(actions)
    
        icm_loss=self.model.train_on_batch([s_t, s_t1,
                                        np.array(actions),
                                            np.array(rewards).reshape((-1, 1))],
                                            np.zeros((self.batch_size,)))
    '''
    # Função ajustada pelo chat gpt
    def learn(self, prev_states, states, actions, rewards):
        # Assegura que estados estão em arrays do tipo float32
        s_t = np.array(prev_states, dtype=np.float32)
        s_t1 = np.array(states, dtype=np.float32)
        
        # Codifica as ações como one-hot
        #one_hot_actions = np.array([self.one_hot_encode_action(a) for a in actions], dtype=np.float32)
        one_hot_actions = np.array(actions, dtype=np.float32)
        
        # Assegura que recompensas estão na forma (batch_size, 1)
        rewards = np.array(rewards, dtype=np.float32).reshape((-1, 1))

        # Rótulos fictícios para perda (modelo final retorna um escalar por amostra)
        dummy_targets = np.zeros((self.batch_size,), dtype=np.float32)

        # Treinamento em batch
        icm_loss = self.model.train_on_batch(
            [s_t, s_t1, one_hot_actions, rewards],
            dummy_targets
        )

        return icm_loss

    def get_intrinsic_reward(self, x):
        ## x -> [prev_state, state, action]
        return K.function([self.model.get_layer("state_t").input,
                       self.model.get_layer("state_t1").input,
                       self.model.get_layer("action").input],
                      [self.model.get_layer("reward_intrinsic").output])(x)[0]

class OptionStepCounter:
    def __init__(self, num_options):
        # Contagem de timesteps por option
        self.option_timestep_counts = {option: 0 for option in range(num_options)}

    def incCount(self, option):
        # Incrementa o contador da opção especificada
        if option in self.option_timestep_counts:
            self.option_timestep_counts[option] += 1
        else:
            raise ValueError(f"Option {option} is not recognized")

    def getCount(self, option):
        # Retorna o contador da opção especificada
        if option in self.option_timestep_counts:
            return self.option_timestep_counts[option]
        else:
            raise ValueError(f"Option {option} is not recognized")

    def resetCount(self):
        # Incrementa o contador da opção especificada
        for option in self.option_timestep_counts:
            self.option_timestep_counts[option] = 0

def apply_her_v3(transitions, env, icm, k, hermvobj, objcoeff, itschrew, itsheroff):
    augmented_transitions = []
    totalTimesteps = 0

    try:
        initial_position = transitions[0]["state"][3:6]
        final_position = transitions[-1]["state"][3:6]

        #Separa o observation space atual em obs, desired_goal e achieved_goal
        aux_t = transitions[-1] #Seleciona o último elemento como o goal (Nesse ponto, é apenas para referencia para pegar o split_sizes)
        #next_state serve apenas para referencia de tamanho
        #split_sizes = [aux_t["next_state"].size, aux_t["desired_goal"].size, aux_t["achieved_goal"].size, aux_t["desired_gripper_position"].size]
        split_sizes = [aux_t["next_state"].size, aux_t["desired_goal"].size, aux_t["achieved_goal"].size]
        cumsum_splits = np.cumsum(split_sizes)
        #s1, s2, s3, s4 = cumsum_splits[0], cumsum_splits[1], cumsum_splits[2], cumsum_splits[3]
        s1, s2, s3 = cumsum_splits[0], cumsum_splits[1], cumsum_splits[2]

        #if (hermvobj==0 or (not np.allclose(initial_position, final_position, atol=5e-3))) and (iters_so_far <= 220):
        if (hermvobj==0 or (not np.allclose(initial_position, final_position, atol=1e-4))) and (iters_so_far <= itsheroff):

            #Computa decaimento do k_her
            k = math.ceil(k*(1-(iters_so_far/itsheroff)))

            #if (not np.allclose(initial_position, final_position, atol=5e-3)):
            #if True:

            #print('\n')
            #for t in transitions:
            #    print(t["achieved_goal"])

            #print('\n')
            #for t in transitions:
            #    print(t["achieved_goal"])

            #if (iters_so_far >= itsheroff):
            #    k=1

            for j in range(k):

                # ===== Estrategia "future" ORIGINAL
                for t_idx, t in enumerate(transitions): #Estrategia "future"
                    '''
                    # ===== Implementação de HER -> gripper-object=====
                    #future_timestep = np.random.randint(t_idx, len(transitions)) #Para future strategy
                    #future_timestep = 49 #Para final strategy. Verificar também final para goal e future para oject

                    #future_timestep = np.random.randint(t_idx, min(t_idx + 15, len(transitions))) #-> new
                    #hindsight_object = transitions[future_timestep]["state"][0:3] #new double HER / É "STATE" OU "NEXT_STATE"??????????

                    # É "STATE" OU "NEXT_STATE"??????????
                    #next_state_reconstructed, desired_goal_reconstructed, achieved_goal_reconstructed, desired_gripper_position_reconstructed = np.split(t["state"], np.cumsum(split_sizes)[:-1])
                    next_state_reconstructed = t["state"][:s1]
                    desired_goal_reconstructed = t["desired_goal"]#t["state"][s1:s2]
                    achieved_goal_reconstructed = t["achieved_goal"]#t["state"][s2:s3]

                    #desired_gripper_position_reconstructed = hindsight_object
                    #object_reward = env.compute_reward(t["state"][0:3], hindsight_object, info={}) #ORIGINAL

                    #print('\n')
                    #print(t["next_state"][3:6])
                    #print(t["achieved_goal"])                

                    # ===== Implementação de Double-HER -> object-goal=====
                    #future_timestep = np.random.randint(t_idx, len(transitions)) #-> old / Teste para redução processamento
                    future_timestep = np.random.randint(future_timestep, len(transitions)) # Esse método funciona muito melhor com o Push (mudar para final)
                    hindsight_goal = transitions[future_timestep]["achieved_goal"] #new double HER
                    goal_reward = env.compute_reward(t["achieved_goal"], hindsight_goal, info={})
                    #
                    #distance = np.linalg.norm(t["achieved_goal"] - hindsight_goal)
                    #goal_reward = -distance
                    #
                    #new_reward = 0.5*object_reward + 0.5*goal_reward
                    '''
                    next_state_reconstructed = t["state"][:s1]
                    future_timestep = np.random.randint(t_idx, len(transitions))
                    #desired_goal_reconstructed = t["desired_goal"]#t["state"][s1:s2]
                    #achieved_goal_reconstructed = t["achieved_goal"]#t["state"][s2:s3]

                    if iters_so_far <= itschrew:
                        #future_timestep = np.random.randint(t_idx, len(transitions)) #Para future strategy                        
                        hindsight_object = transitions[future_timestep]["state"][0:3]
                        object_reward = env.compute_reward(t["state"][0:3], hindsight_object, info={}) #ORIGINAL

                        future_timestep = np.random.randint(future_timestep, len(transitions)) # Esse método funciona muito melhor com o Push
                        hindsight_goal = transitions[future_timestep]["achieved_goal"] #new double HER
                        goal_reward = env.compute_reward(t["achieved_goal"], hindsight_goal, info={})
                    
                        new_reward = objcoeff*object_reward + (1-objcoeff)*goal_reward
                        arrBlockGripperPos = [round(hindsight_object[0]-t['state'][0],8), round(hindsight_object[1]-t['state'][1],8), round(hindsight_object[2]-t['state'][2],8)]
                        next_state_reconstructed[3:6] = hindsight_object
                        next_state_reconstructed[6:9] = arrBlockGripperPos
                        #next_state_reconstructed[11:14] = [0, 0, 0]
                        #next_state_reconstructed[14:17] = -1*next_state_reconstructed[20:23] #Tambem funciona sem, mas fica melhor com
                        #next_state_reconstructed[17:20] = [0, 0, 0]
                    else:
                        future_timestep = np.random.randint(future_timestep, len(transitions)) # Esse método funciona muito melhor com o Push (Descomentar caso não apresente bons resultados)
                        hindsight_goal = transitions[future_timestep]["achieved_goal"] #new double HER
                        goal_reward = env.compute_reward(t["achieved_goal"], hindsight_goal, info={})

                        new_reward = goal_reward
                        #hindsight_object = t["achieved_goal"]

                    # ===== Calcula e atualiza no espaço de observação distancia entre gripper e block =====
                    #arrBlockGripperPos = [round(hindsight_object[0]-t['state'][0],8), round(hindsight_object[1]-t['state'][1],8), round(hindsight_object[2]-t['state'][2],8)]
                    #next_state_reconstructed[3:6] = hindsight_object
                    #next_state_reconstructed[6:9] = arrBlockGripperPos

                    if goal_reward == 0:
                        new_reward = 0
                    
                    #if (random.random() < 1/10): # Aleatoriedade para aplicação o HER; Obs. Para versão final, não executar nada do código anterior dentro da função do HER, para ganhar tempo de execução
                    new_transition = {
                        # "state": np.concatenate(( #next_observation
                        #     next_state_reconstructed,
                        #     #hindsight_goal, #Original
                        #     desired_goal_reconstructed, #MODIFICADA para fetch push
                        #     hindsight_goal, #achieved_goal_reconstructed,
                        #     hindsight_object #desired_gripper_position_reconstructed
                        # )),
                        "state": np.concatenate(( #next_observation
                            next_state_reconstructed,
                            hindsight_goal, #desired_goal #Esse nao funciona
                            t["achieved_goal"], #achieved_goal,
                        #    hindsight_object #desired_gripper_position
                        )),
                        #"state": t["state"],
                        "new": t["new"],
                        "option": t["option"],
                        "last_option": t["last_option"],
                        "prevac": t["prevac"],
                        "action": t["action"],
                        "reward": new_reward,
                        # "next_state": next_state_reconstructed,
                        # "achieved_goal": achieved_goal_reconstructed,
                        # "desired_goal": desired_goal_reconstructed,
                        # "her": 1,
                        # "observation": np.concatenate(( #next_observation
                        #     t['next_state'],
                        #     t['desired_goal'],
                        #     t['achieved_goal']
                        # )),
                    }
                    #print('\n',new_transition["state"])
                    augmented_transitions.append(new_transition)

    except Exception as e:
        print(f"Erro ao aplicar HER: {e}")

    return augmented_transitions

def traj_segment_generator(pi,env,horizon,stochastic,num_options,saves,rewbuffer,epoch,seed,w_intfc,switch,gamma,eta,option_counter,render,is_goal_env,kher,hermvobj,randomact,objcoeff,itschrew,itsheroff):
    
    her_state = np.empty((0,))
    her_new = np.empty((0,))
    her_option = np.empty((0,))
    her_last_option = np.empty((0,))
    her_prevac = np.empty((0,))
    her_action = np.empty((0,))
    her_reward = np.empty((0,))
    # her_next_state = np.empty((0,))
    # her_achieved_goal = np.empty((0,))
    # her_desired_goal = np.empty((0,))
    # her_observation = np.empty((0,))
    # her_applied = np.empty((0,))

    cont_episodes = 0
    cont_solved = 0

    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob, obs_orig = env.reset(seed=seed)
    global her_enabled

    # ICM
    icm = ICM(env)
    icm_episode_rewards = []    
    icm_prev_states = []
    icm_states = []
    icm_actions = []
    icm_rewards = []

    '''
    if hasattr(env,'NAME')and env.NAME=='AntWalls':
        switch_iter = 240
    elif env.spec.id == 'HalfCheetahDir-v1':
        switch_iter = 150
    elif env.spec.id == 'Walker2dStand-v1':
        switch_iter = 240
    else: #Para outros ambientes (valor qualquer, deve ser ajustado, no momento é apenas a declaração, o env não altera o goal)
        switch_iter = 240
    '''

    #render=0
    iters_so_far=0

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of completed episodes in this segment

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    realrews = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    opts = np.zeros(horizon, 'int32')
    activated_options = np.zeros((horizon, num_options), 'float32')
    last_options=np.zeros(horizon, 'int32')

    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    print("ob: ", ob)
    option,active_options_t = pi.get_option(ob)
    last_option=option


    ep_states=[[] for _ in range(num_options)] 
    ep_states[option].append(ob)
    ep_states_term=[[] for _ in range(num_options)] 
    ep_num =0
    episode_transitions = []

    opt_duration = [[] for _ in range(num_options)]
    curr_opt_duration = 0.

    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

    is_slide = base_env.spec.id.startswith("FetchSlide")
    if is_slide:
        dist_solved = 0.2
    else:
        dist_solved = 0.07
    print('dist_solved: ', dist_solved)
    
    while True:
        prevac = ac
        ac = pi.act(stochastic, ob, option)
        #print(f'\nAção original: {ac}')

        # ===== Acrescenta ruído e random action ao sistema =====
        #ac = ac.cpu().numpy().squeeze()
        '''
        if iters_so_far <= 100:
            #print(f'Com ruído gaussiano ({iters_so_far})')
            ac += 0.2 * np.random.randn(*ac.shape) #Acrescenta ruído gaussiano
        #else:
            #print(f'Sem ruído gaussiano ({iters_so_far})')

        # # Probabilidade independente de aplicar random action
            if np.random.rand() < 0.3:
                ac = np.random.uniform(low=-1.0, high=1.0, size=ac.shape)

        elif iters_so_far <= 200:
            ac += 0.1 * np.random.randn(*ac.shape) #Acrescenta ruído gaussiano
            if np.random.rand() < 0.15:
                ac = np.random.uniform(low=-1.0, high=1.0, size=ac.shape)
        else:
            if np.random.rand() < 0.05:
                ac = np.random.uniform(low=-1.0, high=1.0, size=ac.shape)
        '''
        
        if randomact==1 and iters_so_far <= 75:
            if np.random.rand() < 0.3: #Ver para desabilitar após 200 épocas
                ac = np.random.uniform(low=-1.0, high=1.0, size=ac.shape)
        elif randomact==2: #Random actions com taxa de decaimento
            if iters_so_far <= 25:
                if np.random.rand() < 1:
                    ac = np.random.uniform(low=-1.0, high=1.0, size=ac.shape)
            elif iters_so_far <= 50:
                if np.random.rand() < 0.75:
                    ac = np.random.uniform(low=-1.0, high=1.0, size=ac.shape)
            elif iters_so_far <= 75:
                if np.random.rand() < 0.50:
                    ac = np.random.uniform(low=-1.0, high=1.0, size=ac.shape)
            elif iters_so_far <= 125:
                if np.random.rand() < 0.3:
                    ac = np.random.uniform(low=-1.0, high=1.0, size=ac.shape)
            else:
                if np.random.rand() < 0.1:
                    ac = np.random.uniform(low=-1.0, high=1.0, size=ac.shape)

        #else:
        #    if np.random.rand() < 0.1: #Ver para desabilitar após 200 épocas
        #        ac = np.random.uniform(low=-1.0, high=1.0, size=ac.shape)

        '''
        if iters_so_far <= 50:
            kher = 8
        elif iters_so_far <= 75:
            kher = 7
        elif iters_so_far <= 100:
            kher = 6
        elif iters_so_far <= 150:
            kher = 5
        elif iters_so_far <= 200:
            kher = 4
        elif iters_so_far <= 250:
            kher = 3
        elif iters_so_far <= 300:
            kher = 2
        elif iters_so_far <= 350:
            kher = 1
        '''
        
        # # Clipa sempre no final
        #ac = np.clip(ac, -1.0, 1.0)
        #print(f'Ação final: {ac}')
        #if np.random.rand() < 0.2:
        #    ac = np.random.uniform(low=-1.0, high=1.0, size=ac.shape)
        
        #ac = np.clip(ac, -1.0, 1.0)

        # if np.any(np.isnan(ac)) or np.any(np.isinf(ac)):
        #     #print(f"Ação inválida: {ac}")
        #     ob, _ = env.reset(seed=seed)
        #     ac = np.zeros_like(ac)  # ou alguma ação neutra
        
        #Acrescenta ruído normal à ação -> Sem diferenças significativas
        #noise = np.random.normal(loc=0, scale=0.2, size=ac.shape)
        # ac = ac + noise

        #print(f"Passo: {t}, Opção selecionada: {option}")
        #option_timestep_counts[option] += 1 #Incrementa contador da option
        option_counter.incCount(option)
        #option_counter = OptionStepCounter(num_options)
        if render:
        #if t>1000000:
            #option=1
            env.render()
            time.sleep(0.05)
            print(option)#,cur_ep_ret)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:

            # ===== Aplicação e cálculos do HER =====

            # ===== PROVISORIO =====
            augmented_transitions = apply_her_v3(episode_transitions, env, icm, kher, hermvobj, objcoeff, itschrew, itsheroff)


            # ===== Incrementa todos os buffers para tratamento subsequente
            if (len(augmented_transitions)>0): #Retornou algum elemenso, seja por haver transição ou pela execução aleatória interna da função apply_her
                # Atualiza her_state
                if her_state.size == 0:
                    her_state = np.array([transition["state"] for transition in augmented_transitions])
                else:
                    her_state = np.vstack([her_state] + [transition["state"] for transition in augmented_transitions])
                # Atualiza her_new (obs. como um vetor 1D)
                if her_new.size == 0:
                    her_new = np.array([transition["new"] for transition in augmented_transitions]).flatten().astype(int)
                else:
                    her_new = np.hstack([her_new, [transition["new"] for transition in augmented_transitions]]).astype(int)
                # Atualiza her_option
                if her_option.size == 0:
                    her_option = np.array([transition["option"] for transition in augmented_transitions]).flatten().astype(int)
                else:
                    her_option = np.hstack([her_option] + [transition["option"] for transition in augmented_transitions]).astype(int)
                # Atualiza her_last_option
                if her_last_option.size == 0:
                    her_last_option = np.array([transition["last_option"] for transition in augmented_transitions]).flatten().astype(int)
                else:
                    her_last_option = np.hstack([her_last_option] + [transition["last_option"] for transition in augmented_transitions]).astype(int)
                # Atualiza her_prevac
                if her_prevac.size == 0:
                    her_prevac = np.array([transition["prevac"] for transition in augmented_transitions])
                else:
                    her_prevac = np.vstack([her_prevac] + [transition["prevac"] for transition in augmented_transitions])
                # Atualiza her_action
                if her_action.size == 0:
                    her_action = np.array([transition["action"] for transition in augmented_transitions])
                else:
                    her_action = np.vstack([her_action] + [transition["action"] for transition in augmented_transitions])
                # Atualiza her_reward
                if her_reward.size == 0:
                    her_reward = np.array([transition["reward"] for transition in augmented_transitions]).flatten()
                else:
                    her_reward = np.hstack([her_reward] + [transition["reward"] for transition in augmented_transitions])
                # # Atualiza her_next_state
                # if her_next_state.size == 0:
                #     her_next_state = np.array([transition["next_state"] for transition in augmented_transitions])
                # else:
                #     her_next_state = np.vstack([her_next_state] + [transition["next_state"] for transition in augmented_transitions])
                # # Atualiza her_achieved_goal
                # if her_achieved_goal.size == 0:
                #     her_achieved_goal = np.array([transition["achieved_goal"] for transition in augmented_transitions])
                # else:
                #     her_achieved_goal = np.vstack([her_achieved_goal] + [transition["achieved_goal"] for transition in augmented_transitions])
                # # Atualiza her_desired_goal
                # if her_desired_goal.size == 0:
                #     her_desired_goal = np.array([transition["desired_goal"] for transition in augmented_transitions])
                # else:
                #     her_desired_goal = np.vstack([her_desired_goal] + [transition["desired_goal"] for transition in augmented_transitions])
                # # Atualiza her_reward
                # if her_applied.size == 0:
                #     her_applied = np.array([transition["her"] for transition in augmented_transitions]).flatten()
                # else:
                #     her_applied = np.hstack([her_applied] + [transition["her"] for transition in augmented_transitions])

                # Atualiza her_desired_goal
                # if her_observation.size == 0:
                #     her_observation = np.array([transition["observation"] for transition in augmented_transitions])
                # else:
                #     her_observation = np.vstack([her_observation] + [transition["observation"] for transition in augmented_transitions])

                # Exibe o estado atual de todos os arrays
                # print(f"her_state: {her_state.shape}")
                # print(f"her_new: {her_new.shape}")
                # print(f"her_option: {her_option.shape}")
                # print(f"her_last_option: {her_last_option.shape}")
                # print(f"her_prevac: {her_prevac.shape}")
                # print(f"her_action: {her_action.shape}")
                # print(f"her_reward: {her_reward.shape}")
                # print(f"her_next_state: {her_next_state.shape}")
                # print(f"her_achieved_goal: {her_achieved_goal.shape}")
                # print(f"her_desired_goal: {her_desired_goal.shape}")
                # print(f"her_observation: {her_observation.shape}")

            episode_transitions = []  # Reseta transições para o próximo episódio        

            # ==========

            # Adicionando dados às variáveis com a soma dos elementos
            # obs_2 = np.concatenate((obs, her_state), axis=0)

            # print(f"{type(her_state)} / {len(her_state)} elements / her_state: {her_state}")
            # print(f"{type(obs_2)} / {len(obs_2)} elements / obs_2: {obs_2}")

            # ===== Substituição dos elementos pelos her_elements =====
            # obs = her_state
            # news = her_new
            # opts = her_option
            # last_options = her_last_option
            # prevacs = her_prevac
            # acs = her_action
            # rews = her_reward
            # realrews = her_reward
            # ==========

            #print(f"her_buffer_len: {len(her_state)}")
            #print(f"Original_len: {len(obs)}")

            # ===== Concatenação dos elementos her_elements ao buffer original =====
            # Armazena os valores originais para considerar o her buffer apenas nessa etapa
            backup_obs = obs
            backup_news = news
            backup_opts = opts
            backup_last_options = last_options
            backup_prevacs = prevacs
            backup_acs = acs
            backup_rews = rews
            backup_realrews = realrews

            # ===== Comentado para substituição
            #if (len(augmented_transitions)>0):
            try:
                obs = np.concatenate((her_state, obs), axis=0)
                news = np.concatenate((her_new, news), axis=0)
                opts = np.concatenate((her_option, opts), axis=0)
                last_options = np.concatenate((her_last_option, last_options), axis=0)
                prevacs = np.concatenate((her_prevac, prevacs), axis=0)
                acs = np.concatenate((her_action, acs), axis=0)
                rews = np.concatenate((her_reward, rews), axis=0)
                realrews = np.concatenate((her_reward, realrews), axis=0)
            except Exception as e:
                print(f"Erro ao aplicar HER: {e}")

            # conc_obs = np.concatenate((her_state, obs), axis=0)
            # conc_news = np.concatenate((her_new, news), axis=0)
            # conc_opts = np.concatenate((her_option, opts), axis=0)
            # conc_last_options = np.concatenate((her_last_option, last_options), axis=0)
            # conc_prevacs = np.concatenate((her_prevac, prevacs), axis=0)
            # conc_acs = np.concatenate((her_action, acs), axis=0)
            # conc_rews = np.concatenate((her_reward, rews), axis=0)
            # conc_realrews = np.concatenate((her_reward, realrews), axis=0)
            # ==========
            
            #print(f"Final_len: {len(obs)}")


            vpreds, op_vpreds, vpred, op_vpred, op_probs, intfc, pi_I = pi.get_allvpreds(obs, ob)
            term_ps, term_p, all_term_ps = pi.get_alltpreds(obs, ob)
            last_betas=term_ps[range(len(last_options)),last_options]

            all_opts = np.append(opts,option)
            term_ratios=np.zeros((len(all_opts),num_options))
            for o in range(num_options):
                one_hot = np.zeros(len(all_opts))
                one_hot[np.where(all_opts==o)] = 1.
                term_ratios[:,o]=(all_term_ps[:,o] * pi_I[range(len(all_opts)),all_opts] + (1-all_term_ps[:,o]) * one_hot)
            term_ratios = np.log(term_ratios[1:]) - np.log(term_ratios[range(1,len(all_opts)),all_opts[:-1]][...,None])
            

            logps = np.zeros( (len(obs),num_options))
            for o in range(num_options):
                logps[:,o] = pi._logps(True,obs,[o],acs)[0]
            action_ratios = logps - logps[range(len(obs)), opts][...,None]
            
            prev_action_ratios = np.vstack((action_ratios[0],action_ratios[:-1])) # a little bias here

            last_options_onehot = np.zeros((len(last_options),num_options))
            last_options_onehot[range(len(last_options)),last_options] = 1.
            prob_curr_opt = last_betas[...,None] * pi_I[:-1] + (1-last_betas[...,None]) * last_options_onehot
            prob_prev_opt = np.vstack((prob_curr_opt[0],prob_curr_opt[:-1])) # a little bias here

            sampled_eta = float(np.random.rand()<eta)
            options_onehot = np.zeros((len(opts),num_options))
            options_onehot[range(len(opts)),opts] = 1.
            prob_curr_opt = sampled_eta * prob_curr_opt + (1-sampled_eta) * options_onehot
            prob_prev_opt= sampled_eta * prob_prev_opt + (1-sampled_eta) * last_options_onehot

            yield {"ob" : obs, "rew" : rews, "realrew": realrews, "vpred" : vpreds, "op_vpred": op_vpreds, "new" : news,
                    "ac" : acs, "opts" : opts, "opt": option, "prevac" : prevacs, "nextvpred": vpred * (1 - new), "nextop_vpred": op_vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, 'term_p': term_ps, 'next_term_p':term_p,
                     "op_probs":op_probs, "last_betas":last_betas, "intfc":intfc, 
                      "action_ratios": action_ratios, "term_ratios":term_ratios, "prev_action_ratios": prev_action_ratios,
                      "last_options": last_options, "last_option":last_option, "prob_curr_opt": prob_curr_opt, "prob_prev_opt":prob_prev_opt, "cont_episodes":cont_episodes, "cont_solved":cont_solved, "opt_dur": opt_duration}

            ep_rets = []
            ep_lens = []
            #opt_duration = [[] for _ in range(num_options)]
            #curr_opt_duration = 0.
            iters_so_far+=1
            cont_episodes = 0
            cont_solved = 0

            ###### Switching Goal ##########
            '''
            if iters_so_far==switch_iter and switch:
                # import pdb;pdb.set_trace()
                if hasattr(env,'NAME') and env.NAME=='AntWalls': # Switch the goal for AntWalls
                    from antwalls import AntWallsEnv
                    env=AntWallsEnv(num_walls=2)
                    env.seed(seed) 
                elif env.spec.id == 'HalfCheetahDir-v1':
                    env.env.env.reset_task({'direction':-1})
                elif env.spec.id == 'Walker2dStand2-v1':
                    env.env.reset_task('run')
            '''
            ################################

            # print(f"{type(obs)} / {len(obs)} elements / obs: {obs}")
            # print(f"{type(her_state)} / {len(her_state)} elements / her_state: {her_state}")

            # print(f"{type(news)} / {len(news)} elements / news: {news}")
            # print(f"{type(her_new)} / {len(her_new)} elements / her_new: {her_new}")

            # print(f"{type(opts)} / {len(opts)} elements / opts: {opts}")
            # print(f"{type(her_option)} / {len(her_option)} elements / her_option: {her_option}")

            # print(f"{type(last_options)} / {len(last_options)} elements / last_options: {last_options}")
            # print(f"{type(her_last_option)} / {len(her_last_option)} elements / her_last_option: {her_last_option}")

            # print(f"{type(prevacs)} / {len(prevacs)} elements / prevacs: {prevacs}")
            # print(f"{type(her_prevac)} / {len(her_prevac)} elements / her_prevac: {her_prevac}")

            # print(f"{type(acs)} / {len(acs)} elements / acs: {acs}")
            # print(f"{type(her_action)} / {len(her_action)} elements / her_action: {her_action}")

            # print(f"{type(rews)} / {len(rews)} elements / rews: {rews}")
            # print(f"{type(her_reward)} / {len(her_reward)} elements / her_reward: {her_reward}")

            #Reset das variáveis referentes ao HER buffer
            her_state = np.empty((0,)) #substituir por obs
            her_new = np.empty((0,)) #substituir por news
            her_option = np.empty((0,)) #substituir por opts
            her_last_option = np.empty((0,)) #substituir por last_options
            her_prevac = np.empty((0,)) #substituir por prevacs
            her_action = np.empty((0,)) #substituir por acs
            her_reward = np.empty((0,)) #substituir por rews obs. realrews=rews
            #her_applied = np.empty((0,))

            # #Não são utilizados
            # her_next_state = np.empty((0,))
            # her_achieved_goal = np.empty((0,))
            # her_desired_goal = np.empty((0,))
            # her_observation = np.empty((0,))

            # ===== retorna as trajetórias originais, desconsiderando valores gerados pelo her buffer
            obs = backup_obs
            news = backup_news
            opts = backup_opts
            last_options = backup_last_options
            prevacs = backup_prevacs
            acs = backup_acs
            rews = backup_rews
            realrews = backup_realrews

        i = t % horizon
        obs[i] = ob
        last_options[i]=last_option

        news[i] = new
        opts[i] = option
        acs[i] = ac
        prevacs[i] = prevac
        activated_options[i] = active_options_t

        ## RL loop ##
        #print(f"Ação no passo {t}: {ac}, tipo: {type(ac)}, forma: {np.shape(ac)}")
        if not isinstance(env.unwrapped, gym.envs.mujoco.MujocoEnv): # Para ambientes que não são mujoco, deve-se converter o action space
            ac = ac[0]

        if (is_goal_env): #obs_orig retorna o observation space original (observation, desired_goal, achieved_goal) para implementação do HER
            state = ob #Get state before action
            #state_org = obs_orig
            ob, rew, done, truncated, _, obs_orig, phase = env.step(ac,iters_so_far)

            # ========== Recompensa interação com o objeto
            '''
            if(iters_so_far <= 100): #Apenas durante as primeiras iterações
                #print(f"Treinamento com recompensa por interação com o objeto... ep.{iters_so_far}")
                if (episode_transitions == []):
                    initial_position = obs_orig["achieved_goal"]

                final_position = obs_orig["achieved_goal"]

                if not np.allclose(initial_position, final_position, atol=1e-2):
                    rew = 1
                else:
                    rew = 0
            elif not her_enabled:
                her_enabled = 1

            #print(f"Posição inicial: {initial_position} / Posição final: {final_position}")

            #if (random.random() < 4/10): #Realiza a aplicação do HER a cada x% de trajetórias
            #if not np.allclose(initial_position, final_position, atol=1e-2):
            #if not np.array_equal(initial_position, final_position):
            '''
            # ==========

            #print(f'her obs: {state}')
            #print(f"her next_obs: {obs_orig['observation']}")
            '''
            # ===== ICM
            s_t = state
            s_t1 = ob
            a_t_onehot = ac
            r_ext = rew
            s_t_batch = s_t.reshape(1, -1)  # Lote de tamanho 1, forma (1, 16)
            s_t1_batch = s_t1.reshape(1, -1)  # Lote de tamanho 1, forma (1, 16)
            a_t_batch = a_t_onehot.reshape(1, -1)  # Lote de tamanho 1, forma (1, 4)
            # Predição com batch de tamanho 1
            int_reward = icm.model.predict([s_t_batch, s_t1_batch, a_t_batch, r_ext.reshape(1, -1)])  # r_ext pode precisar de reshape também
            #print(f'\nOrig_rew: {rew}')
            rew = rew + 0.2*int_reward[0][0]
            #print(f'New_rew: {rew}')

            # Aprendizado
            # Guarda dados
            icm_prev_states.append(state)
            icm_states.append(ob)
            icm_actions.append(ac)
            icm_rewards.append(rew)

            # Quando atingir batch_size, treina ICM
            if len(icm_prev_states) >= icm.batch_size:
            #if False:
                print('iniciando aprendizado icm...')
                icm.learn(icm_prev_states, icm_states, icm_actions, icm_rewards)

                # Limpa buffers
                icm_prev_states.clear()
                icm_states.clear()
                icm_actions.clear()
                icm_rewards.clear()
            # =====
            '''
            transition = {
                "state": state,
                "new": new,
                "option": option,
                "last_option": last_option,
                "prevac": prevac, #previous action
                "action": ac,
                "reward": rew,
                "her": 0,
                "next_state": obs_orig['observation'], #next_obs
                "achieved_goal": obs_orig["achieved_goal"], #next_obs
                "desired_goal": obs_orig["desired_goal"], #next_obs
                #"desired_gripper_position": state_org["desired_gripper_position"], #next_obs
                "phase": phase, #Estágio do ambiente (agente procura objeto ou objetivo final)
            }
        else:
            state = ob #Get state before action
            ob, rew, done, truncated, _ = env.step(ac,iters_so_far)
            transition = {
                "state": state,
                "action": ac,
                "reward": rew,
                "next_state": ob,
            }
            

        if np.any(np.isnan(ob)) or np.any(np.isinf(ob)):
            print("Reiniciando episódio por instabilidade")
            ob, obs_orig = env.reset(iters_so_far)
        
        new = done or truncated
        episode_transitions.append(transition)

        #print(ob)
        #print(f"\ndesired_goal: {obs_orig['desired_goal']}")
        #print(f"achieved_goal: {obs_orig['achieved_goal']}")


        #transition = {"state": state, "action": ac, "reward": rew,
        #          "next_state": ob['observation'], "achieved_goal": ob['achieved_goal'],
        #          "desired_goal": ob['desired_goal']}
        #buffer.append(transition)

        #print(buffer)

        rews[i] = rew
        realrews[i] = rew
        # print('rews:')
        # print(rews)
        # print('realrews:')
        # print(realrews)
        #if isinstance(ob, dict):
        #    ob = ob['observation']
        ## RL loop ##

        '''
        curr_opt_duration += 1
        '''

        candidate_option,active_options_t = pi.get_option(ob)

        term = pi.get_term([ob],[option])
        last_option=option
        #print(option)
        if term:
            '''
            opt_duration[option].append(curr_opt_duration)
            #print(f"opt_duration[{option}]", opt_duration[option])
            curr_opt_duration = 0.
            '''

            ep_states_term[option].append(ob)
            option = candidate_option


        ep_states[option].append(ob)
        cur_ep_ret += rew
        cur_ep_len += 1
        
        if new:

            # ========== Verifica se o ambiente foi resolvido com base na ultima posição registrada ==========
            #print('encerrou episodio...')
            #print(f"achieved_goal: {obs_orig['achieved_goal']}")
            #print(f"desired_goal: {obs_orig['desired_goal']}")
            dist_euclidiana = np.linalg.norm(obs_orig['desired_goal'] - obs_orig['achieved_goal'])
            #print(f"distancia euclidiana: {dist_euclidiana}")
            if (dist_euclidiana <= dist_solved): #<0.05 para recompensa 0, <0.07 é considerado resolvido e <0.2 para slide
                #print('Solucionado')
                cont_solved += 1
            cont_episodes += 1

            # ==========
            # Aplica HER nas transições do episódio e adiciona ao buffer de replay
            #### ==== desativado temporario -> augmented_transitions = apply_her(episode_transitions, env)
            #if (random.random() < 1/10):
            #aux_her_state, aux_her_new, aux_her_option, aux_her_last_option, aux_her_prevac, aux_her_action, aux_her_reward, aux_her_next_state, aux_her_achieved_goal, aux_her_desired_goal = apply_her_v2(episode_transitions, env)
            #her_desired_goal.append(aux_her_desired_goal)
            #print(len(her_desired_goal))
            #print(her_desired_goal)
            #print(aux_her_state)
            #### ==== desativado temporario -> replay_buffer.extend(augmented_transitions)
            #print(replay_buffer)

            # Extrair os valores de reward
            #her_reward = [t["reward"] for t in augmented_transitions]
            # Calcular a média
            #average_her_reward = np.mean(her_reward)
            #print(f'\nher_reward: {average_her_reward}')
            
            augmented_transitions = apply_her_v3(episode_transitions, env, icm, kher, hermvobj, objcoeff, itschrew, itsheroff)

            # ===== Incrementa todos os buffers para tratamento subsequente
            if (len(augmented_transitions)>0): #Retornou algum elemento, seja por haver transição ou pela execução aleatória interna da função apply_her
                # Atualiza her_state
                if her_state.size == 0:
                    her_state = np.array([transition["state"] for transition in augmented_transitions])
                else:
                    her_state = np.vstack([her_state] + [transition["state"] for transition in augmented_transitions])
                # Atualiza her_new (obs. como um vetor 1D)
                if her_new.size == 0:
                    her_new = np.array([transition["new"] for transition in augmented_transitions]).flatten().astype(int)
                else:
                    her_new = np.hstack([her_new, [transition["new"] for transition in augmented_transitions]]).astype(int)
                # Atualiza her_option
                if her_option.size == 0:
                    her_option = np.array([transition["option"] for transition in augmented_transitions]).flatten().astype(int)
                else:
                    her_option = np.hstack([her_option] + [transition["option"] for transition in augmented_transitions]).astype(int)
                # Atualiza her_last_option
                if her_last_option.size == 0:
                    her_last_option = np.array([transition["last_option"] for transition in augmented_transitions]).flatten().astype(int)
                else:
                    her_last_option = np.hstack([her_last_option] + [transition["last_option"] for transition in augmented_transitions]).astype(int)
                # Atualiza her_prevac
                if her_prevac.size == 0:
                    her_prevac = np.array([transition["prevac"] for transition in augmented_transitions])
                else:
                    her_prevac = np.vstack([her_prevac] + [transition["prevac"] for transition in augmented_transitions])
                # Atualiza her_action
                if her_action.size == 0:
                    her_action = np.array([transition["action"] for transition in augmented_transitions])
                else:
                    her_action = np.vstack([her_action] + [transition["action"] for transition in augmented_transitions])
                # Atualiza her_reward
                if her_reward.size == 0:
                    her_reward = np.array([transition["reward"] for transition in augmented_transitions]).flatten()
                else:
                    her_reward = np.hstack([her_reward] + [transition["reward"] for transition in augmented_transitions])
                # # Atualiza her_next_state
                # if her_next_state.size == 0:
                #     her_next_state = np.array([transition["next_state"] for transition in augmented_transitions])
                # else:
                #     her_next_state = np.vstack([her_next_state] + [transition["next_state"] for transition in augmented_transitions])
                # # Atualiza her_achieved_goal
                # if her_achieved_goal.size == 0:
                #     her_achieved_goal = np.array([transition["achieved_goal"] for transition in augmented_transitions])
                # else:
                #     her_achieved_goal = np.vstack([her_achieved_goal] + [transition["achieved_goal"] for transition in augmented_transitions])
                # # Atualiza her_desired_goal
                # if her_desired_goal.size == 0:
                #     her_desired_goal = np.array([transition["desired_goal"] for transition in augmented_transitions])
                # else:
                #     her_desired_goal = np.vstack([her_desired_goal] + [transition["desired_goal"] for transition in augmented_transitions])
                # # Atualiza her_reward
                # if her_applied.size == 0:
                #     her_applied = np.array([transition["her"] for transition in augmented_transitions]).flatten()
                # else:
                #     her_applied = np.hstack([her_applied] + [transition["her"] for transition in augmented_transitions])

                # Atualiza her_desired_goal
                # if her_observation.size == 0:
                #     her_observation = np.array([transition["observation"] for transition in augmented_transitions])
                # else:
                #     her_observation = np.vstack([her_observation] + [transition["observation"] for transition in augmented_transitions])

                # Exibe o estado atual de todos os arrays
                # print(f"her_state: {her_state.shape}")
                # print(f"her_new: {her_new.shape}")
                # print(f"her_option: {her_option.shape}")
                # print(f"her_last_option: {her_last_option.shape}")
                # print(f"her_prevac: {her_prevac.shape}")
                # print(f"her_action: {her_action.shape}")
                # print(f"her_reward: {her_reward.shape}")
                # print(f"her_next_state: {her_next_state.shape}")
                # print(f"her_achieved_goal: {her_achieved_goal.shape}")
                # print(f"her_desired_goal: {her_desired_goal.shape}")
                # print(f"her_observation: {her_observation.shape}")
                
                #print(f"Total de elementos em her_desired_goal ({type(her_desired_goal)}): {len(her_desired_goal)}")
                #print(her_desired_goal)  # Mostra o array cumulativo
                

            episode_transitions = []  # Reseta transições para o próximo episódio
            # ==========
            #print(f'org_rewards: {cur_ep_ret}')

            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0

            ep_num +=1
            ob, obs_orig = env.reset(iters_so_far)
            option,active_options_t = pi.get_option(ob)
            last_option=option
            ep_states[option].append(ob)
        t += 1


def add_vtarg_and_adv(seg, gamma, lam, num_options):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    T = len(seg["rew"])
    arrival_options = np.append(seg["last_options"],seg["last_option"])
    opts = np.append(seg["opts"],seg["opt"])
    rew = seg["rew"]

    op_vpred = np.append(seg["op_vpred"], seg["nextop_vpred"])
    term_p = np.vstack((np.array(seg["term_p"]),np.array(seg["next_term_p"])))
    q_sw = np.vstack((seg["vpred"],seg["nextvpred"]))
    all_u_sw = (1-term_p) * q_sw + term_p * np.tile(op_vpred[:,None],num_options)
    u_sw = all_u_sw[range(len(all_u_sw)),arrival_options]
    
    
    seg["op_adv"] = gaelam = np.empty(T, 'float32')
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * u_sw[t+1] * nonterminal - u_sw[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam


    seg["adv"] = gaelam = np.empty(T, 'float32')
    vpred= q_sw[range(len(opts)),opts]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

    seg["tdlamret"] = seg["adv"] + vpred[:-1]

    seg["term_adv"] = seg["vpred"] - np.tile(seg["op_vpred"][:,None],num_options)




def learn(env, policy_func, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        num_options=1,
        app='',
        saves=False,
        wsaves=False,
        epoch=0,
        seed=1,
        w_intfc=True,switch=False,intlr=1e-4,piolr=1e-4,multi=False,eta=0.1,
        render=False,
        is_goal_env=False,
        kher=1,
        hermvobj=1,
        randomact=0,
        objcoeff=1,
        itschrew=150,
        itsheroff=300
        ):


    optim_batchsize_ideal = optim_batchsize 
    np.random.seed(seed)
    tf.set_random_seed(seed)
    


    ### Book-keeping
    if hasattr(env,'NAME'):
        gamename = env.NAME.lower() #change this for plots
    else:
        gamename = env.spec.id[:-3].lower()
    gamename += 'seed' + str(seed)

    #Saving the wigths
    dirname = 'savedmodels/{}_{}opts_saves/'.format(gamename,num_options)
    #dirname = 'savedmodels_2/{}_{}opts_saves/'.format(gamename,num_options)

    if wsaves:
        first=True
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            first = False
    ###


    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon



    prob_cur_opt = tf.placeholder(dtype=tf.float32, shape=[None]) # Probability of current option
    is_ratio = tf.placeholder(dtype=tf.float32, shape=[None]) # IS ratio for correcting off-policyness


    ob = U.get_placeholder_cached(name="ob")
    option = U.get_placeholder_cached(name="option")
    term_adv = U.get_placeholder(name='term_adv', dtype=tf.float32, shape=[None])
    op_adv = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    betas = tf.placeholder(dtype=tf.float32, shape=[None]) # Probability of termination (Used to weight meta-updates)
    oldvpred = tf.placeholder(tf.float32, [None])
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac) + is_ratio)
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg 
    pol_surr = - U.mean(tf.minimum(surr1, surr2)  * prob_cur_opt )  # PPO's pessimistic surrogate (L^CLIP)


    vf_loss = U.mean(tf.square(pi.vpred - ret) * tf.exp(is_ratio) * prob_cur_opt)

    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    # Loss for termination function
    option_hot = tf.one_hot(option,depth=num_options)
    term_loss= U.mean(( tf.reduce_sum(pi.tpred * option_hot, axis=1) * term_adv) )

    # Loss for interest function
    pi_w = tf.placeholder(dtype=tf.float32, shape=[None,num_options])
    pi_I = (pi.intfc ) * pi_w / tf.expand_dims(tf.reduce_sum((pi.intfc ) * pi_w,axis=1),1)
    pi_I = tf.clip_by_value(pi_I,1e-6,1-1e-6)
    int_loss = - tf.reduce_sum(betas *tf.reduce_sum(pi_I * option_hot,axis=1)    * op_adv)

    # Loss for policy over options
    intfc = tf.placeholder(dtype=tf.float32, shape=[None,num_options])
    pi_I = (intfc ) * pi.op_pi / tf.expand_dims(tf.reduce_sum( (intfc ) * pi.op_pi,axis=1),1)
    pi_I = tf.clip_by_value(pi_I,1e-6,1-1e-6)
    op_loss = - tf.reduce_sum(betas *tf.reduce_sum(pi_I * option_hot,axis=1)    * op_adv)
    log_pi = tf.log(tf.clip_by_value(pi.op_pi, 1e-20, 1.0))
    op_entropy = -tf.reduce_mean(pi.op_pi * log_pi, reduction_indices=1)
    op_loss -= 0.01*tf.reduce_sum(op_entropy)



    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult, option,  prob_cur_opt, is_ratio], losses + [U.flatgrad(total_loss, var_list)])
    termgrad = U.function([ob, option, term_adv], [U.flatgrad(term_loss, var_list)]) # Since we might use a different step size.
    opgrad = U.function([ob, option, betas, op_adv, intfc], [U.flatgrad(op_loss, var_list)]) # Since we might use a different step size.
    intgrad = U.function([ob, option, betas, op_adv, pi_w], [U.flatgrad(int_loss, var_list)]) # Since we might use a different step size.
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult, option], losses)


    U.initialize()
    adam.sync()


    saver = tf.train.Saver(max_to_keep=10000)
    #saver = tf.train.Saver(max_to_keep=0)

    ### More book-kepping
    # results=[]
    # if saves:
    #     directory_res = "res/opt{}/".format(num_options) #if not fewshot else "res_fewshot/opt{}/".format(num_options) 
    #     print(f'\ndirectory_res: {directory_res}')
    #     if not os.path.exists(directory_res):
    #         os.makedirs(directory_res)       
    #     if w_intfc: 
    #         results = open(directory_res + gamename +'intfc{}_lr_{}_intlr{}_piolr{}_eta{}_seed{}'.format(int(w_intfc),optim_stepsize,intlr,piolr,eta,seed) + '.csv','w')
    #     else:
    #         results = open(directory_res + gamename +'intfc{}_lr_{}_intlr{}_piolr{}_eta{}_seed{}'.format(int(w_intfc),optim_stepsize,intlr,piolr,eta,seed) + '.csv','w')
    #     out = 'epoch,avg_reward,num_opts_used'

    #     out+='\n'
    #     results.write(out)
    #     results.flush()

    if epoch:

        #dirname = 'savedmodels/moc/{}_{}opts_saves/'.format(gamename,num_options)
        dirname = 'savedmodels/{}_{}opts_saves/'.format(gamename,num_options)
        print("Loading weights from iteration: " + str(epoch) + " from " + dirname)

        filename = dirname + '{}_epoch_{}.ckpt'.format(gamename,epoch)
        saver.restore(U.get_session(),filename)
    ###  


    episodes_so_far = 0
    timesteps_so_far = 0
    global iters_so_far
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=10) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=10) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"








    ######################################################### Prepare for rollouts #########################################################
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    option_counter = OptionStepCounter(num_options)
    seg_gen = traj_segment_generator(pi,env,timesteps_per_batch,stochastic=True,num_options=num_options,saves=saves,rewbuffer=rewbuffer,epoch=epoch,seed=seed,w_intfc=w_intfc,switch=switch,gamma=gamma,eta=eta,option_counter=option_counter,render=render,is_goal_env=is_goal_env,kher=kher,hermvobj=hermvobj,randomact=randomact,objcoeff=objcoeff,itschrew=itschrew,itsheroff=itsheroff)

    datas = [0 for _ in range(num_options)]

    while True:
        #print('Iniciando learning...')
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)
        seg = seg_gen.__next__()

        add_vtarg_and_adv(seg, gamma, lam,num_options)

        # ===== Exibição da utilização das options =====
        '''
        opt_d = []
        for i in range(num_options):
            #print(i, ": ", seg['opt_dur'][i])
            dur = np.mean(seg['opt_dur'][i]) if len(seg['opt_dur'][i]) > 0 else 0.
            opt_d.append(dur)

        print("mean opt dur:", opt_d)  
        '''           
        #print("mean op probs:", np.mean(np.array(seg['op_probs']),axis=0))         
        #print("mean term p:", np.mean(np.array(seg['term_p']),axis=0))
        #print("mean vpreds:", np.mean(np.array(seg['vpred']),axis=0))
        # =====


        ob, ac, opts, atarg, tdlamret, op_atarg  = seg["ob"], seg["ac"], seg["opts"], seg["adv"], seg["tdlamret"], seg["op_adv"] 
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy
        assign_old_eq_new() # set old parameter values to new parameter values

        #Savind weigths
        if iters_so_far % 10 == 0 and wsaves:
            print("weights are saved...")
            filename = dirname + '{}_epoch_{}.ckpt'.format(gamename,iters_so_far)
            save_path = saver.save(U.get_session(),filename)
        

        min_batch=160 # Arbitrary
        #print('Atualizando options...')
        for opt in range(num_options):
                       

            if multi: ### multi-updates here ###
                inds = np.arange(len(ob))
                is_ratios =seg["action_ratios"] + seg["term_ratios"]
                is_ratios=is_ratios[:,opt]
                prob_curr_opt= seg["prob_curr_opt"][:,opt]
                d = Dataset(dict(ob=ob[inds], ac=ac[inds], atarg=atarg[inds], vtarg=tdlamret[inds],  prob_curr_opt=prob_curr_opt[inds], is_ratios=is_ratios[inds], oldvpred=seg["vpred"][inds,opt]), shuffle=not pi.recurrent)

                #logger.log("Optimizing...")
                # Here we do a bunch of optimization epochs over the data
                for _ in range(optim_epochs):
                    losses = [] # list of tuples, each of which gives the loss for a minibatch
                    for batch in d.iterate_once(optim_batchsize):
                        *newlosses, grads = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult, [opt], batch["prob_curr_opt"], batch["is_ratios"])
                        adam.update(grads, optim_stepsize * cur_lrmult) 
                        losses.append(newlosses)

            else:
                indices = np.where(opts==opt)[0]
                print("batch size:",indices.size)
                if not indices.size:
                    continue

                if datas[opt] != 0:

                    if (indices.size < min_batch and datas[opt].n > min_batch):
                        datas[opt] = Dataset(dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], vtarg=tdlamret[indices]), shuffle=not pi.recurrent)
                        continue

                    elif indices.size + datas[opt].n < min_batch:
                        oldmap = datas[opt].data_map

                        cat_ob = np.concatenate((oldmap['ob'],ob[indices]))
                        cat_ac = np.concatenate((oldmap['ac'],ac[indices]))
                        cat_atarg = np.concatenate((oldmap['atarg'],atarg[indices]))
                        cat_vtarg = np.concatenate((oldmap['vtarg'],tdlamret[indices]))
                        datas[opt] = Dataset(dict(ob=cat_ob, ac=cat_ac, atarg=cat_atarg, vtarg=cat_vtarg), shuffle=not pi.recurrent)
                        continue

                    elif (indices.size + datas[opt].n > min_batch and datas[opt].n < min_batch) or (indices.size > min_batch and datas[opt].n < min_batch):

                        oldmap = datas[opt].data_map
                        cat_ob = np.concatenate((oldmap['ob'],ob[indices]))
                        cat_ac = np.concatenate((oldmap['ac'],ac[indices]))
                        cat_atarg = np.concatenate((oldmap['atarg'],atarg[indices]))
                        cat_vtarg = np.concatenate((oldmap['vtarg'],tdlamret[indices]))
                        datas[opt] = d = Dataset(dict(ob=cat_ob, ac=cat_ac, atarg=cat_atarg, vtarg=cat_vtarg), shuffle=not pi.recurrent)

                    if (indices.size > min_batch and datas[opt].n > min_batch):
                        datas[opt] = d = Dataset(dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], vtarg=tdlamret[indices]), shuffle=not pi.recurrent)

                elif datas[opt] == 0:
                    datas[opt] = d = Dataset(dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], vtarg=tdlamret[indices]), shuffle=not pi.recurrent)



                optim_batchsize = optim_batchsize or ob.shape[0]

                #logger.log("Optimizing...")
                #========== HER
                # Here we do a bunch of optimization epochs over the data
                # for _ in range(optim_epochs):
                    
                #     # Sample a minibatch from the replay buffer
                #     #for batch in replay_buffer.sample(minibatch_size):
                #     # Extract the components needed for training
                #     batch = replay_buffer#.sample(minibatch_size)

                #     states = np.array([b["state"] for b in batch])
                #     actions = np.array([b["action"] for b in batch])
                #     rewards = np.array([b["reward"] for b in batch])
                #     next_states = np.array([b["next_state"] for b in batch])
                #     goals = np.array([b["desired_goal"] for b in batch])

                #     # Perform optimization using the sampled minibatch
                #     *newlosses, grads = lossandgrad(states, actions, rewards, next_states, goals)
                #     adam.update(grads, optim_stepsize * cur_lrmult)
                #========== original
                #Here we do a bunch of optimization epochs over the data
                for _ in range(optim_epochs):
                    for batch in d.iterate_once(optim_batchsize):
                        *newlosses, grads = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult, [opt], np.ones_like(batch["vtarg"]), np.zeros_like(batch["vtarg"]))
                        adam.update(grads, optim_stepsize * cur_lrmult)
                #==========


        termg = termgrad(seg["ob"], seg['last_options'], seg["term_adv"][range(len(seg["last_options"])),seg["last_options"]] )[0]
        adam.update(termg, piolr)

        if w_intfc:
            intgrads = intgrad(seg['ob'],seg['opts'], seg["last_betas"], op_atarg, seg["op_probs"])[0]
            adam.update(intgrads, intlr)

        opgrads = opgrad(seg['ob'],seg['opts'], seg["last_betas"], op_atarg, seg["intfc"])[0]
        adam.update(opgrads, intlr)     
        

        # Atualmente o cálculo de recompensa pode ser utilizado dessa forma, pois ep_lens e ep_rets estão considerando apenas os elementos originais
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs=[lrlocal]
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        #print(f'original len {len(rews)}')
        rewbuffer.extend(rews) # -> original, computa a recompensa de todas as trajetórias (aparentemente no método HER com concatenação, apenas os rews de trajetorias originais são utilizados no cálculo da recompensa...)
        
        # ========== auxiliar para exibir valores de recompensa apenas em trajetos sem aplicação do HER
        #np.set_printoptions(threshold=np.inf) #Com print, exibe valores completos dos np.array
        #aux_her_applied = seg["her"][::50][:-1]
        #indices_with_her_not_applied = np.where(aux_her_applied == 0)[0]
        #filtered_rewbuffer = np.array(rews)[indices_with_her_not_applied]
        #print("Média de rewbuffer:", np.mean(rews))
        #print("Média de rewbuffer sem HER aplicado:", np.mean(filtered_rewbuffer))
        #print(f'filtered len {len(filtered_rewbuffer)}')
        #rewbuffer.extend(filtered_rewbuffer) # -> modificado para não contabilizar a recompensa recalculada pelo HER
        # ==========
        
        try:
            percent_solved = seg["cont_solved"]/seg["cont_episodes"]
        except:
            percent_solved = 0

        # logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpSolved", percent_solved)
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        # logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        # logger.record_tabular("EpisodesSoFar", episodes_so_far)
        # logger.record_tabular("TimestepsSoFar", timesteps_so_far) #numero total de timesteps até o momento
        # logger.record_tabular("TimeElapsed", time.time() - tstart)

        for option in range(num_options): # Exibir os contadores de timesteps por option
            count = option_counter.getCount(option) # Get the count for each option
            total_steps = sum(option_counter.option_timestep_counts.values())  # Total de passos acumulados
            percent = (count / total_steps) * 100 if total_steps > 0 else 0  # Calcula o percentual
            #print(f"Option {option}: {count} steps / {percent:.1f}%")            
            option_str = f"{count} stp, {percent:.2f}%"
            logger.record_tabular("Option" + str(option), option_str)

        logger.dump_tabular()
        option_counter.resetCount()

        ### Book keeping
        # if saves:
        #     out = "{},{},{}"
        #     out+="\n"
        #     #avg_num_options=np.mean(np.sum(seg["activated_options"],axis=1))
        #     info = [iters_so_far, np.mean(rewbuffer), '']#,avg_num_options]
        #     results.write(out.format(*info))
        #     results.flush()
        ###




def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
