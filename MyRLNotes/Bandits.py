"""
Code used partially adopted from RL web-notes: https://www.kaggle.com/code/parsasam/reinforcement-learning-notes-multi-armed-bandits
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from tqdm import tqdm
import pandas as pd
import scipy


# Bandit
def bandit(action, problem, arr):
    return np.random.normal(arr[problem, action], 1) #Draw random samples from a normal (Gaussian) distribution.


def load_datafile(filename):
    pass

def save_data(filename, arms, rewards):
    '''
    Required format:
    Trial # | Arm chosen | Reward
    '''
    df = pd.DataFrame({
        "Trial " : range(1, len(rewards)+1 ),
        "Chosen Arm " : arms,
        "Reward Recieved ": rewards
    })
    df.to_csv(filename, index=False)

# Action selection funct
def simple_max(Q, N, t):
    return np.random.choice(np.flatnonzero(Q == Q.max()))   # Break ties at random

def UCB(Q, N, t, confidence=2):
    if N.min() == 0:
        return np.random.choice(np.flatnonzero(N == N.min()))
    M = Q + confidence*np.sqrt(np.divide(np.log2(t), N))
    return softmax(M, temp=1)

def SMUCB(Q, N, t, uncertParam, temp):
    # print(N.size)
    last_arm_reward = np.zeros(shape=N.size)
    for i in range(N.size):
        x = np.array(np.where(N[t] == i))
        if x.size == 0:
            last_arm_reward[i] = 0
        else:
            last_arm_reward = np.max(x)
    uncert = (uncertParam * (t - last_arm_reward)) / 100

    num = np.exp(np.multiply(Q + uncert, temp ))
    denom = sum(np.exp(np.multiply(Q + uncert, temp)))

    return np.argmax(np.cumsum(num / denom) > np.random.rand())
# def softmax_UCB(Q, N, t, temp)

def softmax(qVal, temp):
    num = np.exp(np.multiply(qVal, temp))
    denom = sum(np.exp(np.multiply(qVal, temp)))
    return np.argmax(np.cumsum(num / denom) > np.random.rand()) # return max arg where mac value is less than a randdom value chosen from a uniform distrubution

# TODO: Fix LL function
def loglik_vect(x, *args):
    '''
    From: https://www.pymc.io/projects/examples/en/latest/case_studies/reinforcement_learning.html#estimating-the-learning-parameters-via-maximum-likelihood
    '''
    alpha, beta = x
    actions, rewards = args

    Qs = np.ones((1000, 10), dtype="float64")
    Qs[0] = 0.5

    for time_t, (a, r) in enumerate(zip(actions[:-1], rewards[:-1])):
        Qs[time_t + 1, a] = Qs[time_t, a] + alpha * (r - Qs[time_t, a])
        Qs[time_t + 1, 1 - a] = Qs[time_t, 1 - a]

    # apply softmax
    return softmax(Qs)

def paramCreate(num):
    """
    Generates random params for each of the models
    (From TF notebook)
    """

    paraCreate = np.zeros(shape=[num, 7])

    # eGreedy 
    paraCreate[:,0] = np.random.beta(1.1, 1.1, size=num)    #EG - LR
    paraCreate[:,1] = np.random.beta(1.1, 1.1, size=num)    #EG - Eps

    # Softmax
    paraCreate[:,2] = np.random.beta(1.1, 1.1, size=num)    #Rand - LR
    paraCreate[:,3] = np.random.uniform(1, 30, size=num)    #SM - Temp

    # Softmax Explore
    paraCreate[:,4] = np.random.beta(1.1, 1.1, size=num)    #SMUCB - LR
    paraCreate[:,5] = np.random.uniform(1, 30, size=num)    #SMUCB - Temp
    paraCreate[:,6] = np.random.gamma(1.1, 1.1, size=num)   #SMUCB - Uncertainty



class Testbed():
    '''
    Creates a testing environment with k arms and num_sim number of trials
    ie. k=10, num_sim=1000 will generate a testbed with 10 arms, each with 1000
    different reward distrubutions to sample from depending on which trial is 
    in progress
    '''
    def __init__(self, k=10, num_sim=1000) -> None:
        self.k = k
        self.num_problems = num_sim

        self.arms = [0] * k
        self.q_star = np.random.normal(0, 1, (num_sim, k)) #Random sample with mean=0, stddev=1

        for i in range(k):
            self.arms[i] = np.random.normal(self.q_star[0,i], 1, num_sim)
        
        self.means = [np.mean(d) for d in self.arms]
    
    def show_testbed(self):
        plt.figure(figsize=(12,8))
        plt.ylabel("Rewards Distribution")
        plt.xlabel("Action")
        plt.xticks(range(1,11))
        plt.yticks(np.arange(-5,5,0.5))
        plt.violinplot(self.arms, positions=(range(1,11)), showmedians=True)
        for i in range(0,10):
            plt.scatter(i+1, self.means[i])
            plt.text(i+1.15, self.means[i], f"q*({i+1})")
        plt.show()
    
    def show_mean(self):
        print(f"Testbed Means: {self.means}")


class Bandit():
    """
    Parent Bandit class containing attrutes and method declarations used by
    each bandit type.
    """
    def __init__(self, model_params, trial_params, reward_values, start_val) -> None:
        # self.model_params = model_params
        # self.trial_params = trial_params
        # self.reward_values = reward_values
        # self.start_val = start_val

        #==========================================

        self.total_rewards = np.zeros(trial_params[1])      # rewards array
        self.total_actions = np.zeros(trial_params[1])      # Choices array

        self.avg_rewards = None
        self.avg_actions = None

        self.final_N = None
    
    def simulate(self, num_sim=1000, save=False) -> None:
        pass

    def show_results(self):
        pass
    def show_actions(self):
        pass
            
class EG_Bandit(Bandit):
    def __init__(self, model_params, trial_params, reward_values, start_val) -> None:
        super().__init__(model_params, trial_params, reward_values, start_val)
        self.model_type = "eGreedy"

        self.k = trial_params[0]
        self.steps = trial_params[1]

        self.alpha = model_params[0]           # if no alpha is passed of alpha=0, alg uses 1/n
        self.epsilon = model_params[1]

        self.q_star = reward_values
        self.initial_Q = start_val

        self.argmax_func = simple_max

    def simulate(self, num_sim=1000, save=False) -> None:
        for i in tqdm(range(num_sim)):
            Q = np.ones(self.k) * self.initial_Q    # rewards array
            N = np.zeros(self.k)                    # number of times each arm is chosen (choices)
            
            best_action = np.argmax(self.q_star[i])     # select best action for trial i
            chosen_arm = np.zeros(self.steps, dtype=int)

            # perform trial steps
            for time_t in range(self.steps):
                if np.random.rand() < self.epsilon:
                    a = np.random.randint(self.k)       #Explore
                else:
                    a = self.argmax_func(Q, N, time_t)    #epxloit using chosen argmax_func ie. take best choice

                # get reward
                reward = bandit(a, i, self.q_star)      

                # Update chosen arm
                N[a] +=  1   

                # Get prediction error
                prediction_error = reward - Q[a]

                # update reward values
                if self.alpha > 0:
                    Q[a] = Q[a] + (prediction_error) * self.alpha    #contsant step-size
                else:
                    Q[a] = Q[a] + (prediction_error) / N[a]          #incremental step-size: Sample average case | alpha_n(a) = 1/n
                
                # Save reward
                self.total_rewards[time_t] += reward
                chosen_arm[time_t] = a + 1          # record proper arm chosen

                # record if action is best action
                if a == best_action:                
                    self.total_actions[time_t] += 1

        self.avg_rewards = np.divide(self.total_rewards, num_sim)
        self.avg_actions = np.divide(self.total_actions, num_sim)

        self.final_N = N        # Final Arm tally on last trial

        if save == True:
            save_data("rewards.csv", chosen_arm, self.avg_rewards)
    
    def show_results(self, *args):
        print("="*30)
        print("Showing the following test: ",
              f"Model name: {self.model_type}",
              f"Action selection: {self.argmax_func.__name__}",
              f"Number of arms: {self.k}",
              f"Epsilon: {self.epsilon}",
              f"Step-size/Learning rate (alpha): {self.alpha if self.alpha>0 else '1/n'}",
              f"Steps: {self.steps}",  
              f"Initial Q: {self.initial_Q}",
              f"Average Reward: {np.average(self.avg_rewards)}", 
              sep='\n\t')
        print("="*30)
        plt.figure(figsize=(12,6))
        plt.title(f"Average Reward Over Time {self.initial_Q}")
        plt.plot(self.avg_rewards, 'r', label=f'epsilon = {self.epsilon}')
        plt.xlabel("Steps")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.show()

        # # Flag: ALL- prints final arm tally
        # if "a" in args:
        #     print()
    def show_actions(self):
        plt.figure(figsize=(12,6))
        plt.title(f"Action Selection ({self.model_type})")
        plt.plot(self.avg_actions, "b", label=f"epsilon = {self.epsilon}")
        plt.xlabel("Steps")
        plt.ylabel("Action Selection")
        plt.legend()
        plt.show()
    
    
class SM_Bandit(Bandit):
    def __init__(self, model_params, trial_params, reward_values, start_val) -> None:
        super().__init__(model_params, trial_params, reward_values, start_val)
        self.model_type = "Softmax"

        self.k = trial_params[0]
        self.steps = trial_params[1]

        self.alpha = model_params[0]
        self.temp = model_params[1]

        self.q_star = reward_values
        self.initial_Q = start_val


    def simulate(self, num_sim=1000, save=False) -> None:
        for i in tqdm(range(num_sim)):
            Q = np.ones(self.k) * self.initial_Q
            N = np.zeros(self.k)

            best_action = np.argmax(self.q_star[i])
            chosen_arm = np.zeros(self.steps, dtype=int)

            for time_t in range(self.steps):

                # Choose arm
                a = softmax(Q, (self.temp))
                # print(a)

                # Get reward
                reward = bandit(a, i, self.q_star)
                N[a]+=1

                # Get prediction error
                prediction_error = reward - Q[a]

                # Update Q
                Q[a] = Q[a] + prediction_error * self.alpha
                
                self.total_rewards[time_t] += reward
                chosen_arm[time_t] = a + 1

                if a == best_action:
                    self.total_actions[time_t] += 1

        self.avg_rewards = np.divide(self.total_rewards, num_sim)
        self.avg_actions = np.divide(self.total_actions, num_sim)

        self.final_N = N        # Final Arm tally on last trial

        if save == True:
            save_data("rewards.csv", chosen_arm, self.avg_rewards)
    
    def show_results(self):
        print("="*30)
        print("Showing the following test: ",
              f"Model name: {self.model_type}",
              f"Action selection: {self.model_type} | temperature = {self.temp}",
              f"Number of arms: {self.k}",
              f"Learning rate: {self.alpha}",
              f"Steps: {self.steps}",
              f"Initial Q: {self.initial_Q}",
              f"Average reward: {np.average(self.avg_rewards)}",
              sep='\n\t')
        print("="*30)
        plt.figure(figsize=(12,6))
        plt.title(f"Average Reward Over Time ({self.model_type})")
        plt.plot(self.avg_rewards, 'r', label=f'temp = {self.temp}')
        plt.xlabel("Steps")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.show()
    
    def show_actions(self):
        plt.figure(figsize=(12,6))
        plt.title(f"Action Selection ({self.model_type})")
        plt.plot(self.avg_actions, "b", label=f"temp = {self.temp}")
        plt.xlabel("Steps")
        plt.ylabel("Action Selection")
        plt.legend()
        plt.show()
    
class SMUCB_Bandit(Bandit):
    """
    Incorporates a hybrid strategy: combines softmax probablistic random exploration
    with directed exploration (operationalized as uncertainty reduction)
    """
    def __init__(self, model_params, trial_params, reward_values, start_val) -> None:
        super().__init__(model_params, trial_params, reward_values, start_val)
        self.model_type = "Softmax-UCB"

        self.k = trial_params[0]
        self.steps = trial_params[1]

        self.alpha = model_params[0]
        self.temp = model_params[1]
        self.uncertainty_param = model_params[2]

        self.q_star = reward_values
        self.initial_Q = start_val
    
    def simulate(self, num_sim=1000, save=False) -> None:
        for i in tqdm(range(num_sim)):
            Q = np.ones(self.k) * self.initial_Q        #rewards per trial
            N = np.zeros(self.k)                        #actions/choices per trial

            best_action = np.argmax(self.q_star[i])
            chosen_arm = np.zeros(self.steps, dtype=int)

            for time_t in range(self.steps):

                # Calculate uncertainty
                a = SMUCB(Q, N, time_t, uncertParam=self.uncertainty_param, temp=self.temp)
                print(a)
                # Compute Softmax Values
        

                reward = bandit(a, i, self.q_star)
                N[a]+=1
                
                # Get prediction error
                prediction_error = reward - Q[a]

                # Update Q
                Q[a] = Q[a] + prediction_error * self.alpha

                self.total_rewards[time_t] += reward
                chosen_arm[time_t] = a + 1

                if a == best_action:
                    self.total_actions[time_t] += 1

        self.avg_rewards = np.divide(self.total_rewards, num_sim)
        self.avg_actions = np.divide(self.total_actions, num_sim)

        self.final_N = N        # Final Arm tally on last trial

        if save == True:
            save_data("rewards.csv", chosen_arm, self.avg_rewards)


    def show_results(self):
        print("="*30)
        print("Showing the following test: ",
            f"Model name: {self.model_type}",
            f"Action selection: {self.model_type} | temp = {self.temp}",
            f"Number of arms: {self.k}",
            f"Learning rate: {self.alpha}",
            f"Steps: {self.steps}",
            f"Initial Q: {self.initial_Q}",
            f"Average reward: {np.average(self.avg_rewards)}",
            sep="\n\t")
        print("="*30)
        plt.figure(figsize=(12,6))
        plt.title(f"Average Reward Over Time ({self.model_type})")
        plt.plot(self.avg_rewards, 'r', label=f'temp = {self.temp}')
        plt.xlabel("Steps")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.show()
    

    
def create_bandit_task(model_type, model_params, trial_params, reward_values, start_val):
    
    if model_type == "EG":
        return EG_Bandit(model_params, trial_params, reward_values, start_val)
    elif model_type == "SM":
        return SM_Bandit(model_params, trial_params, reward_values, start_val)
    elif model_type == "SMUCB":
        return SMUCB_Bandit(model_params, trial_params, reward_values, start_val)
    else:
        raise ValueError("Model type not accepted")