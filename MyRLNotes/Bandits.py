"""
=========================================
Title: Bandit Library
Author: Bryan Kostelyk, Thomas Ferguson
Licence: TBD
=========================================

Description:
    Python library of customizable bandit Class, designed to simulate choice-reward tasks
    with emphasis on the explore exploit-dilemna. 
    
    Current bandit variants:
        -> e-Greedy
        -> Softmax
        -> SoftmaxUCB


Some functions partially adapted from RL web-notes: https://www.kaggle.com/code/parsasam/reinforcement-learning-notes-multi-armed-bandits
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

def SMUCB(rewards, choices, trial, uncertParam, temp):
    print(choices.size)
    last_arm_reward = np.zeros(shape=10)
    for i in range(10):
        x = np.array(np.where(choices[0:trial] == i))
        if x.size == 0:
            last_arm_reward[i] = 0
        else:
            last_arm_reward = np.max(x)
    uncert = (uncertParam * (trial - last_arm_reward)) / 100

    num = np.exp(np.multiply(rewards + uncert, temp ))
    denom = sum(np.exp(np.multiply(rewards + uncert, temp)))

    return np.argmax(np.cumsum(num / denom) > np.random.rand())
# def softmax_UCB(Q, N, t, temp)

def softmax(qVal, temp):
    num = np.exp(np.multiply(qVal, temp))
    denom = sum(np.exp(np.multiply(qVal, temp)))
    return np.argmax(np.cumsum(num / denom) > np.random.rand()) # return max arg where max value is less than a randdom value chosen from a uniform distrubution

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
    def __init__(self, k=10, num_problems=1000) -> None:
        self.k = k
        self.num_problems = num_problems

        self.arms = [0] * k
        self.q_star = np.random.normal(0, 1, (num_problems, k)) #Random sample with mean=0, stddev=1

        for i in range(k):
            self.arms[i] = np.random.normal(self.q_star[0,i], 1, num_problems) # NOTE: Displays what the reward distribution could look like
        
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
    
    def show_test_walk(self):
        '''
        Shows an example of the arm reward for a single given simulation
        '''

        plt.figure(figsize=(12,8))
        plt.ylabel("Rewards")
        plt.xlabel("Steps")
        for i in range(len(self.arms)):
            plt.plot(self.arms[i])
        plt.show()


class Bandit():
    """
    Parent Bandit class containing attrutes and method declarations used by
    each bandit type.
    """
    def __init__(self, model_params, trial_params, reward_values, start_val) -> None:
        self.model_params=model_params

        # NOTE: Total Actions should be total correct actions as it tracks the correct arm choices of all sims
        self.total_rewards = np.zeros(trial_params[1])      # rewards array
        self.total_actions = np.zeros(trial_params[1])      # Choices array

        self.selection_matrix = None                        # Contains all choice data for each step in each agent/problem
        self.reward_matrix = None                           # Contains all reward data for each step in each agent/problem
        self.predictionErr_matrix = None                    # Contains prediction error for each step in each agent/problem

        self.avg_rewards = None
        self.avg_actions = None

        self.final_N = None
    
    def simulate(self, num_problems=1000, save=False) -> None:
        pass

    def simulate_LL(self, num_problems=1000, save=False) -> None:
        pass

    def show_results(self):
        pass
    def show_actions(self):
        pass
            
class EG_Bandit(Bandit):
    """
    eGreedy Bandit class that inherits all common attributes among the bandit models
    Key Parameters: Epsilon, Alpha (learning rate)
    """
    def __init__(self, model_params, trial_params, reward_values, start_val) -> None:
        super().__init__(model_params, trial_params, reward_values, start_val)
        self.model_type = "eGreedy"

        self.k = trial_params[0]
        self.steps = trial_params[1]

        self.alpha = model_params[0]           # if no alpha is passed of alpha=0, alg uses 1/n by default
        self.epsilon = model_params[1]

        self.q_star = reward_values
        self.initial_Q = start_val

        self.argmax_func = simple_max

    def simulate(self, num_problems=1000, save=False) -> None:
    

        self.selection_matrix = np.empty(num_problems, dtype=np.ndarray)
        self.reward_matrix = np.empty(num_problems, dtype=np.ndarray)
        self.predictionErr_matrix = np.empty(num_problems, dtype=np.ndarray)

        for i in tqdm(range(num_problems)):
            # Initialize Q Values
            Q = np.ones(self.k) * self.initial_Q    # Qvalue array

            # Initialize arm choice array
            N = np.zeros(self.k)                    # number of times each arm is chosen (choices)
            
            best_action = np.argmax(self.q_star[i])     # select best action for trial i

            rewards_arr = np.zeros(self.steps)
            chosen_arm_arr = np.zeros(self.steps, dtype=int)
            prediction_error_arr = np.zeros(self.steps)

            # perform trial steps
            for time_t in range(self.steps):
                if np.random.rand() < self.epsilon:
                    a = np.random.randint(self.k)         #Explore
                else:
                    a = self.argmax_func(Q, N, time_t)    #epxloit using chosen argmax_func ie. take best choice

                # get reward
                reward = bandit(a, i, self.q_star)      

                # Update chosen arm
                N[a] +=  1   

                # Get prediction error
                prediction_error = reward - Q[a]
                prediction_error_arr[time_t] = prediction_error

                # update Q values
                if self.alpha > 0:
                    Q[a] = Q[a] + (prediction_error) * self.alpha    #contsant step-size
                else:
                    Q[a] = Q[a] + (prediction_error) / N[a]          #incremental step-size: Sample average case | alpha_n(a) = 1/n
                
                # Save reward
                self.total_rewards[time_t] += reward
                rewards_arr[time_t] = reward
                chosen_arm_arr[time_t] = a                               # record proper arm chosen

                # record if action is best action
                if a == best_action:                
                    self.total_actions[time_t] += 1

            # Update Model Matrices
            self.selection_matrix[i] = chosen_arm_arr
            self.reward_matrix[i] = rewards_arr
            self.predictionErr_matrix[i] = prediction_error_arr

        self.avg_rewards = np.divide(self.total_rewards, num_problems)
        self.avg_actions = np.divide(self.total_actions, num_problems)

        self.final_N = N        # Final Arm tally on last trial

        if save == True:
            save_data("rewards.csv", chosen_arm_arr, self.avg_rewards)
    
    def simulate_LL(self, num_problems=1000, save=False) -> None:
        self.total_LL_array = np.zeros(num_problems)        # stores the LL sum of each problem

        for i in tqdm(range(num_problems)):
            LL_array = np.zeros(shape=[self.steps])
            qValue = np.zeros(self.k) * self.initial_Q

            for time_t in range(self.steps):

                # Find actual selection
                selection = int(self.selection_matrix[i][time_t])     # TODO: FIX! selection must be chosen arm at time_t (between 0 and 9)

                if selection == -1:
                    LL_array[time_t] = 1                    # NOTE: ????

                else:
                    # Compute eGreedy values

                    greedy_result = (self.epsilon/(len(qValue)-1)) * np.ones(shape=len(qValue))

                    # Convert Q value array to list
                    qValueList = qValue.tolist()

                    # Find max choice
                    maxLoc = qValueList.index(max(qValueList))

                    # Find greedy arm
                    greedy_result[maxLoc] = 1-self.epsilon

                    # compute reward
                    reward = self.total_rewards[time_t]

                    # compute prediction error
                    predError = reward - qValue[selection]

                    # Update reward - Non-stationary
                    qValue[selection] = qValue[selection] + self.alpha * predError

                    # Compute Likelihood
                    LL_array[time_t] = greedy_result[selection]

                    if LL_array[time_t] <= 0:
                        LL_array[time_t] = 1e+300

            # Deal with Nans
            LL_array[np.isnan(LL_array)] = 1e+300

            # update likelihood values
            LL_array_sum = -np.sum(np.log(LL_array))
            self.total_LL_array[i] = LL_array_sum


    
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

        self.k = trial_params[0]        # number of arms
        self.steps = trial_params[1]    # number of steps the agent takes

        self.alpha = model_params[0]    # learning param
        self.temp = model_params[1]     # temperature param

        self.q_star = reward_values     # reward values
        self.initial_Q = start_val      # initial q


    def simulate(self, num_problems=1000, save=False) -> None:

        self.selection_matrix = np.empty(num_problems, dtype=np.ndarray)
        self.reward_matrix = np.empty(num_problems, dtype=np.ndarray)
        self.predictionErr_matrix = np.empty(num_problems, dtype=np.ndarray)


        for i in tqdm(range(num_problems)):
            # Initialize Q-values
            Q = np.ones(self.k) * self.initial_Q
            
            # Initialie Arm
            N = np.zeros(self.k)


            best_action = np.argmax(self.q_star[i])

            rewards_arr = np.zeros(self.steps)
            chosen_arm_arr = np.zeros(self.steps, dtype=int)
            prediction_error_arr = np.zeros(self.steps)

            for time_t in range(self.steps):

                # Choose arm
                a = softmax(Q, (self.temp))
                # print(a)

                # Get reward
                reward = bandit(a, i, self.q_star)
                N[a]+=1

                # Get prediction error
                prediction_error = reward - Q[a]
                prediction_error_arr[time_t] = prediction_error

                # Update Q
                Q[a] = Q[a] + prediction_error * self.alpha
                
                self.total_rewards[time_t] += reward
                chosen_arm_arr[time_t] = a

                if a == best_action:
                    self.total_actions[time_t] += 1
            
            #Update Model Matrices
            self.selection_matrix[i] = chosen_arm_arr
            self.reward_matrix[i] = rewards_arr
            self.predictionErr_matrix[i] = prediction_error_arr

        self.avg_rewards = np.divide(self.total_rewards, num_problems)
        self.avg_actions = np.divide(self.total_actions, num_problems)

        self.final_N = N        # Final Arm tally on last trial

        if save == True:
            save_data("rewards.csv", chosen_arm_arr, self.avg_rewards)

    # TODO: Implement
    def simulate_LL(self, num_problems=1000, save=False) -> None:
        pass
    
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
    
# TODO: Optimize
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
    
    def simulate(self, num_problems=1000, save=False) -> None:
        
        # Initialize Matrices
        self.selection_matrix = np.empty(num_problems, dtype=np.ndarray)
        self.reward_matrix = np.empty(num_problems, dtype=np.ndarray)
        self.predictionErr_matrix = np.empty(num_problems, dtype=np.ndarray)


        for i in tqdm(range(num_problems)):
            Q = np.ones(self.k) * self.initial_Q        #Initialize Q values
            N = np.zeros(self.k)                        #actions/choices per trial

            best_action = np.argmax(self.q_star[i])

            rewards_arr = np.zeros(self.steps)
            chosen_arm_arr = np.zeros(self.steps, dtype=int)
            prediction_error_arr = np.zeros(self.steps)

            # needed for uncertainty calculation
            n = 1

            for time_t in range(self.steps):
                # print(chosen_arm_arr[i])

                # Calculate uncertainty
                # a = SMUCB(Q, chosen_arm_arr, n, uncertParam=self.uncertainty_param, temp=self.temp)     #NOTE: Fix
                lastCount = np.zeros(shape=self.k)
                for cTc in range(self.k):
                    x = np.array(np.where(chosen_arm_arr[0:time_t] == cTc))
                    if x.size == 0: 
                        lastCount[cTc] = 0
                    else:
                        lastCount[cTc] = np.amax(x)
                uncert = (self.uncertainty_param * (n - lastCount)) / 100
                
                num = np.exp(np.multiply(Q + uncert, self.temp))

                denom = sum(np.exp(np.multiply(Q + uncert, self.temp)))

                softmaxResult = num/denom

                softmaxOptions = np.cumsum(softmaxResult) > np.random.rand()

                a = np.argmax(softmaxOptions)

                # print(a)
        
                # Get reward
                reward = bandit(a, i, self.q_star)
                 
                # Update chosen arm tally 
                N[a]+=1
                chosen_arm_arr[time_t] = a
                
                # Get prediction error
                prediction_error = reward - Q[a]
                prediction_error_arr[time_t] = prediction_error

                # Update Q
                Q[a] = Q[a] + prediction_error * self.alpha

                # Update n
                n += 1

                # Save reward
                self.total_rewards[time_t] += reward
                rewards_arr[time_t] = reward

                if a == best_action:
                    self.total_actions[time_t] += 1 

            # Update Model Matrices
            self.selection_matrix[i] = chosen_arm_arr
            self.reward_matrix[i] = rewards_arr
            self.predictionErr_matrix[i] = prediction_error_arr    

        self.avg_rewards = np.divide(self.total_rewards, num_problems)
        self.avg_actions = np.divide(self.total_actions, num_problems)

        self.final_N = N        # Final Arm tally on last trial

        if save == True:
            save_data("rewards.csv", chosen_arm_arr, self.avg_rewards)

    # TODO: Implement
    def simulate_LL(self, num_problems=1000, save=False) -> None:
        pass


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
    
    def show_actions(self):
        plt.figure(figsize=(12,6))
        plt.title(f"Action Selection ({self.model_type})")
        plt.plot(self.avg_actions, "b", label=f"temp = {self.temp}")
        plt.xlabel("Steps")
        plt.ylabel("Action Selection")
        plt.legend()
        plt.show()

    
def create_bandit_task(model_type, model_params, trial_params, reward_values, start_val) -> Bandit:
    
    if model_type == "EG":
        return EG_Bandit(model_params, trial_params, reward_values, start_val)
    elif model_type == "SM":
        return SM_Bandit(model_params, trial_params, reward_values, start_val)
    elif model_type == "SMUCB":
        return SMUCB_Bandit(model_params, trial_params, reward_values, start_val)
    else:
        raise ValueError("Model type not accepted")
    
def model_performance_summary(bandits: list[Bandit]):
    fig, axs = plt.subplots(3,3, figsize=(12,12), dpi=500)
    fig.suptitle("Simulated Performance", fontsize = 35)

    for b in range(len(bandits)):
        # Optimal arm choice
        axs[b,0].plot(bandits[b].avg_actions, color='green')
        axs[b,0].set_xlabel("Trials", fontsize=12)
        axs[b,0].set_ylabel("Optimal Arm (%)", fontsize=12)
        axs[b,0].set_ylim(0,1)
        # axs[b,0].set_xticks()

        # TODO: arm_switching

        # Observed Rewardds
        axs[b,1].set_title(bandits[b].model_type)
        axs[b,1].plot(bandits[b].avg_rewards)
        axs[b,1].set_xlabel("Trials", fontsize=12)
        axs[b,1].set_ylabel("Observed Rewards", fontsize=12)

        # Prediction Error
        predErr = (np.average(bandits[b].predictionErr_matrix) * 100)
        axs[b,2].plot(predErr, color="orange")
        axs[b,2].set_xlabel("Trials", fontsize=12)
        axs[b,2].set_ylabel("Prediction Error", fontsize=12)
        axs[b,2].set_ylim(-100, 1)

    plt.tight_layout()
    plt.show()
    

# def parameter_recovery(bandits: list[Bandit], environment: Testbed , fit_attempts=5) -> None:
#     reward_model_PR = np.zeros(shape=[len(bandits), num_problems, num_trial])
#     choice_model_PR = np.zeros(shape=[len(bandits), num_problems, num_trial])

#     for pCt in tqdm(range(num_problems)):

