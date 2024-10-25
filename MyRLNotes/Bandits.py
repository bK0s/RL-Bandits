"""
=========================================
Title: Bandit Library
Author: Bryan Kostelyk, Thomas Ferguson
License: TBD
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

from ast import List
from calendar import c
from re import L, T
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from tqdm import tqdm
import pandas as pd
from scipy.optimize import minimize


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
def simple_max(Q):
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


def softmax(qVal, temp):
    '''
    Function used to get softmax distrubution of choice probablity

    @Parameters: numpy.array of qvals for given step, temperature param
    @Returns: (Action/arm chosen, SM Probablitity of chosen arm)
    '''
    num = np.exp(np.multiply(qVal, temp))
    denom = sum(np.exp(np.multiply(qVal, temp)))
    choice = np.argmax(np.cumsum(num / denom) > np.random.rand())
    return choice, num/denom


class Testbed(object):
    '''
    Creates a testing environment with k arms and num_sim number of trials
    ie. k=10, num_sim=1000 will generate a testbed with 10 arms, each with 1000
    different reward distrubutions to sample from depending on which trial is 
    in progress
    '''
    def __init__(self, k=10, num_problems=1000, stationary=True ) -> None:
        self.k = k
        self.num_problems = num_problems

        self.stationary = stationary

        self.arms = [0] * k
        # self.q_star = np.random.normal(0, 1, (num_problems, k)) #Random sample with mean=0, stddev=1
        self.q_star = np.random.normal(0, 1, size=k) #Random sample with mean=0, stddev=1

        self.optimal_arm = self.q_star.argmax()

    def new(self) -> object:
        """
        @Returns: a new Testbed object with same number of arms, problems, and stationarity
        """
        return Testbed(self.k, self.num_problems, self.stationary)
    
    def reset_true_value(self) -> None:
        """
        Resets mean reward value for all choices
        """
        self.q_star = np.random.normal(0, 1, size=self.k)
        self.optimal_arm = self.q_star.argmax()

    
    def generate_reward(self, arm):
        """
        Given a choice from 1..K, selects an option to get reward from

        @Returns: a random value selected from a normal distrubution around the true value
        of a given arm
        """
        if not self.stationary:
            self.q_star += np.random.normal(0, scale=0.03, size=self.k)     # walk rewards
            self.optimal_arm = self.q_star.argmax()
        return np.random.normal(self.q_star[arm], scale=1)
    
    
    def show_mean(self):
        """
        Displays scatterplot of arms and their mean reward at the time
        """
        print(f"Testbed Means: {self.q_star}")
        plt.figure(figsize=(12,8))
        plt.title("True Values", fontsize=14)
        plt.xlabel("Arms")
        plt.ylabel("Mean Reward")
        plt.xticks(range(1, self.k+1))
        plt.axhline(0, color="black", lw=1)
        # plt.plot(self.q_star)
        for i in range(0,self.k):
            plt.scatter(i+1, self.q_star[i])
            plt.text(i+1.15, self.q_star[i], f"q*({i+1}) = {self.q_star[i]:.3f}")
        # for i in range(0,10):
        #     plt.plot(self.q_star[i])
        plt.show()
    
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
    Parent Bandit class containing attributes and method declarations used by
    each bandit model.
    """
    def __init__(self, env: Testbed, model_params, steps, start_val) -> None:
        self.env = env              # reference to env
        self.k = env.k
        self.steps = steps
        
        self.model_params=model_params

        # NOTE: Total Actions should be total correct actions as it tracks the correct arm choices of all sims
        self.total_rewards = np.zeros(steps)      # rewards array
        self.total_actions = np.zeros(steps)      # Choices array

        self.selection_matrix = None                        # Contains all choice data for each step in each agent/problem
        self.reward_matrix = None                           # Contains all reward data for each step in each agent/problem
        self.predictionErr_matrix = None                    # Contains prediction error for each step in each agent/problem

        self.avg_rewards = None
        self.avg_actions = None

        self.final_N = None
    
    def simulate(self, num_problems=1000, save=False) -> None:
        """
        Runs the given model instance <num_problems> amount of times
        """
        pass

    def simulate_LL(self, num_problems=1000, save=False) -> None:
        """
        NOT IMPLEMENTED
        """
        pass

    def show_results(self):
        '''
        Displays the average reward data for each step of a simulation over each problem.
        If there is only one problem, then display raw reward data of that problem.
        '''
        pass
    def show_actions(self):
        pass
            
class EG_Bandit(Bandit):
    """
    eGreedy Bandit class that inherits all common attributes among the bandit models
    Key Parameters: Epsilon, Alpha (learning rate)
    """
    def __init__(self, env: Testbed, model_params: List, steps, start_val) -> None:
        super().__init__(env, model_params, steps, start_val)
        self.model_type = "eGreedy"


        # self.steps = steps

        self.alpha = model_params[0]           # if no alpha is passed of alpha=0, alg uses 1/n by default
        self.epsilon = model_params[1]

        self.initial_Q = start_val

        self.argmax_func = simple_max

    def simulate(self, num_problems=1000, save=False) -> None:

        self.selection_matrix = np.empty(num_problems, dtype=np.ndarray)
        self.reward_matrix = np.empty(num_problems, dtype=np.ndarray)
        self.predictionErr_matrix = np.empty(num_problems, dtype=np.ndarray)

        for i in tqdm(range(num_problems)):
            # Initialize Q Values
            Q = np.zeros(self.k) + self.initial_Q    # Qvalue array

            # Initialize arm choice array
            N = np.zeros(self.k)                    # number of times each arm is chosen (choices)
            
            # best_action = np.argmax(self.q_star[i])     # select best action for trial i

            rewards_arr = np.zeros(self.steps)
            chosen_arm_arr = np.zeros(self.steps, dtype=int)
            prediction_error_arr = np.zeros(self.steps)

            # perform trial steps
            for time_t in range(self.steps):
                best_action = self.env.optimal_arm

                if np.random.rand() < self.epsilon:
                    a = np.random.randint(self.k)         #Explore
                else:
                    a = self.argmax_func(Q)    #epxloit using chosen argmax_func ie. take best choice

                # get reward
                # reward = bandit(a, i, self.q_star)
                reward = self.env.generate_reward(a)

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

            # reset environment
            self.env.reset_true_value()

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
                selection = int(self.selection_matrix[i][time_t])     

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
                    reward = self.reward_matrix[i][time_t]

                    # compute prediction error
                    predError = reward - qValue[selection]

                    # Update reward - Non-stationary
                    qValue[selection] = qValue[selection] + self.alpha * predError

                    # Compute Likelihood
                    LL_array[time_t] = greedy_result[selection]

                    # if LL_array[time_t] <= 0:
                    #     LL_array[time_t] = 1e+300

            # Deal with Nans
            # LL_array[np.isnan(LL_array)] = 1e+300

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
    """
    Softmax Bandit variant
    """
    def __init__(self, env, model_params, steps, start_val) -> None:
        super().__init__(env, model_params, steps, start_val)
        self.model_type = "Softmax"

        self.alpha = model_params[0]    # learning param
        self.temp = model_params[1]     # temperature param

        # self.q_star = reward_values     # reward values
        self.initial_Q = start_val      # initial q

    def simulate(self, num_problems=1000, save=False) -> None:

        self.selection_matrix = np.empty(num_problems, dtype=np.ndarray)
        self.reward_matrix = np.empty(num_problems, dtype=np.ndarray)
        self.predictionErr_matrix = np.empty(num_problems, dtype=np.ndarray)


        for i in tqdm(range(num_problems)):
            # Initialize Q-values
            Q = np.zeros(self.k) + self.initial_Q
            
            # Initialize Arm
            N = np.zeros(self.k)


            # best_action = np.argmax(self.q_star[i])

            rewards_arr = np.zeros(self.steps)
            chosen_arm_arr = np.zeros(self.steps, dtype=int)
            prediction_error_arr = np.zeros(self.steps)

            for time_t in range(self.steps):
                best_action = self.env.optimal_arm

                # Choose arm
                a, _ = softmax(Q, (self.temp))
                # print(a)

                # Get reward
                # reward = bandit(a, i, self.q_star)
                reward = self.env.generate_reward(a)
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

            # reset environment    
            self.env.reset_true_value()
            
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
    # NOTE: Occassional runtime overflow exception in softmax alg
    def simulate_LL(self, num_problems=1000, save=False) -> None:
        self.total_LL_array = np.zeros(num_problems)

        for i in tqdm(range(num_problems)):
            LL_array = np.zeros([self.steps])
            qValue = np.zeros(self.k) * self.initial_Q

            for time_t in range(self.steps):

                selection = int(self.selection_matrix[i][time_t])

                if selection == -1:
                    LL_array[time_t] = 1
                
                else:
                    #compute Softmax values
                    num = np.exp(np.multiply(qValue, self.temp))
                    denom = sum(np.exp(np.multiply(qValue, self.temp)))

                    softmaxResult = num/denom
                    
                    reward = self.total_rewards[time_t]
                    
                    predError = reward - qValue[selection]
                    
                    # Update Reward – non-stationary
                    qValue[selection] = qValue[selection] + self.alpha * predError

                    # Compute Likelihood
                    LL_array[time_t] = softmaxResult[selection]

                    if LL_array[time_t] <= 0:
                        LL_array[time_t] = 1e+300
            
            #Deal with Nans
            LL_array[np.isnan(LL_array)] = 1e+300

            #Update Likelihood values
            LL_array_sum = -np.sum(np.log(LL_array))
            self.total_LL_array[i] = LL_array_sum
    
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
    def __init__(self, env, model_params, steps, start_val) -> None:
        super().__init__(env, model_params, steps, start_val)
        self.model_type = "Softmax-UCB"

        self.alpha = model_params[0]
        self.temp = model_params[1]
        self.uncertainty_param = model_params[2]

        self.initial_Q = start_val
    
    def simulate(self, num_problems=1000, save=False) -> None:
        
        # Initialize Matrices
        self.selection_matrix = np.empty(num_problems, dtype=np.ndarray)
        self.reward_matrix = np.empty(num_problems, dtype=np.ndarray)
        self.predictionErr_matrix = np.empty(num_problems, dtype=np.ndarray)


        for i in tqdm(range(num_problems)):
            Q = np.ones(self.k) * self.initial_Q        #Initialize Q values
            N = np.zeros(self.k)                        #actions/choices per trial

            # best_action = np.argmax(self.q_star[i])

            rewards_arr = np.zeros(self.steps)
            chosen_arm_arr = np.zeros(self.steps, dtype=int)
            prediction_error_arr = np.zeros(self.steps)

            # needed for uncertainty calculation
            n = 1

            for time_t in range(self.steps):
                best_action = self.env.optimal_arm
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
                reward = self.env.generate_reward(a)
                 
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
            
            # reset environment
            self.env.reset_true_value()

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
        self.total_LL_array = np.zeros(num_problems)
        n = 1

        for i in tqdm(range(num_problems)):
            LL_array = np.zeros(shape=[self.steps])
            qValue = np.zeros(self.k) * self.initial_Q

            for time_t in range(self.steps):
                selection = int(self.selection_matrix[i][time_t])

                if selection == -1:
                    LL_array[time_t] = 1

                else:
                    # Version from S&K 2015
                    lastcount = np.zeros(shape=self.k)
                    for cTc in range(self.k):
                        x = np.array(np.where(self.selection_matrix[i][0:time_t]))
                        if x.size == 0:
                            lastcount[cTc] = 0
                        else:
                            lastcount = np.amax(x)

                    uncert = (self.uncertainty_param * (n - lastcount))

                    # Calculate sm probabilities
                    num = np.exp(np.multiply(qValue + uncert, self.temp))
                    denom = sum(np.exp(np.multiply(qValue + uncert, self.temp)))

                    # Find softmax result
                    softmaxResult = num/denom

                    reward = self.reward_matrix[i][time_t]

                    # Compute Prediction error
                    predError = reward - qValue[selection]
                    
                    # Update Reward
                    qValue[selection] = qValue[selection] + self.alpha * predError

                    # Compute Likelihood array
                    LL_array[time_t] = softmaxResult[selection]

                    if LL_array[time_t] <= 0:
                        LL_array[time_t] = 1e300
            
            # Deal With Nans
            LL_array[np.isnan(LL_array)] = 1e+300

            # Update Likelihood vals
            LL_array_sum = -np.sum(np.log(LL_array))
            self.total_LL_array[i] = LL_array_sum


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

class VKF_Bandit(Bandit):
    """
    Volatile Kalman Filter Model
    (Measure -> Update -> Predict)

    Description: Computes mean and variance of the posterior distrubution of true average reward at time t
    https://speekenbrink-lab.github.io/modelling/2019/02/28/fit_kf_rl_1.html
    """
    def __init__(self, env, model_params, trial_params, start_val) -> None:
        super().__init__(env, model_params, trial_params, start_val)
        self.model_type = "Volatile Kalman Filter"

        # self.k = trial_params[0]
        # self.steps = trial_params[1]

        # self. = model_params[0]  # constant and does not change
        self.observation_noise = model_params[0]    # constant and does not change
        self.lamda = model_params[1]   # Volatility update param | Constrained to unit range

        self.initial_volatility = model_params[2]       # v_0
        self.initial_posterior_mean = model_params[3]        # <----------------------
        self.initial_posterior_var = model_params[4]        # <----------------------

        self.temp = model_params[5]

        self.intitial_Q = start_val


    def simulate(self, num_problems=1000, save=False) -> None:
        self.num_problems = num_problems
        
        self.selection_matrix = np.empty(num_problems, dtype=np.ndarray)
        self.reward_matrix = np.empty(num_problems, dtype=np.ndarray)
        self.volatility_matrix = np.empty(num_problems, dtype=np.ndarray)       # Prediction error

        for i in tqdm(range(num_problems)):
            Q = np.ones(self.k) + self.intitial_Q
            N = np.zeros(self.k) 

            rewards_arr =  np.zeros(self.steps)
            chosen_arm_arr = np.zeros(self.steps, dtype=int)

            volatility_arr = np.zeros(self.steps) + self.initial_volatility
            posterior_mean_arr = np.zeros(self.steps) + self.initial_posterior_mean
            posterior_variance_arr = np.zeros(self.steps) + self.initial_posterior_var

            posterior_mean = self.initial_posterior_mean
            posterior_variance = self.initial_posterior_var
            volatility = self.initial_volatility
            
            for time_t in range(self.steps):
                best_action = self.env.optimal_arm


                # Action Selection
                # a = simple_max(Q)
                a, _ = softmax(Q, self.temp)                              # NOTE: Function softmax
                N[a] += 1
                chosen_arm_arr[time_t] = a

                outcome =  self.env.generate_reward(a)
                rewards_arr[time_t] = outcome
                self.total_rewards[time_t] += outcome

                kalman_gain = np.divide(posterior_variance + volatility, posterior_variance + volatility + np.square(self.observation_noise)) 
                
                # Mean update
                posterior_mean_new = posterior_mean + kalman_gain * (outcome - posterior_mean)
                
                # Posterior variance update
                posterior_variance_new = (1 - kalman_gain) * (posterior_variance + volatility)
                
                # Covariance Update
                posterior_covariance = (1 - kalman_gain) * posterior_variance

                # Volatility Update
                volatility = volatility + self.lamda * ((np.square(posterior_mean_new - posterior_mean) + posterior_variance + posterior_variance_new - 2 * posterior_covariance - volatility))

                # Update
                # Q[a] = Q[a] + posterior_mean_new + posterior_variance_new
                # Q[a] = posterior_mean_new + posterior_variance_new
                Q[a] = posterior_mean_new

                if a == best_action:
                    self.total_actions[time_t] += 1

            # Reset env
            self.env.reset_true_value()

            # Update Model Matrices
            self.selection_matrix[i] = chosen_arm_arr
            self.reward_matrix[i] = rewards_arr
            self.volatility_matrix[i] = volatility_arr

        self.avg_rewards = np.divide(self.total_rewards, num_problems)
        self.avg_actions = np.divide(self.total_actions, num_problems)

    def LL_VKF(self, lamda, initial_V, temp) -> np.float64:        # Parameters to be fitted
        """
        Negative Log Likelihood function for VKF variant
        @Parameters: 
            lamda: Volatility update parameter
            initial_V: Initial volatility
            temp: temperature parameter for softmax function
        @Returns: 
            Total Negative Log Likelihood of each choicegiven a set of parameters and choice reward data

        Note: Current implementation will average NLL if provided a model with > 1 problems
        """
        likelihood_sum_arr = np.empty(self.num_problems, np.ndarray)

        for i in range(self.num_problems):
            q_value = np.zeros(self.k) + self.intitial_Q
            likelihood_array = np.zeros(self.steps)

            volatility_update = lamda
            volatility = initial_V

            posterior_mean = self.initial_posterior_mean
            posterior_variance = self.initial_posterior_var
            observation_noise = self.observation_noise

            for j in range(len(self.selection_matrix[i])):
                choice = self.selection_matrix[i][j]
                reward = self.reward_matrix[i][j]

                _, sm = softmax(q_value, temp)
                # print(_,softmax_result)
                softmax_result = sm[choice]
                kalman_gain = np.divide(posterior_variance + volatility, posterior_variance + volatility + np.square(observation_noise))
                
                 # Mean update
                posterior_mean_new = posterior_mean + kalman_gain * (reward - posterior_mean)
                
                # Posterior variance update
                posterior_variance_new = (1 - kalman_gain) * (posterior_variance + volatility)
                
                # Covariance Update
                posterior_covariance = (1 - kalman_gain) * posterior_variance

                # Volatility Update
                volatility = volatility + volatility_update * ((np.square(posterior_mean_new - posterior_mean) + posterior_variance + posterior_variance_new - 2 * posterior_covariance - volatility))

                #Update
                # q_value[choice] = q_value[choice] + posterior_mean_new + posterior_variance_new
                # q_value[choice] = posterior_mean_new + posterior_variance_new
                q_value[choice] = posterior_mean_new

                likelihood_array[j] = softmax_result
                if likelihood_array[j] <= 0:
                    likelihood_array[j] = 1e300
            likelihood_array[np.isnan(likelihood_array)] = 1e+300

            likelihood_sum = -np.sum(np.log(likelihood_array))
            likelihood_sum_arr[i] = likelihood_sum
            
        # return likelihood_sum_arr

        avg_likelihood_sum = np.average(likelihood_sum_arr)
        
        # print(f"Average NLL Sum: {avg_likelihood_sum} after {i+1} times")
        return avg_likelihood_sum
    
    def vary_param_NLL(self, p_update: str, interval:list, **parameters):
        """
        Function used to vary parameters used to test VKF_NLL.
        @Parameters:
            p_update: name of parameter to vary
            interval: list containing values to test p_update with in the form [start,stop,step]
            **parameters: list of keyword args defining the constant parameters
        @Returns:
            [0]: Array of Log Likelihood (Y)
            [1]: Array of tested parameters (X)
        """

        start = interval[0]
        stop = interval[1]
        step = interval[2]

        param_count = int((stop-start) // step)
        print("number of parameters:\n", param_count)
        LL_arr = np.zeros(shape=param_count)
        tested_parameters = np.zeros(shape=param_count)


        for i in tqdm(range(param_count)):
            parameters[p_update] = start
            # print()
            LL_arr[i] = self.LL_VKF(parameters["v_update"], parameters["v_init"], parameters["temp"])
            tested_parameters[i] = parameters[p_update]
            start += step

        return LL_arr, tested_parameters
    
    def plot_parameter_likelihood(self, *args, **kwargs) -> None:
        """
        Plots the change in NLL of a model given a parameter and an interval to vary
        @Parameters:
            args: arguments of vary_param_NLL functions
            kwargs: key-value pairs of constant parameters
        """
        print("Plotting the following parameters:")
        plt.figure(figsize=(12,6))
        plt.title("VKF Parameter NLL")
        data, parameters = self.vary_param_NLL(args[0], args[1], **kwargs)
        # print(data )
        # print(parameters)
        # print(args[1])
        plt.plot(parameters, data, 'g')
        plt.xlabel(f"Parameter: {args[0]}")
        plt.ylabel("NLL")
        plt.show()

    def show_results(self):
        print("="*30)
        print("Showing the following test: ",
              f"Model name: {self.model_type}",
              f"Action selection: {self.model_type} | temp = {self.temp}",
              f"Number of arms: {self.k}",
              f"Steps: {self.steps}",
              f"Average reward: {np.average(self.avg_rewards)}",
              sep="\n\t")
        print("="*30)
        plt.figure(figsize=(12,6))
        plt.title(f"{self.model_type}")
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

    
def create_bandit_task(model_type:str , env:Testbed, model_params:List, steps:int, start_val:np.float64) -> EG_Bandit | SM_Bandit | SMUCB_Bandit | VKF_Bandit:
    """
    Function that returns a Bandit object of model_type
    @Parameters:
        model_type: Initials of bandit model
        env: Testbed object
        model_params: list of parameters specific to the model_type
        steps: number of choices agent will make
        start_val: initial choice value
    @Returns:
        Bandit Object
    """
    if model_type == "EG":
        return EG_Bandit(env, model_params, steps, start_val)
    elif model_type == "SM":
        return SM_Bandit(env, model_params, steps, start_val)
    elif model_type == "SMUCB":
        return SMUCB_Bandit(env, model_params, steps, start_val)
    elif model_type == "VKF":
        return VKF_Bandit(env, model_params, steps, start_val)
    else:
        raise ValueError("Model type not accepted")
    
def model_performance_summary(bandits: list[Bandit]):
    """
    INCOMPLETE
    """
    fig, axs = plt.subplots(3,3, figsize=(12,12), dpi=500)
    fig.suptitle("Simulated Performance", fontsize = 35)

    for b in range(len(bandits)):
        # Optimal arm choice
        axs[b,0].plot(bandits[b].avg_actions, color='green')
        axs[b,0].set_xlabel("Trials", fontsize=12)
        axs[b,0].set_ylabel("Optimal Arm (%)", fontsize=12)
        axs[b,0].set_ylim(0,1)
        # axs[b,0].set_xticks()

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
        axs[b,2].set_ylim(-100, 5)

    plt.tight_layout()
    plt.show()
    
