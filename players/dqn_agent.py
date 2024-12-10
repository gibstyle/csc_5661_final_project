import numpy as np
import copy
import seaborn as sb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .player import Player

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size * 2)    #First fully connected layer
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)    #Second fully connected layer
        self.fc3 = nn.Linear(hidden_size, 1)    #Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))    #ReLU activation for the first layer
        x = F.relu(self.fc2(x))    #ReLU activation for the second layer
        x = self.fc3(x)    #Output layer (no activation because this is a regression problem)
        return x

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class DQNAgent(Player):
    '''A class to manage the agent to play battleship'''
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def __init__(self, id: int, team_id: int, train: bool, config: dict, name="DQN Agent"):
        '''Set up the constructor
            Takes -- config, a dictionary specifying the board dimensions and initial state
        '''
        super().__init__(id=id, team_id=team_id, train=train)
        self.config = config
        self.name=name
        self.Q = MLP(input_size=self.config['input_size'], hidden_size=self.config['hidden_size'])
        self.Q.train()    #Set the model to training mode
        self.Q_prime = copy.deepcopy(self.Q)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.config['alpha'])
        self.D = []  # init the replay buffer
        self.p = []  # init the replay probs
        self.A = self.config['A']

        self.double_dqn = self.config['double_dqn']  # using double dqn?
        self.prioritized_replay = self.config['prioritized_replay']  # using prioritized replay?
        self.multi_step = self.config['multi_step']  # using multistep?
        if not self.multi_step:  # if not using multistep, ensure the n_step_return set to 0
            self.config['n_step_return'] = 0

    def choose_action(self, state):
        hand = self.get_trick_hand(state)
        self.config['A'] = hand
        action = self.pi(s_t=state['s_t'], epsilon=self.epsilon_t(count=state['count'], n_episodes=state['episode']))
        self.config['A'] = self.A
        self.update_hand(action)
        return action
    
    def train_update(self, state_t_1, reward_t_1, count):
        self.data_t['r_t+1'] = reward_t_1
        self.data_t['s_t+1'] = state_t_1
        self.D.append(self.data_t)
        self.p.append(5 ** self.config['omega'])
        if len(self.D) > self.config['M']:
            self.D.pop(0)
            self.p.pop(0)
        
        if count % self.config['n_steps'] == 0:    #if it is time for a new update...
            batch = self.make_batch()    #make a batch for training from the memory buffer
            X = batch[0]    #pull out the features
            y = batch[1]    #pull out the target
            self.update_Q(X,y)    #update the MLP modeling Q

        if count % self.config['C'] == 0:    #if it is time for an update...
            self.update_Q_prime()    #overwrite the target approximator

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def Q_reset(self):
        '''A function reset the MLP to random initial parameters'''
        self.Q = MLP(input_size=self.config['input_size'], hidden_size=self.config['hidden_size'])
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def update_Q_prime(self):
        '''A function reset the MLP to random initial parameters'''
        self.Q_prime = copy.deepcopy(self.Q)
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
    def make_options(self, s_t):
        '''A function to create the state action pairs
            Takes:
                s_t -- a list of the state information
            Returns:
                a torch tensor with the first six columns the state information and the last two columns the actions
        '''
        s_tA = []    #init a list to hold the state action information
        for a in self.config['A']:    #loop over actions
            s_tA.append(s_t + [self.config['card_values'][a]])    #add and record
        return torch.tensor(s_tA).to(torch.float32)

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
    def epsilon_t(self, count, n_episodes):
        '''Lets try out a dynamic epsilon
            Takes:
                count -- int, the number of turns so far
            Returns:
                float, a value for epsilon
        '''
        if count <= self.config['epsilon_burnin']:    #if we're still in the initial period...
            return 1    #choose random action for sure
        else:
            return 1/(n_episodes**0.5)    #otherwise reduce the size of epsilon

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
    def pi(self, s_t, epsilon):
        '''A function to choose actions using Q-values
            Takes:
                s_t -- a torch tensor with the first six columns the state information and the last two columns the actions
                epsilon -- the probability of choosing a random action
        '''
        if np.random.uniform() < epsilon:    #if a random action is chosen...
            a = self.config['A'][np.random.choice(a = range(len(self.config['A'])))]     #return the random action
        else:
            a = self.config['A'][torch.argmax(self.Q(self.make_options(s_t)))]    #otherwise return the action with the highest Q-value as predicted by the MLP
        return a
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
    def n_step_return(self, idx):
        """
        Calculate the n-step return.
        """
        if 'G_t_n' not in self.D[idx]:  # need to calculate the n-step return
            G_t_n = 0  # init the n-step return
            gamma = self.config['gamma']  # set param for gamma as this will change based on which step
            lst = list(range(idx, idx + self.config['n_step_return']))  # create list of future steps
            flag = True  # flag for if done is found, will be set to False
            for k in lst:  # loop through n-steps
                if self.D[k]['done']:  # if current step is done
                    self.D[idx]['s_t+n'] = self.D[k]['s_t+1']  # set first state to current step state
                    flag = False
                    break
                else:
                    G_t_n += gamma * self.D[k]['r_t+1']  # add the current step reward to n-step return
                gamma *= self.config['gamma']  # update gamma
            if flag:  # if done was not found, set first state to last n-step state
                self.D[idx]['s_t+n'] = self.D[idx + self.config['n_step_return']]['s_t']
            return G_t_n  # return n-step return
        return None  # n-step return already exists so return None

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def make_batch(self):
        '''A function to make a batch from the memory buffer and target approximator
            Returns:
                a list with the state-action pair at index 0 and the target at index 1
        '''
        if self.prioritized_replay:  # if using prioritized replay
            batch_indices = np.random.choice(range(len(self.D) - self.config['n_step_return']), self.config['B'], p=(np.array(self.p[0: len(self.p) - self.config['n_step_return']]) ** self.config['beta']  / np.sum(np.array(self.p[0: len(self.p) - self.config['n_step_return']]) ** self.config['beta'])))
        else:   # sample uniformly with no prioritization
            batch_indices = np.random.choice(range(len(self.D) - self.config['n_step_return']), self.config['B'])

        X,y = [],[]    #init the state-action pairs and target
        for idx in batch_indices:    #loop over all the data collected
            d = self.D[idx]
            X.append(d['d_s_a'])    #record the state action pair
            
            if not self.multi_step:  # is no using multi-step
                y_t = d['r_t+1']    #compute the target
                options = self.make_options(d['s_t+1'])  # generate state action pairs
            else:  # if using multi-step
                G_t_n = self.n_step_return(idx)  # calculate n-step return
                if G_t_n is not None:  # if no n-step return
                    self.D[idx]['G_t_n'] = G_t_n
                d = self.D[idx]
                y_t = d['G_t_n']  # compute target
                options = self.make_options(d['s_t+n'])  # generate state action pairs
                
            approximator_main = self.Q(options)  # main appoximator
            approximator_target = self.Q_prime(options)  # target approximator
            
            
            if not self.double_dqn:  # if not double dqn
                max_a_Q = float(max(approximator_target))    #compute the future value using the target approximator
            else:  # if double dqn
                max_a_Q = float(self.Q_prime(options[torch.argmax(approximator_main)]))  # compute target for online main approximator
            y_t += self.config['gamma'] * max_a_Q    #update the target with the future value

            if self.prioritized_replay:  # if using prioritized replay, update the priority of the current data
                self.p[idx] = abs(d['r_t+1'] + self.config['gamma'] * float(max(approximator_target)) - float(self.Q(torch.tensor(d['d_s_a']).to(torch.float32)))) + self.config['omega']

            y.append(y_t)    #record the target   
        return [X, y]

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
    def update_Q(self,X,y):
        '''A function to update the MLP
            Takes:
                X -- the features collected from the replay buffer
                y -- the targets
        '''
        #do the forward pass
        X = torch.tensor(X).to(torch.float32)
        y = torch.tensor(y).to(torch.float32).view(len(y),1)
        outputs = self.Q(X)    #pass inputs into the model (the forward pass)
        loss = self.criterion(outputs,y)    #compare model outputs to labels to create the loss

        #do the backward pass
        self.optimizer.zero_grad()    #zero out the gradients    
        loss.backward()    #compute gradients
        self.optimizer.step()    #perform a single optimzation step
