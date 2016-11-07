'''
Contains Agent for implementation of Q Learning Algorithms

Refs:
https://discussions.udacity.com/t/how-do-i-capture-two-states-in-order-to-implement-the-q-learning-algorithm/191327/2


'''

import random
import pdb
from pprint import pprint

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

GEAR = 2

reducer_func = lambda x: ( x * .95 // 0.001) * 0.001

ALL_STATES = ['light', 'oncoming', 'left', 'right']
ALL_ACTIONS = [None, 'forward', 'left', 'right']

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env) # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self) # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.state = None
        self.reward = None
        self.action = None
        self.Q = dict()
        self.alpha = .75 # learning rate
        self.gamma = .10 # discount factor
        self.epsilon = .35 # randomness
        self.counter = 0
        # A trails session varibles
        self.prev_state = None
        self.prev_reward = None
        self.prev_action = None

    def reset(self, destination=None):
        '''
        * Alpha(learning factor): When the problem is stochastic, the algorithm still converges
             under some technical conditions on the learning rate, that require it to decrease to zero.
             In practice, often a constant learning rate is used, such as 
             alpha(s,a)=0.1 for all t.
        * Gamma(discount factor): starting with a lower discount factor and increasing it towards its
             final value yields accelerated learning
        * Epsilon(exploration factor): higher the randomness higher the exploration
        '''
        self.planner.route_to(destination)
        random.shuffle(ALL_ACTIONS)
        self.prev_state = None
        self.prev_reward = None
        self.prev_action = None

        # learning rate
        if self.counter < 50:
            self.alpha -= 0.01
            # self.alpha = (self.alpha // 0.0001) * 0.0001

        # long term focus
        if self.counter < 20:
            self.gamma += 0.01

        # randomness
        if self.epsilon < 0:
            # self.epsilon -= 0.01
            self.epsilon = (self.epsilon * .9 // 0.0001) * 0.0001

        if self.counter >= 10 * 98:
            pprint(self.Q)
            # pdb.set_trace()

        self.counter += 10

    def get_q_val(self, state, action):
        try:
            return self.Q[(self.state, action)]
        except KeyError:
            # optimism in the face of uncertainty
            self.Q[(self.state, action)] = 0.05
        return 0

    def best_q_action(self, s):
        '''
        Greedy Selection - To find the best action according to Q

        Note: First selection after Q initialisation is always random
        '''
        max_reward = 0
        best_action = ''
        if self.prev_state or (random.random() > (1 - self.epsilon)):
            for act in ALL_ACTIONS:
                tmp = self.get_q_val(s, act)
                if not max_reward or max_reward <= tmp:
                    max_reward = tmp
                    best_action = act
        else:
            best_action = random.choice(ALL_ACTIONS)
            max_reward = self.get_q_val(s, best_action)
        return best_action, max_reward

    def update_q_policy(self):
        # Q Learning Policy Updation
        if self.prev_state:
            s, r, a = self.prev_state, self.prev_reward, self.prev_action
            s1, r1, a1 = self.state, self.reward, self.action
            #
            estimated_optimal_future_value = max(map(lambda x: self.get_q_val(s1, x), ALL_ACTIONS))
            self.Q[(s, a)] = self.get_q_val(s, a) + \
                    (self.alpha * ( r + \
                            # self.gamma * (
                                # max([self.get_q_val(s1, a1) - self.get_q_val(s, a)]) )
                                # self.get_q_val(s1, a1) - self.get_q_val(s, a)
                                self.gamma * estimated_optimal_future_value  - self.get_q_val(s, a)
                            # )
                        )
                    )

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        state = (inputs['light'],
                inputs['left'],
                inputs['oncoming'],
                inputs['right'],
                self.next_waypoint)
        self.state = state
        
        # TODO: Select action according to your policy
        action, expected_reward = self.best_q_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.reward = reward

        # TODO: Learn policy based on state, action, reward
        self.update_q_policy()

        # passing values for next state
        self.prev_state = state
        self.prev_action = action
        self.prev_reward = reward

        # [debug]
        params = (deadline, inputs, action, expected_reward, reward)
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, expected_reward = {}, reward = {}".format(*params)


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    show = True
    # NOTE: To speed up simulation, set show = False.
    # NOTE: To show the GUI, set show = True

    if show:
        # Now simulate it
        sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
        sim.run(n_trials=100)  # run for a specified number of trials
        # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    else:
        sim = Simulator(e, update_delay=0.0, display=False)
        sim.run(n_trials=5)



if __name__ == '__main__':
    run()
