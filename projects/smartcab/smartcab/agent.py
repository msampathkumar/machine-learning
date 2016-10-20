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

ALL_STATES = ['light', 'oncoming', 'left', 'right']
ALL_ACTIONS = [None, 'forward', 'left', 'right']

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.state = None
        self.reward = None
        self.action = None
        self.alpha = 0.8
        self.gamma = 0.8
        self.epsilon = 0.8
        self.prev_state = None
        self.prev_reward = None
        self.prev_action = None
        self.Q = dict()

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def get_q_val(self, state, action):
        try:
            return self.Q[(self.state, action)]
        except KeyError:
            self.Q[(self.state, action)] = 0
        return 0

    def best_q_action(self, s):
        # greedy selection to find the best action according to Q
        # returns action & reward
        max_reward = 0
        best_action = ''
        if self.prev_state or (random.random() > self.epsilon):
            for act in ALL_ACTIONS:
                tmp = self.get_q_val(s, act)
                if max_reward <= tmp:
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
            self.Q[(s, a)] = self.get_q_val(s, a) + \
                    (self.alpha * ( r + \
                            self.gamma * (
                                # max([self.get_q_val(s1, a1) - self.get_q_val(s, a)]) )
                                self.get_q_val(s1, a1) - self.get_q_val(s, a)
                            )
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
        print 'LearningAgent.update(): expected_reward = {}, received_reward = {}'.format(expected_reward, reward)

        # TODO: Learn policy based on state, action, reward
        self.update_q_policy()

        # passing values for next state
        self.prev_state = state
        self.prev_action = action
        self.prev_reward = reward
        if deadline == 0:
            pprint(self.Q)
            # pdb.set_trace()
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0005, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
