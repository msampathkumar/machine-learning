import random
import pdb
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
        # row will be state - light, oncoming, left, right
        # col will be action - None, forward, left, right
        self.Q_table = [[0] * 4 ] * 4
        # learning variable
        self.gamma = 0.8
        #
        self.state = []

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = [inputs[_] for _ in ALL_ACTIONS]
        
        # TODO: Select action according to your policy
        action = None
        if GEAR == 0:
            # manual driving - set
            pdb.set_trace() # update action accordingly
        elif GEAR == 1:
            # random
            action = (None, 'forward', 'left', 'right')[random.randrange(0,4)]
        elif GEAR == 2:
            # controlled
            if inputs['light'] == 'red' and \
             not (inputs['left'] == inputs['left'] == inputs['left'] == None):
                action = None
            else:
                action = self.next_waypoint
        elif GEAR == 3:
            # reckless
            action = self.next_waypoint
        elif GEAR == 4:
            # controlled2
            if inputs['light'] == 'red' and \
             not (inputs['left'] == inputs['left'] == inputs['left'] == None):
                action = None
                if self.next_waypoint == 'right':
                    action = self.next_waypoint
            else:
                action = self.next_waypoint

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=2)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
