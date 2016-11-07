'''

Welcome to Smart Cab Log Analyser !!!

This is a simple script to do ETL on Agent.py log/output data to analyse the performance of Agent Learning

* trail : success or failure
* deadline : last deadline value
* reward : last reward value

You can redirect the trails logs posted by smartcab agent to a log_file and provide it as a input to analyse.

Note:
    * for learning update outformat is expecter as below. You need to add EXPECTED_REWARD to the following
    
    LearningAgent.update(): deadline = 30, inputs = {'light': 'red', 'oncoming': None, 'right': None, 'left': None}, action = None, expected_reward = 0, reward = 0.0
'''

import sys

from re import findall as re_findall

import pandas as pd

FILE = 'smartcab/test_log100'

def fetch_data(filename):
    data = open(filename).readlines()
    return data

def fetch_value(inp_str):
    return float(inp_str.split('=')[-1])

def check_environment_reset(env_reset):
    data = []
    start, end, deadline = '', '', ''
    if env_reset.startswith('Environment.reset()'):
        data = re_findall('[-\w.]+', env_reset)
    for i, each in enumerate(data):
        if each == 'start':
            start = map(int, [data[i + 1], data[i + 2]])
        elif each == 'destination':
            end = map(int, [data[i + 1], data[i + 2]])
        elif each == 'deadline':
            deadline = int(data[i+1])
    return start, end, deadline

def check_learning_update(learning_update):
    deadline, expected_reward, reward = 0, 0, 0
    data = []
    if learning_update.startswith('LearningAgent.update()'):
        data = re_findall('[-\w.]+', learning_update)
    for i, each in enumerate(data):
        if each == 'deadline':
            deadline = data[i + 1]
        elif each == 'expected_reward':
            expected_reward = data[i + 1]
        elif each == 'reward':
            reward = data[i + 1]
    return map(float, [deadline, expected_reward, reward])


def check_learning_update_old(learning_update):
    deadline, expected_reward, reward = 0, 0, 0
    if learning_update.startswith('LearningAgent.update()'):
        learning_update = learning_update[24:].split(', ')
        # values
        deadline = fetch_value(learning_update[0])
        expected_reward = fetch_value(learning_update[-2])
        reward = fetch_value(learning_update[-1])
    return deadline, expected_reward, reward


def success_check(data):
    '''
    Sequentially Parses the log and fetches data as and when we get
    '''
    all_outcomes = []
    all_trails = []
    all_deadlines = []
    all_expected_rewards = []
    # vars
    all_rewards = []
    all_deadlines = []
    all_start = []
    all_destinations = []
    all_main_deadlines = []
    # vars
    trail_last_update = False
    tmp_success = -1
    counter = 0
    reached_msg, aborted_msg = 'Primary agent has reached destination', 'Trial aborted'

    for i, each in enumerate(data):
        if each.startswith('Environment.reset()'):
            counter += 1
            start_point, destination_point, main_deadline = check_environment_reset(each)

        if each.startswith('LearningAgent.update()'):
            deadline, expected_reward, reward = check_learning_update(each)
            #
            all_trails.append(counter)
            all_outcomes.append(-1)
            all_rewards.append(reward)
            #
            all_expected_rewards.append(expected_reward)
            all_start.append(start_point)
            all_destinations.append(destination_point)
            all_deadlines.append(deadline)
            all_main_deadlines.append(main_deadline)

        if trail_last_update:
            trail_last_update = False
            all_outcomes[-1] = tmp_success
            tmp_success = -1
        # trail end check - if it is
        reached_chk = reached_msg in each 
        aborted_chk = aborted_msg in each
        # trail ended
        if reached_chk or aborted_chk:
            trail_last_update = True
            if reached_chk:
                tmp_success = 1
            else:
                tmp_success = 0
    ret_dict = {
        'all_trails' : all_trails,
        'all_outcomes' : all_outcomes,
        'all_rewards' : all_rewards,
        'all_deadlines' : all_deadlines,
        'all_expected_rewards' : all_expected_rewards,
        'all_start' : all_start,
        'all_destinations' : all_destinations,
        'all_main_deadline' : all_main_deadlines
        }
    return ret_dict


def total_stats(filename=FILE, return_dict=False):
    data = fetch_data(filename)
    game = success_check(data)
    game_stats = pd.DataFrame.from_dict(game)
    if return_dict:
        return game_stats

    for col in [u'all_deadlines',  u'all_expected_rewards', u'all_trails',
               u'all_main_deadline', u'all_outcomes', u'all_rewards', # u'all_start', u'all_destinations'
               ]:
        game_stats[col] = pd.to_numeric(game_stats[col])

    game_stats['Q_pred'] = game_stats.all_expected_rewards - game_stats.all_outcomes
    game_stats['steps'] = game_stats.all_main_deadline - game_stats.all_deadlines
    game_stats['avg_steps'] = game_stats.all_rewards / game_stats.steps

    #
    # Checking how many reached desctination
    #
    rewarded_deadlines = (game_stats.all_rewards >= 12) & (game_stats.all_deadlines >= 0)
    print '\nNo.of Successfully Trips is: %s' % game_stats[rewarded_deadlines].all_rewards.count()
    print 'Total reward sum: %s' % game_stats.all_rewards.sum()
    print 'Total time saved: %s' % int(game_stats[['steps']][rewarded_deadlines].sum())

if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = str(sys.argv[1])
        total_stats(filename)
    else:
        total_stats('q_log02')


