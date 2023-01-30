import os
import argparse

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

import tasks.readout as readout
from tasks.utils.general_utils import setup_seeding
from tasks.utils.tasks_helper import get_task


def get_full_runname(args):
    full_runname = f'{args.runname}_task={args.task}_testratio={args.test_ratio}_steps={args.steps}_trial={args.trial}'

    return full_runname


def setup_runfolder(args):
    full_runname = get_full_runname(args)
    runpath = os.path.join(args.data_path, args.groupname, full_runname)
    os.makedirs(runpath, exist_ok=True)

    return runpath


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', help='Path to input numpy array.')
    parser.add_argument('--states', help='Path to statematrix numpy array.')
    parser.add_argument('--steps', type=int, help='Number of steps for the task.')
    parser.add_argument('--task', help='Name of the task that should be performed.')
    parser.add_argument('--groupname', type=str, default='defaultgroup', help='Name of the group the run belongs to.')
    parser.add_argument('--runname', type=str, default='defaultname', help='Name of the current run. A folder with this name and the parameters will be created.')
    parser.add_argument('--test_ratio', type=float, default=0.3, help="Ratio of the total number of steps to be used as test set")
    parser.add_argument('--seed', type=int, default=None, help='Seed for numpy random number generators')
    parser.add_argument('--data_path', type=str, default='data', help='Path to the parent directory where the group '
                                                                      'folder and run folder are created '
                                                                      '(the results will be stored in data_path/groupname/runname')
    parser.add_argument('--trial', type=int, default=1, help="Trial number. Can be used if you want to run multiple trials with the same parameters.")

    return parser.parse_args()


def main(args):
    args.seed = setup_seeding(args.seed)
    runpath = setup_runfolder(args)
    task = get_task(args.task, args.steps, args.input)
    states = np.load(args.states)
    if states.shape[0] < states.shape[1]:
        states = states.T
    states_filter = task.get_state_filter()
    states = states[states_filter]

    train_states, test_states, train_targets, test_targets = train_test_split(states, task.target, test_size=args.test_ratio)
    reconstructed_signal_train, regression = readout.train_readout(train_states, train_targets)

    evaluator = task.get_default_evaluator()
    train_results = evaluator.evaluate(train_targets, reconstructed_signal_train)
    print('==== Training ====')

    print(train_results)
    with open(os.path.join(runpath, 'train_results.yml'), 'w') as train_file:
        yaml.dump(train_results, train_file)

    reconstructed_signal_test = regression.predict(test_states)
    test_results = evaluator.evaluate(test_targets, reconstructed_signal_test)
    print('==== Testing ====')
    print(test_results)
    with open(os.path.join(runpath, 'test_results.yml'), 'w') as test_file:
        yaml.dump(test_results, test_file)


if __name__ == '__main__':
    main(args=parse_cmd())

