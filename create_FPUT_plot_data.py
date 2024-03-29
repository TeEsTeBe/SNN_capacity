from run_FPUT import main as run_FPUT
from FPUT.FPUT_input import filename
from run_capacity import main as actually_run_capacity
from evaluate_task_separately import main as actually_run_task
from evaluate_task_separately import setup_runfolder as get_taskfolder


from itertools import product
from pathlib import Path
import os
import numpy as np
from multiprocessing import Pool
import multiprocessing as mp

def run_task(args, overwrite_task_data=False):
    runpath = get_taskfolder(args)
    if (not overwrite_task_data) and os.path.isfile(Path(runpath) / 'test_results.yml'):
        return
    actually_run_task(args)

def run_capacity(args, overwrite_capacity_data=False):
    runpath = get_taskfolder(args)
    if (not overwrite_capacity_data) and os.path.isfile(args.capacity_results):
        return
    actually_run_task(args)

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def generate_data(input_duration, input_amplitude, trial):
    FPUT_args = dict(
        alpha=0.25,
        tau_relax=10,
        nbr_batches=100000,
        warmup_batches=10,
        init_epsilon=0.05,
        trial=trial,
        force=False,
        in_dim=1,
        uniques=4,
        in_width=64,
        in_variance=1/3,
        input_amplitude=input_amplitude,
        input_duration=input_duration,
        osc=64
    )
    print(120 * '--', input_duration, input_amplitude, trial)

    # Task stuff
    FPUT_args['discrete'] = True
    run_FPUT(FPUT_args)
    fput_file_prefix = filename(prefix="./data/FPUT/", postfix="/", **FPUT_args)
    fput_file_prefix = Path(fput_file_prefix)
    metadata_file = fput_file_prefix / "metadata.yaml"
    trajectory_file = fput_file_prefix / "trajectories.npy"
    input_file = fput_file_prefix / "scaled_input_seq.npy"

    assert(os.path.isfile(metadata_file))


    task_results = filename(prefix="", postfix="", **FPUT_args)
    FPUT_args['force'] = False
    XOR_args = Namespace(
        input=input_file,
        states=trajectory_file,
        steps=100000,
        task='xor',
        groupname='xor',
        runname=task_results,
        test_ratio=0.3,
        seed=None,
        data_path='./data/FPUT_tasks',
        trial=trial,
    )
    run_task(XOR_args)

    # temporal xor specific stuff
    FPUT_args = dict(
        alpha=0.25,
        tau_relax=10,
        nbr_batches=1000,
        warmup_batches=10,
        init_epsilon=0.05,
        trial=trial,
        force=False,
        in_dim=1,
        uniques=2,
        in_width=64,
        in_variance=1/3,
        input_amplitude=input_amplitude,
        input_duration=input_duration,
        osc=64
    )
    FPUT_args['discrete'] = True

    run_FPUT(FPUT_args)
    fput_file_prefix = filename(prefix="./data/FPUT/", postfix="/", **FPUT_args)
    fput_file_prefix = Path(fput_file_prefix)
    metadata_file = fput_file_prefix / "metadata.yaml"
    trajectory_file = fput_file_prefix / "trajectories.npy"
    input_file = fput_file_prefix / "scaled_input_seq.npy"

    FPUT_args['force'] = False
    TEMPORAL_XOR_args = Namespace(
        input=input_file,
        states=trajectory_file,
        steps=1000,
        task='temporal_xor',
        groupname='t_xor',
        runname=task_results,
        test_ratio=0.3,
        seed=None,
        data_path='./data/FPUT_tasks',
        trial=trial,
    )
    run_task(TEMPORAL_XOR_args)
    # capacity stuff
    FPUT_args = dict(
        alpha=0.25,
        tau_relax=10,
        nbr_batches=100000,
        warmup_batches=10,
        init_epsilon=0.05,
        trial=trial,
        force=False,
        in_dim=1,
        uniques=4,
        in_width=64,
        in_variance=1/3,
        input_amplitude=input_amplitude,
        input_duration=input_duration,
        osc=64
    )
    FPUT_args['discrete'] = False
    run_FPUT(FPUT_args)
    fput_file_prefix = filename(prefix="./data/FPUT/", postfix="/", **FPUT_args)
    fput_file_prefix = Path(fput_file_prefix)
    metadata_file = fput_file_prefix / "metadata.yaml"
    trajectory_file = fput_file_prefix / "trajectories.npy"
    input_file = fput_file_prefix / "scaled_input_seq.npy"
    assert(os.path.isfile(metadata_file))

    capacity_results = filename(prefix="./data/FPUT_capacities/", postfix=".pkl", **FPUT_args)


    capacity_args = Namespace(
        name='fput',
        input=input_file,
        states_path=trajectory_file,
        results_file=None,
        capacity_results=capacity_results,
        max_degree=100,
        max_delay=1000,
        m_variables=True,
        m_powerlist=True,
        m_windowpos=True,
        orth_factor=2.,
        figures_path='figures',
        n_warmup=0,
        use_scipy=True,
        sample_ids=None,
        sample_size=None,
        sample_step=None,
        delskip=0,
        windowskip=0,
        verbosity=0,
    )

    run_capacity(capacity_args)


    narma_input_file = fput_file_prefix / "narma_input_seq.npy"
    input_seq = np.reshape(np.load(input_file), -1)
    np.save(arr=input_seq, file=narma_input_file)
    FPUT_args['force'] = False
    NARMA_args = Namespace(
        input=narma_input_file,
        states=trajectory_file,
        steps=100000,
        task='NARMA5',
        groupname='narma5',
        runname=task_results,
        test_ratio=0.3,
        seed=None,
        data_path='./data/FPUT_tasks',
        trial=trial,
    )
    run_task(NARMA_args)

if __name__ == "__main__":

    input_durations=range(1,50)
    input_amplitudes = [ 
        0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010,
        0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.020,
        0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.030,
        0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.040,
        0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.050,
    ]
    trials = 5
    multiprocessing = True

    if multiprocessing:
        pool = mp.Pool(mp.cpu_count())
        pool.starmap(generate_data, product(input_durations, input_amplitudes, range(trials)))
    else:
        for input_duration, input_amplitude, trial  in product(input_durations, input_amplitudes, range(trials)):
            generate_data(input_duration=input_duration, input_amplitude=input_amplitude, trial=trial)

