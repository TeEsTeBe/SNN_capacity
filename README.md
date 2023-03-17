# SNN_capacity

This repository contains the code that you need to run the experiments presented in the paper 
"_Quantifying the processing capacity of continuous dynamical systems: from vibrating strings to cortical circuits_" by 
Tobias Schulte to Brinke, Michael Dick, Renato Duarte and Abigail Morrison.
We also warmly thank Joni Dambre for sharing her code that she used in her paper
[Information Processing Capacity of Dynamical Systems](https://www.nature.com/articles/srep00514) and that was the basis
for the ESN and capacity computation parts of our implementation.

To run the code in this repository, you have to install  [NEST 3](https://nest-simulator.readthedocs.io/en/v3.0/installation/index.html) first.

You also have to install the other requirements from the `requirements.txt` file.

Then add this repository to your PYTHONPATH:
```
export PYTHONPATH=$PYTHONPATH:/path/to/this/repository
```

## Simulation

### ESN simulation

To simulate the echo state network (ESN) you have to call the script `run_ESN.py`.
If you want to get the results from the paper you have to add the `--orthogonalize` flag and change the parameters
`--input_scaling` (in the paper called $\iota$) `--spectral_radius` (in the paper feedback gain or $\rho$). Then you 
can also adjust the parameters `--data_path`, `--groupname` and `--runname` to your needs. The results will then be 
stored in the folder `{--data_path}/{--groupname}/{--runname}`.

To get more information about the possible parameters you can call `python run_ESN.py --help` und you will get the 
following output:

```
usage: run_ESN.py [-h] [--input INPUT] [--runname RUNNAME] [--groupname GROUPNAME] [--data_path DATA_PATH] [--steps STEPS] [--nodes NODES] [--input_scaling INPUT_SCALING] [--spectral_radius SPECTRAL_RADIUS] [--n_warmup N_WARMUP] [--init_normal]
                  [--orthogonalize] [--ortho_density_denominator ORTHO_DENSITY_DENOMINATOR] [--use_relu] [--relu_slope RELU_SLOPE] [--relu_start RELU_START] [--use_linear_activation] [--recurrence_density RECURRENCE_DENSITY] [--trial TRIAL]
                  [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Path to input. If not set, the input is generated and stored to the run path.
  --runname RUNNAME     Name of the current run. A folder with this name and the parameters will be created inside the group path.
  --groupname GROUPNAME
                        Name of the group of experiments. A folder with this name will be created inside the data_path.
  --data_path DATA_PATH
                        Path were the results for all experiment groups should be stored.
  --steps STEPS         Number of inputs. Only used if --input file does not exist.
  --nodes NODES         Number of nodes the ESN is made of.
  --input_scaling INPUT_SCALING
                        Input scaling parameter iota
  --spectral_radius SPECTRAL_RADIUS
                        Spectral radius of the recurrent weights
  --n_warmup N_WARMUP   Number of warmup steps the ESN is simulated before the main simulation is started.
  --init_normal         Use normal distribution to initialize recurrent weigths.
  --orthogonalize       Orthogonalize the recurrent weigths
  --ortho_density_denominator ORTHO_DENSITY_DENOMINATOR
                        Create a sparse but orthogonalized weight matrix with density 1/ortho_density_denominator
  --use_relu            Use relu activation function.
  --relu_slope RELU_SLOPE
                        Slope of the relu function.
  --relu_start RELU_START
                        Offset for the relu function
  --use_linear_activation
                        use a linear activation function
  --recurrence_density RECURRENCE_DENSITY
                        Density of the recurrent weight matrix. For orthogonalization use --ortho_density_denominator!
  --trial TRIAL         Number of the current trial
  --seed SEED           Random seed for the simulation
```
### FPUT simulation

To simulate the Fermi-Pasta-Ulam-Tsingou oscillator chain first the cython code it is implemented in needs to be compiled.
```
cd ./FPUT
./build_sim_util.sh
cd -
```
a simulation can then be started by calling the script `run_FPUT.py`
```
usage: run_FPUT.py [-h] [--alpha ALPHA] [--tau_relax TAU_RELAX]
                   [--nbr_batches NBR_BATCHES]
                   [--warmup_batches WARMUP_BATCHES]
                   [--init_epsilon INIT_EPSILON] [--trial TRIAL] [--discrete]
                   [--force] [--in_dim IN_DIM] [--uniques UNIQUES]
                   [--in_width IN_WIDTH] [--in_variance IN_VARIANCE]
                   [--osc OSC]

options:
  -h, --help            show this help message and exit
  --alpha ALPHA         strength of nonlinearity
  --tau_relax TAU_RELAX
                        dampening time constant
  --nbr_batches NBR_BATCHES
                        number of batches
  --warmup_batches WARMUP_BATCHES
                        number of warmup batches
  --init_epsilon INIT_EPSILON
                        initial energy of the chain
  --trial TRIAL         trial parameter. doesnt change simulation, is just
                        averaged over for plot
  --discrete            if set, input will be discrete
  --force               if set, save files will be overwritten
  --in_dim IN_DIM       input dimension
  --uniques UNIQUES     how many unique values are to be presented in case of
                        discrete input. ignored if --discrete is not given
  --in_width IN_WIDTH   number of oscillators each input attaches to
  --in_variance IN_VARIANCE
                        variance of input
  --osc OSC             number of oscillators
```
The Data will be stored in `./data/FPUT/`.

If you want to get all data for the plots from the paper, there is a convenience script`create_FPUT_plot_data.py`

### SNN simulation
To simulate the spiking neural networks you have to call the `run_SNN.py` script with a path to a yaml file
as argument. This yaml file defines the parameters for your simulation.

#### BRN
Example parameters for the balanced random network: 
```yaml
data_dir: null
group_name: your-group-name
run_title: your-run-name
num_steps: 200000
step_duration: 50.0
input_type: spatial_DC
input_max_value: 100.0
input_min_value: 0.0
spatial_std_factor: 20
n_spatial_encoder: n_input_neurons
input_connection_probability: 1.0
network_type: brunel
network_params:
  N: 1250
  neuron_model: iaf_psc_delta
  neuron_params:
    C_m: 1.0
    E_L: 0.0
    I_e: 0.0
    V_m: 0.0
    V_reset: 10.0
    V_th: 20.0
    t_ref: 2.0
    tau_m: 20.0
noise_loop_duration: step_duration
stop_if_statemat_exists: true
raster_plot_duration: 1000.0
spike_recorder_duration: 10000.0
trial: 1
add_ac_current: false
background_rate: null
background_weight: null
batch_steps: 10000
dt: 0.1
```

You have to adjust the parameters `step_duration` ($\Delta s$), `input_max_value` ($a_{max}$) and `input_type` to parameterize the input to the network.
The input type defines the encoding (step, uniform or spatial), whether rate or DC is used and possibly a task (if it is
not for capacity calculation). So the `input_type` can have the following values:

For capacity calculation:
- 'uniform_DC'
- 'uniform_rate'
- 'step_rate'
- 'step_DC'
- 'spatial_rate'
- 'spatial_DC' 
 
For tasks: 
- 'uniform_DC_classification' 
- 'spatial_DC_classification'
- 'spatial_rate_classification'
- 'uniform_DC_XORXOR'
- 'spatial_DC_XORXOR'
- 'spatial_rate_XORXOR'
- 'uniform_DC_XOR'
- 'spatial_DC_XOR'
- 'spatial_rate_XOR'
- 'spatial_DC_temporal_XOR'
- 'uniform_DC_temporal_XOR'

**Remark**: In the paper, `step` is called `amplitude-value` and `uniform` is called `distributed-value`

For the `spatial` encoding you have to adjust the `spatial_std_factor` ($\sigma$ in the paper), for `step` and `uniform`
you have to adjust `input_connection_probability` ($p$ in the paper).

If you want to simulate with frozen background noise, `noise_loop_duration` has to be the same value as `step_duration`
or you can just write 'step_duration' (as it is done in the example above). If you want to switch to changing 
noise, set the `noise_loop_duration` to 'null'. The other parameters can be left as they are.

#### Microcircuit
To run the microcircuit model instead of the BRN you have to change the parameters `network_type` and `network_params`
to the following values:

```yaml
network_type: microcircuit
network_params:
  N: 560
  neuron_model: iaf_cond_exp
  neuron_params_exc:
    C_m: 346.36
    E_L: -80.0
    E_ex: 0.0
    E_in: -75.0
    I_e: 0.0
    V_m: -80.0
    V_reset: -80.0
    V_th: -55.0
    g_L: 15.5862
    t_ref: 3.0
    tau_syn_ex: 3.0
    tau_syn_in: 6.0
  neuron_params_inh:
    C_m: 346.36
    E_L: -80.0
    E_ex: 0.0
    E_in: -75.0
    I_e: 0.0
    V_m: -80.0
    V_reset: -80.0
    V_th: -55.0
    g_L: 15.5862
    t_ref: 3.0
    tau_syn_ex: 3.0
    tau_syn_in: 6.0
  vt_l23exc: -52.0
  vt_l23inh: -55.0
  vt_l4exc: -49.0
  vt_l4inh: -55.0
  vt_l5exc: -57.0
  vt_l5inh: -65.0
```

## Evaluation

### Tasks

To evaluate the tasks you need to have done the simulations first to get the input values and the state matrices.
These need to be handed over as the arguments `--input` and `--states` to the script `evaluate_tasks_separately.py`.
You also have to define the task with the parameter `--task`.
Possible task names are: `XOR`, `XORXOR`, `TEMPORAL_XOR`, `NARMA5`, `DELAYED-CLASSIFICATION`
You can also change the `5` in `NARMA5` to a value of your choice. In addition, you can evaluate a delayed version of 
every task by adding a `_DELAY=` + the delay to the task name. For example `--task XOR_DELAY=13` would evaluate the XOR task
of the inputs delayed by 13 steps.

For an explanation of further parameters you can call `python evaluate_tasks_separately.py --help`:
```
usage: evaluate_task_separately.py [-h] [--input INPUT] [--states STATES] [--steps STEPS] [--task TASK] [--groupname GROUPNAME] [--runname RUNNAME] [--test_ratio TEST_RATIO] [--seed SEED] [--data_path DATA_PATH] [--trial TRIAL]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Path to input numpy array.
  --states STATES       Path to statematrix numpy array.
  --steps STEPS         Number of steps for the task.
  --task TASK           Name of the task that should be performed.
  --groupname GROUPNAME
                        Name of the group the run belongs to.
  --runname RUNNAME     Name of the current run. A folder with this name and the parameters will be created.
  --test_ratio TEST_RATIO
                        Ratio of the total number of steps to be used as test set
  --seed SEED           Seed for numpy random number generators
  --data_path DATA_PATH
                        Path to the parent directory where the group folder and run folder are created (the results will be stored in data_path/groupname/runname
  --trial TRIAL         Trial number. Can be used if you want to run multiple trials with the same parameters.

```

### Information processing capacity computation

Just as with the evaluation of the tasks, you need to run the simulations first to get the inputs and state matrix before
you can calculate the information processing capacity. Then you can run the script `run_capacity.py` with the correct
parameter values for `--input` and `--states_path`.

For other parameter descriptions you can call `python run_capacity.py --help`:


```
usage: run_capacity.py [-h] [--name NAME] [--input INPUT] [--states_path STATES_PATH] [--results_file RESULTS_FILE] [--capacity_results CAPACITY_RESULTS] [--max_degree MAX_DEGREE] [--max_delay MAX_DELAY] [--m_variables] [--m_powerlist] [--m_windowpos]
                       [--orth_factor ORTH_FACTOR] [--figures_path FIGURES_PATH] [--n_warmup N_WARMUP] [--use_scipy] [--sample_ids SAMPLE_IDS] [--sample_size SAMPLE_SIZE] [--sample_step SAMPLE_STEP] [--delskip DELSKIP] [--windowskip WINDOWSKIP]
                       [-v VERBOSITY]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           name of the current run
  --input INPUT         Path to input numpy array.
  --states_path STATES_PATH
                        Path to statematrix numpy array.
  --results_file RESULTS_FILE
                        Path to a CSV file where the results of multiple runs should be stored.
  --capacity_results CAPACITY_RESULTS
                        Path where the capacity values of this run should be stored
  --max_degree MAX_DEGREE
                        Maximum degree that should be evaluated.
  --max_delay MAX_DELAY
                        Maximum delay that should be evaluated.
  --m_variables         Whether to assume a monotonous decrease of capacity with increasing number of variables
  --m_powerlist         Whether to assume a monotonous decrease of capacity with increasing power list
  --m_windowpos         Whether to assume a monotonous decrease of capacity with increasing positions in the window
  --orth_factor ORTH_FACTOR
                        Factor that increases the cutoff value
  --figures_path FIGURES_PATH
  --n_warmup N_WARMUP   Number of warm up simulation steps.
  --use_scipy
  --sample_ids SAMPLE_IDS
                        Path to a numpy array with ids of unist that should be used for subsampling the state matrix
  --sample_size SAMPLE_SIZE
                        Number of units to use as a random subsample of the state matrix
  --sample_step SAMPLE_STEP
                        Subsample the state matrix by using only every sample_step unit.
  --delskip DELSKIP     Number of delays before a monotonous decrease of capacity values is assumed.
  --windowskip WINDOWSKIP
                        Number of windows before a monotonous decrease of capacity values is assumed.
  -v VERBOSITY, --verbosity VERBOSITY

```

## Figures
You can find the plotting scripts in the folder `plotting`. You can run the script `plot_all_figures.py` to create the 
figures for the results of the ESN, FPUT, BRN and MC experiments. The raw data for the figures can be found in the folder `plotting/data`.

## Tests
To check whether your setup works, you can run the unit tests in the folder `tests`.
