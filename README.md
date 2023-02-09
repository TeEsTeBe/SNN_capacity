# SNN_capacity

## Folder structure

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
### Information processing capacity computation

## Plotting