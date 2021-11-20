import os
import gc
import numpy as np
import pickle
from shutil import copyfile
import nest
import nest.raster_plot
import matplotlib.pyplot as plt

from networks import brunel, alzheimers
from utils import state_utils, input_utils, general_utils


class SimulationRunner:
    implemented_input_types = ['step_rate', 'step_DC', 'spatial_rate', 'spatial_DC', 'None']
    implemented_network_types = ['alzheimers', 'brunel']

    def __init__(self, group_name, run_title, network_type, input_type, step_duration, num_steps, input_min_value,
                 input_max_value, n_spatial_encoder, spatial_std_factor, input_connection_probability, network_params,
                 background_rate, background_weight, noise_loop_duration, paramfile, data_dir, dt, spike_recorder_duration,
                 raster_plot_duration, num_threads=1):

        general_utils.print_memory_consumption('Memory usage - beginning init SimulationRunner')

        assert input_type in self.implemented_input_types,  f'Unknown input type "{input_type}"'
        assert network_type in self.implemented_network_types, f'Unknown network type"{network_type}"'

        self.num_threads = num_threads
        self.dt = dt
        nest.SetKernelStatus({"local_num_threads": self.num_threads, "resolution": self.dt, 'print_time': False})

        self.paramfile = paramfile

        self.group_name = group_name
        self.run_title = run_title

        self.network_type = network_type
        self.network_params = network_params
        self.network = self._create_network()
        self.background_rate = background_rate
        self.background_weight = background_weight
        self.noise_loop_duration = noise_loop_duration
        self.network.add_spiking_noise(rate=self.background_rate, weight=self.background_weight, loop_duration=self.noise_loop_duration)

        self.input_type = input_type
        self.step_duration = step_duration
        self.num_steps = num_steps
        self.input_signal = np.random.uniform(-1, 1, size=num_steps)
        self.input_min_value = input_min_value
        self.input_max_value = input_max_value
        self.n_spatial_encoder = n_spatial_encoder
        self.spatial_std_factor = spatial_std_factor
        self.input_connection_probability = input_connection_probability
        self.input_generators = self._setup_input()

        self._setup_state_recording()
        self.spike_recorder_duration = spike_recorder_duration
        self.raster_plot_duration = raster_plot_duration
        self.spike_recorder = state_utils.create_spike_recorder(self.network.populations.values(), stop=self.spike_recorder_duration)

        if data_dir is None:
            data_dir = general_utils.get_default_data_dir()
        self.results_folder = os.path.join(data_dir, 'simulation_runs', self.group_name, self.run_title)
        os.makedirs(self.results_folder, exist_ok=True)
        general_utils.print_memory_consumption('Memory usage - end init SimulationRunner')

    def _create_network(self):
        general_utils.print_memory_consumption('Memory usage - beginning _create_network')
        if self.network_type == 'brunel':
            network = brunel.BrunelNetwork(**self.network_params)
        elif self.network_type == 'alzheimers':
            network = alzheimers.AlzheimersNetwork(**self.network_params)
        else:
            raise ValueError(f'Network type unknown: "{self.network_type}"')
        general_utils.print_memory_consumption('Memory usage - end _create_network')

        return network

    def _setup_state_recording(self):
        general_utils.print_memory_consumption('Memory usage - beginning _setup_state_recording')
        self.network.set_up_state_multimeter(interval=self.step_duration)
        self.network.set_up_spike_filtering(interval=self.step_duration, filter_tau=20.)
        general_utils.print_memory_consumption('Memory usage - end _setup_state_recording')

    def _setup_input(self):
        general_utils.print_memory_consumption('Memory usage - beginning _setup_input')
        input_generators = None
        if 'step_' in self.input_type:
            if self.input_type == 'step_rate':
                step_enc_generator = input_utils.get_step_encoding_device('inhomogeneous_poisson_generator',
                                                                          input_signal=self.input_signal,
                                                                          step_duration=self.step_duration,
                                                                          min_value=self.input_min_value,
                                                                          max_value=self.input_max_value)
            elif self.input_type == 'step_DC':
                step_enc_generator = input_utils.get_step_encoding_device('step_current_generator',
                                                                          input_signal=self.input_signal,
                                                                          step_duration=self.step_duration,
                                                                          min_value=self.input_min_value,
                                                                          max_value=self.input_max_value)
            nest.Connect(step_enc_generator, self.network.populations['E'],
                         conn_spec={'rule': 'pairwise_bernoulli', 'p': self.input_connection_probability},
                         syn_spec={'weight': self.network.J})
            nest.Connect(step_enc_generator, self.network.populations['I'],
                         conn_spec={'rule': 'pairwise_bernoulli', 'p': self.input_connection_probability},
                         syn_spec={'weight': self.network.J})
            input_generators = step_enc_generator

        elif 'spatial_' in self.input_type:
            input_neuronlist = general_utils.combine_nodelists(list(self.network.get_input_populations().values()))
            neurons_per_device = int(len(input_neuronlist) / self.n_spatial_encoder)

            if self.input_type == 'spatial_rate':
                spatial_enc_devices = input_utils.get_gaussian_spatial_encoding_device(
                    'inhomogeneous_poisson_generator', input_values=self.input_signal, num_devices=self.n_spatial_encoder,
                    step_duration=self.step_duration, min_value=self.input_min_value, max_value=self.input_max_value,
                    std=self.spatial_std_factor / neurons_per_device)
            elif self.input_type == 'spatial_DC':
                spatial_enc_devices = input_utils.get_gaussian_spatial_encoding_device(
                    'step_current_generator', input_values=self.input_signal, num_devices=self.n_spatial_encoder,
                    step_duration=self.step_duration, min_value=self.input_min_value, max_value=self.input_max_value,
                    std=self.spatial_std_factor / neurons_per_device)

            for i, generator in enumerate(spatial_enc_devices):
                start = i * neurons_per_device
                end = start + neurons_per_device
                nest.Connect(generator, input_neuronlist[start:end], 'all_to_all', {'weight': self.network.J})

            input_generators = spatial_enc_devices
        general_utils.print_memory_consumption('Memory usage - end _setup_input')

        return input_generators

    def _save_states(self):
        general_utils.print_memory_consumption('\n\nMemory usage - beginning _save_states')

        print('copying parameters file ...')
        copyfile(self.paramfile, os.path.join(self.results_folder, 'parameters.yaml'))

        print('saving input signal file ...')
        general_utils.print_memory_consumption('\tMemory usage - before saving input_signal')
        np.save(os.path.join(self.results_folder, 'input_signal.npy'), self.input_signal)

        print('saving Vm state matrix ...')
        general_utils.print_memory_consumption('\tMemory usage - before saving vm_statemat')
        statemat_vm = self.network.get_statematrix()
        np.save(os.path.join(self.results_folder, 'vm_statemat.npy'), statemat_vm)
        del statemat_vm
        gc.collect()

        print('saving filtered spikes state matrix ....')
        general_utils.print_memory_consumption('\tMemory usage - before saving filter_statemat')
        statemat_filter = self.network.get_filter_statematrix()
        np.save(os.path.join(self.results_folder, 'filter_statemat.npy'), statemat_filter)
        del statemat_filter
        gc.collect()

        print('saving spikes ...')
        general_utils.print_memory_consumption('\tMemory usage - before saving spike_events')
        spike_events = self.spike_recorder.get('events')
        with open(os.path.join(self.results_folder, 'spike_events.pkl'), 'wb') as spikes_file:
            pickle.dump(spike_events, spikes_file)
        del spike_events
        gc.collect()

        general_utils.print_memory_consumption('Memory usage - end _save_states')

    def _create_plots(self):
        general_utils.print_memory_consumption('\n\nMemory usage - beginning _create_plots')
        print('creating raster plot ...')
        rasterplot_spikelist = general_utils.spikelist_from_recorder(self.spike_recorder, stop=self.raster_plot_duration)

        ax1 = plt.subplot2grid((30, 1), (0, 0), rowspan=23, colspan=1)
        ax2 = plt.subplot2grid((30, 1), (24, 0), rowspan=5, colspan=1, sharex=ax1)

        rasterplot_spikelist.raster_plot(with_rate=True, ax=[ax1, ax2], display=False, markersize=3)
        ax1.tick_params(axis='x', which='both', labelbottom=False)
        ax2.set(xlabel='Time [ms]', ylabel='Rate')
        ax1.set(ylabel='Neuron')
        plt.title('')
        raster_plot_path = os.path.join(self.results_folder, 'raster_plot.pdf')
        plt.savefig(raster_plot_path)
        del rasterplot_spikelist
        gc.collect()

        print('creating Vm state matrix plot ...')
        general_utils.print_memory_consumption('\tMemory usage - before vm state matrix plot')
        plt.clf()
        fig = plt.figure(figsize=(12, 9))
        statemat_vm = self.network.get_statematrix()
        im = plt.matshow(statemat_vm[:, :100], fignum=fig.number, aspect='auto')
        plt.ylabel('neuron id')
        plt.xlabel('steps')
        plt.colorbar(im, label='neuron V_m')
        plt.title('Vm states')
        state_plot_path = os.path.join(self.results_folder, 'state_plot.pdf')
        plt.savefig(state_plot_path)
        plt.clf()
        del im
        del statemat_vm_slice
        gc.collect()

        print('creating filtered spikes state matrix plot ...')
        general_utils.print_memory_consumption('\tMemory usage - before filter state matrix plot')
        plt.clf()
        fig = plt.figure(figsize=(12, 9))
        statemat_filter_slice = self.network.get_filter_statematrix()[:, :100].copy()
        im = plt.matshow(statemat_filter_slice, fignum=fig.number, aspect='auto')
        plt.ylabel('neuron id')
        plt.xlabel('steps')
        plt.colorbar(im, label='filter neuron V_m')
        plt.title('filtered spikes')
        filtered_states_plot_path = os.path.join(self.results_folder, 'filtered_states_plot.pdf')
        plt.savefig(filtered_states_plot_path)
        del im
        del statemat_filter_slice
        plt.clf()
        gc.collect()

        print('creating spike statistics plot ...')
        general_utils.print_memory_consumption('\tMemory usage - before statistics plot')
        statistic_spikelist = general_utils.spikelist_from_recorder(self.spike_recorder)
        hist_data = {
            'CV': statistic_spikelist.cv_isi(),
            'CC': statistic_spikelist.pairwise_pearson_corrcoeff(1000, time_bin=2., all_coef=True),
            'rates': statistic_spikelist.mean_rates(),
        }
        del statistic_spikelist
        gc.collect()

        fig, axes = plt.subplots(ncols=len(hist_data.keys()), figsize=(15, 5))
        for i, (name, data) in enumerate(hist_data.items()):
            axes[i].hist(data)
            axes[i].set_title(name)

        statistics_plot_path = os.path.join(self.results_folder, 'statistics_plot.pdf')
        plt.savefig(statistics_plot_path)
        general_utils.print_memory_consumption('Memory usage - end _create_plots')

    def run(self):
        general_utils.print_memory_consumption('Memory usage - before simulation')
        nest.Simulate(self.num_steps*self.step_duration + self.dt)
        general_utils.print_memory_consumption('Memory usage - after simulation')
        gc.collect()
        general_utils.print_memory_consumption('Memory usage - after simulation and garbage collection.')
        self._save_states()
        self._create_plots()
