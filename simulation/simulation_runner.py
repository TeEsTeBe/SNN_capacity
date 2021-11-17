import os
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
                 paramfile):

        assert input_type in self.implemented_input_types,  f'Unknown input type "{input_type}"'
        assert network_type in self.implemented_network_types, f'Unknown network type"{network_type}"'

        self.paramfile = paramfile

        self.group_name = group_name
        self.run_title = run_title

        self.network_type = network_type
        self.network_params = network_params
        self.network = self._create_network()
        self.network.add_default_noise()

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
        self.spike_recorder = state_utils.create_spike_recorder(self.network.populations.values())

        self.results_folder = os.path.join(general_utils.get_data_dir(), 'simulation_runs', self.group_name, self.run_title)
        os.makedirs(self.results_folder, exist_ok=True)

    def _create_network(self):
        if self.network_type == 'brunel':
            network = brunel.BrunelNetwork(**self.network_params)
        elif self.network_type == 'alzheimers':
            network = alzheimers.AlzheimersNetwork(**self.network_params)
        else:
            raise ValueError(f'Network type unknown: "{self.network_type}"')

        return network

    def _setup_state_recording(self):
        self.network.set_up_state_multimeter(interval=self.step_duration)
        self.network.set_up_spike_filtering(interval=self.step_duration, filter_tau=20.)

    def _setup_input(self):
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

        return input_generators

    def _save_states(self):
        copyfile(self.paramfile, os.path.join(self.results_folder, 'parameters.yaml'))
        np.save(os.path.join(self.results_folder, 'statemat.npy'), self.network.get_statematrix())
        np.save(os.path.join(self.results_folder, 'filtered_statemat.npy'), self.network.get_filter_statematrix())

        spike_events = self.spike_recorder.get('events')
        with open(os.path.join(self.results_folder, 'spike_events.pkl'), 'wb') as spikes_file:
            pickle.dump(spike_events, spikes_file)

    def _create_plots(self):
        spikelist = general_utils.spikelist_from_recorder(self.spike_recorder)
        hist_data = {
            'CV': spikelist.cv_isi(),
            'CC': spikelist.pairwise_pearson_corrcoeff(1000, time_bin=2., all_coef=True),
            'rates': spikelist.mean_rates(),
        }

        fig, axes = plt.subplots(ncols=len(hist_data.keys()), figsize=(15, 5))
        for i, (name, data) in enumerate(hist_data.items()):
            axes[i].hist(data)
            axes[i].set_title(name)

        statistics_plot_path = os.path.join(self.results_folder, 'statistics_plot.pdf')
        plt.savefig(statistics_plot_path)

        ax1 = plt.subplot2grid((30, 1), (0, 0), rowspan=23, colspan=1)
        ax2 = plt.subplot2grid((30, 1), (24, 0), rowspan=5, colspan=1, sharex=ax1)

        spikelist.raster_plot(with_rate=True, ax=[ax1, ax2], display=False, markersize=3)
        ax1.tick_params(axis='x', which='both', labelbottom=False)
        ax2.set(xlabel='Time [ms]', ylabel='Rate')
        ax1.set(ylabel='Neuron')
        plt.title('')
        raster_plot_path = os.path.join(self.results_folder, 'raster_plot.pdf')
        plt.savefig(raster_plot_path)

        im = plt.matshow(self.network.get_statematrix(), aspect='auto')
        plt.ylabel('neuron id')
        plt.xlabel('time [ms]')
        plt.colorbar(im, label='neuron V_m')
        plt.title('Vm states')
        state_plot_path = os.path.join(self.results_folder, 'state_plot.pdf')
        plt.savefig(state_plot_path)

        im = plt.matshow(self.network.get_filter_statematrix(), aspect='auto')
        plt.ylabel('neuron id')
        plt.xlabel('time [ms]')
        plt.colorbar(im, label='filter neuron V_m')
        plt.title('filtered spikes')
        filtered_states_plot_path = os.path.join(self.results_folder, 'filtered_states_plot.pdf')
        plt.savefig(filtered_states_plot_path)

    def run(self):
        nest.Simulate(self.num_steps*self.step_duration + 0.1)
        self._save_states()
        self._create_plots()
