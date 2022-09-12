import os
import gc
import numpy as np
import pickle
import logging 
from time import time
from shutil import copyfile
import nest
import nest.raster_plot
import nest.random
import matplotlib.pyplot as plt

from networks import brunel, alzheimers, microcircuit
from utils import state_utils, input_utils, general_utils, connection_utils


class SimulationRunner:
    implemented_input_types = ['uniform_DC', 'uniform_rate', 'step_rate', 'step_DC', 'spatial_rate', 'spatial_DC', 'None', 'spatial_DC_classification', 'spatial_rate_classification', 'spatial_DC_XORXOR', 'spatial_DC_XOR', 'spatial_rate_XORXOR', 'spatial_rate_XOR']
    implemented_network_types = ['alzheimers', 'brunel', 'microcircuit']

    def __init__(self, group_name, run_title, network_type, input_type, step_duration, num_steps, input_min_value,
                 input_max_value, n_spatial_encoder, spatial_std_factor, input_connection_probability, network_params,
                 background_rate, background_weight, noise_loop_duration, paramfile, data_dir, dt, spike_recorder_duration,
                 raster_plot_duration, batch_steps, add_ac_current, enable_spike_filtering=False, num_threads=1):
        logging.basicConfig(filename=f'{run_title}.log', level=logging.DEBUG, format=f'{run_title} %(asctime)s %(levelname)s: %(message)s')
        self.logger = logging.getLogger(run_title)
        self.logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s \n\t\t %(levelname)s - %(message)s')

        # add formatter to ch
        console_handler.setFormatter(formatter)

        # add ch to logger
        self.logger.addHandler(console_handler)
        general_utils.print_memory_consumption('Memory usage - beginning init SimulationRunner', logger=self.logger)

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
        self.enable_spike_filtering = enable_spike_filtering
        self.background_rate = background_rate
        self.background_weight = background_weight
        self.noise_loop_duration = noise_loop_duration
        self.network.add_spiking_noise(rate=self.background_rate, weight=self.background_weight, loop_duration=self.noise_loop_duration)
        self.add_ac_current = add_ac_current
        if self.add_ac_current:
            self.network.add_ac_current_noise(loop_duration=self.noise_loop_duration)

        self.input_type = input_type
        self.step_duration = step_duration
        self.num_steps = num_steps
        if 'classification' in self.input_type:
            n_classes = 10
            class_values = np.arange(-1, 1., 2./n_classes).round(1)  # if you want to use more classes then 10 you might want to remove the round() call
            self.input_signal = np.random.choice(class_values, size=self.num_steps, replace=True)
        elif 'XORXOR' in self.input_type:
            self.input_signal = np.random.choice(np.arange(0, 16), size=self.num_steps, replace=True)  # values from 0 to 15 can be transformed into 4 zero or one values by using the binary representation
        elif 'XOR' in self.input_type:
            self.input_signal = np.random.choice(np.arange(0, 4), size=self.num_steps, replace=True)  # values from 0 to 3 can be transformed into 2 zero or one values by using the binary representation
        else:
            self.input_signal = np.random.uniform(-1, 1, size=num_steps)
        self.input_min_value = input_min_value
        self.input_max_value = input_max_value
        if n_spatial_encoder == 'n_input_neurons':
            self.n_spatial_encoder = len(general_utils.combine_nodelists(list(self.network.get_input_populations().values())))
        else:
            self.n_spatial_encoder = n_spatial_encoder
        self.spatial_std_factor = spatial_std_factor
        self.input_connection_probability = input_connection_probability
        self.batch_steps = self.num_steps if batch_steps is None else batch_steps
        self.input_generators = self._setup_input()

        self._setup_state_recording()
        self.spike_recorder_duration = spike_recorder_duration
        self.raster_plot_duration = raster_plot_duration
        self.spike_recorder = state_utils.create_spike_recorder(self.network.populations.values(), stop=self.spike_recorder_duration)

        if data_dir is None:
            data_dir = general_utils.get_default_data_dir()
        self.results_folder = os.path.join(data_dir, 'simulation_runs', self.group_name, self.run_title)
        os.makedirs(self.results_folder, exist_ok=True)
        general_utils.print_memory_consumption('Memory usage - end init SimulationRunner', logger=self.logger)

    def _create_network(self):
        general_utils.print_memory_consumption('Memory usage - beginning _create_network', logger=self.logger)
        if self.network_type == 'brunel':
            network = brunel.BrunelNetwork(**self.network_params)
        elif self.network_type == 'alzheimers':
            network = alzheimers.AlzheimersNetwork(**self.network_params)
        elif self.network_type == 'microcircuit':
            network = microcircuit.Microcircuit(**self.network_params)
        else:
            raise ValueError(f'Network type unknown: "{self.network_type}"')
        general_utils.print_memory_consumption('Memory usage - end _create_network', logger=self.logger)

        return network

    def _setup_state_recording(self):
        general_utils.print_memory_consumption('Memory usage - beginning _setup_state_recording', logger=self.logger)
        self.network.set_up_state_multimeter(interval=self.step_duration)
        if self.enable_spike_filtering:
            self.network.set_up_spike_filtering(interval=self.step_duration, filter_tau=20.)
        general_utils.print_memory_consumption('Memory usage - end _setup_state_recording', logger=self.logger)

    def _set_input_to_generators(self, start_step, stop_step):
        start_time = start_step * self.step_duration
        if 'step_' in self.input_type or 'uniform_' in self.input_type:
            input_utils.set_input_to_step_encoder(
                input_signal=self.input_signal[start_step:stop_step],
                encoding_generator=self.input_generators,
                step_duration=self.step_duration,
                min_value=self.input_min_value,
                max_value=self.input_max_value,
                start=start_time
            )
        elif 'spatial_' in self.input_type:
            input_neuronlist = general_utils.combine_nodelists(list(self.network.get_input_populations().values()))
            neurons_per_device = int(len(input_neuronlist) / self.n_spatial_encoder)
            if 'XORXOR' in self.input_type:
                input_utils.set_XORXOR_input_to_gaussian_spatial_encoder(
                    input_values=self.input_signal[start_step:stop_step],
                    encoding_generator=self.input_generators,
                    step_duration=self.step_duration,
                    min_value=self.input_min_value,
                    max_value=self.input_max_value,
                    std=self.spatial_std_factor / neurons_per_device,
                    start=start_time
                )
            elif 'XOR' in self.input_type:
                input_utils.set_XOR_input_to_gaussian_spatial_encoder(
                    input_values=self.input_signal[start_step:stop_step],
                    encoding_generator=self.input_generators,
                    step_duration=self.step_duration,
                    min_value=self.input_min_value,
                    max_value=self.input_max_value,
                    std=self.spatial_std_factor / neurons_per_device,
                    start=start_time
                )
            else:
                input_utils.set_input_to_gaussian_spatial_encoder(
                    input_values=self.input_signal[start_step:stop_step],
                    encoding_generator=self.input_generators,
                    step_duration=self.step_duration,
                    min_value=self.input_min_value,
                    max_value=self.input_max_value,
                    std=self.spatial_std_factor / neurons_per_device,
                    start=start_time
                )
        else:
            raise ValueError(f'Unknown input_type: {self.input_type}')

    def _setup_input(self):
        general_utils.print_memory_consumption('Memory usage - beginning _setup_input', logger=self.logger)
        input_generators = None
        input_neuronlist = general_utils.combine_nodelists(list(self.network.get_input_populations().values()))

        input_weight = 1. if '_DC' in self.input_type else self.network.input_weight
        if '_rate' in self.input_type and self.network_type == 'microcircuit':
            scaling_factor_s1 = 14.85
            input_weight = connection_utils.calc_synaptic_weight(input_weight, scaling_factor_s1,'exc', self.network.neuron_params_exc['g_L'])

        if 'uniform_' in self.input_type:
            input_weight = nest.random.uniform(min=-input_weight, max=input_weight)

        if 'step_' in self.input_type or 'uniform_' in self.input_type:
            if 'rate' in self.input_type:
                step_enc_generator = nest.Create('inhomogeneous_poisson_generator')
            elif 'DC' in self.input_type:
                step_enc_generator = nest.Create('step_current_generator')

            # nest.Connect(step_enc_generator, self.network.populations['E'],
            #              conn_spec={'rule': 'pairwise_bernoulli', 'p': self.input_connection_probability},
            #              syn_spec={'weight': self.network.input_weight})
            # nest.Connect(step_enc_generator, self.network.populations['I'],
            #              conn_spec={'rule': 'pairwise_bernoulli', 'p': self.input_connection_probability},
            #              syn_spec={'weight': self.network.input_weight})
            nest.Connect(step_enc_generator, input_neuronlist,
                         conn_spec={'rule': 'pairwise_bernoulli', 'p': self.input_connection_probability},
                         syn_spec = {'weight': input_weight})

            input_generators = step_enc_generator

        elif 'spatial_' in self.input_type:
            neurons_per_device = int(len(input_neuronlist) / self.n_spatial_encoder)

            if 'spatial_rate' in self.input_type:
                spatial_enc_devices = nest.Create('inhomogeneous_poisson_generator', n=self.n_spatial_encoder)
            elif 'spatial_DC' in self.input_type:
                spatial_enc_devices = nest.Create('step_current_generator', n=self.n_spatial_encoder)

            for i, generator in enumerate(spatial_enc_devices):
                start = i * neurons_per_device
                end = start + neurons_per_device
                # nest.Connect(generator, input_neuronlist[start:end], 'all_to_all', {'weight': 1.})
                nest.Connect(generator, input_neuronlist[start:end], 'all_to_all', {'weight': input_weight})

            input_generators = spatial_enc_devices

        general_utils.print_memory_consumption('Memory usage - end _setup_input', logger=self.logger)

        return input_generators

    def _save_states(self):
        general_utils.print_memory_consumption('\n\nMemory usage - beginning _save_states', logger=self.logger)

        self.logger.info('copying parameters file ...')
        copyfile(self.paramfile, os.path.join(self.results_folder, 'parameters.yaml'))

        self.logger.info('saving input signal file ...')
        general_utils.print_memory_consumption('\tMemory usage - before saving input_signal', logger=self.logger)
        np.save(os.path.join(self.results_folder, 'input_signal.npy'), self.input_signal)

        self.logger.info('saving Vm state matrix ...')
        general_utils.print_memory_consumption('\tMemory usage - before saving vm_statemat', logger=self.logger)
        statemat_vm = self.network.get_statematrix()
        np.save(os.path.join(self.results_folder, 'vm_statemat.npy'), statemat_vm)
        del statemat_vm
        gc.collect()

        if self.enable_spike_filtering:
            self.logger.info('saving filtered spikes state matrix ....')
            general_utils.print_memory_consumption('\tMemory usage - before saving filter_statemat', logger=self.logger)
            statemat_filter = self.network.get_filter_statematrix()
            np.save(os.path.join(self.results_folder, 'filter_statemat.npy'), statemat_filter)
            del statemat_filter
            gc.collect()

        self.logger.info('saving spikes ...')
        general_utils.print_memory_consumption('\tMemory usage - before saving spike_events', logger=self.logger)
        spike_events = self.spike_recorder.get('events')
        with open(os.path.join(self.results_folder, 'spike_events.pkl'), 'wb') as spikes_file:
            pickle.dump(spike_events, spikes_file)
        del spike_events
        gc.collect()

        general_utils.print_memory_consumption('Memory usage - end _save_states', logger=self.logger)

    def _create_plots(self):
        general_utils.print_memory_consumption('\n\nMemory usage - beginning _create_plots', logger=self.logger)
        self.logger.info('creating raster plot ...')

        rasterplot_spikelist = general_utils.spikelist_from_recorder(self.spike_recorder, stop=self.raster_plot_duration)

        if len(rasterplot_spikelist.id_list) > 0:
            ax1 = plt.subplot2grid((30, 1), (0, 0), rowspan=23, colspan=1)
            ax2 = plt.subplot2grid((30, 1), (24, 0), rowspan=5, colspan=1, sharex=ax1)

            rasterplot_spikelist.raster_plot(with_rate=True, ax=[ax1, ax2], display=False, markersize=3)
            ax1.tick_params(axis='x', which='both', labelbottom=False)
            ax2.set(xlabel='Time [ms]', ylabel='Rate')
            ax1.set(ylabel='Neuron')
            plt.title('')
            raster_plot_path = os.path.join(self.results_folder, 'raster_plot.pdf')
            plt.savefig(raster_plot_path)
        else:
            logging.info("No spikes recorded and therefore no raster plot created")
        del rasterplot_spikelist
        gc.collect()

        self.logger.info('creating Vm state matrix plot ...')
        general_utils.print_memory_consumption('\tMemory usage - before vm state matrix plot', logger=self.logger)
        plt.clf()
        fig = plt.figure(figsize=(12, 9))
        statemat_vm_slice = self.network.get_statematrix()[:, :100].copy()
        im = plt.matshow(statemat_vm_slice, fignum=fig.number, aspect='auto')
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

        if self.enable_spike_filtering:
            self.logger.info('creating filtered spikes state matrix plot ...')
            general_utils.print_memory_consumption('\tMemory usage - before filter state matrix plot', logger=self.logger)
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

        self.logger.info('creating spike statistics plot ...')
        general_utils.print_memory_consumption('\tMemory usage - before statistics plot', logger=self.logger)
        statistic_spikelist = general_utils.spikelist_from_recorder(self.spike_recorder)
        if len(statistic_spikelist.id_list) > 0:
            hist_data = {
                'CV': statistic_spikelist.cv_isi(),
                'CC': statistic_spikelist.pairwise_pearson_corrcoeff(1000, time_bin=2., all_coef=True),
                'rates': statistic_spikelist.mean_rates(),
            }

            fig, axes = plt.subplots(ncols=len(hist_data.keys()), figsize=(15, 5))
            for i, (name, data) in enumerate(hist_data.items()):
                try:
                    axes[i].hist(data)
                    axes[i].set_title(name)
                    np.save(os.path.join(self.results_folder, f'{name}.npy'), data)
                    print(f'average {name}: {np.nanmean(data)}')
                except:
                    print(f'{name} hist failed')

            statistics_plot_path = os.path.join(self.results_folder, 'statistics_plot.pdf')
            plt.savefig(statistics_plot_path)
            general_utils.print_memory_consumption('Memory usage - end _create_plots', logger=self.logger)
        del statistic_spikelist
        gc.collect()

    def run(self):
        time_to_simulate = self.num_steps * self.step_duration
        start_real_time = time()
        start_step = 0
        while start_step < self.num_steps:
            batch_start_real_time = time()
            batch_steps = min(self.batch_steps, self.num_steps - start_step)

            self._set_input_to_generators(start_step=start_step, stop_step=start_step+batch_steps)
            general_utils.print_memory_consumption('Memory usage - before simulation', logger=self.logger)
            nest.Simulate(batch_steps*self.step_duration)
            general_utils.print_memory_consumption('Memory usage - after simulation batch', logger=self.logger)
            gc.collect()
            start_step += self.batch_steps

            batch_end_real_time = time()
            simulated_bio_time = start_step*self.step_duration
            percentage_done = 100.*simulated_bio_time/time_to_simulate
            self.logger.info(f'\nSimulated {simulated_bio_time} of {time_to_simulate} ms ({round(percentage_done, 4)}%)')
            simulated_real_time = batch_end_real_time - start_real_time
            self.logger.info(f'Simulated real time: {simulated_real_time} sec ({round(simulated_real_time/60, 2)} min, {round(simulated_real_time/3600,4)} h)')
            time_still_needed = simulated_real_time / (percentage_done/100.) - simulated_real_time
            self.logger.info(f'\tApproximate simulation time until finished: {time_still_needed} sec ({round(time_still_needed/60, 2)} min, {round(time_still_needed/3600,4)} h)')

            batch_duration = batch_end_real_time - batch_start_real_time
            self.logger.info(f'\tBatch took {batch_duration} sec ({round(batch_duration/60, 2)} min, {round(batch_duration/3600,4)}h)\n\n')
        nest.Simulate(self.dt)

        self._save_states()
        self._create_plots()
