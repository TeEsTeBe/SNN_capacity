import numpy as np
import nest
import scipy.stats


def add_poisson_noise(population_list, rate, weight):
    poisson_generator = nest.Create('poisson_generator', {'rate': rate})
    for pop in population_list:
        nest.Connect(poisson_generator, pop, 'all_to_all', {'weight': weight})

    return poisson_generator


def add_repeating_noise(population_list, rate, weight, loop_duration, connect_to_populations=True):
    poisson_generator = nest.Create('poisson_generator', {'rate': rate, 'stop': loop_duration})
    all_parrots = nest.NodeCollection()
    for pop in population_list:
        parrots = nest.Create('parrot_neuron', n=len(pop))
        nest.Connect(poisson_generator, parrots, 'all_to_all')
        nest.Connect(parrots, parrots, 'one_to_one', {'delay': loop_duration})
        if connect_to_populations:
            nest.Connect(parrots, pop, 'one_to_one', {'weight': weight})
        all_parrots += parrots

    return poisson_generator, all_parrots


def get_rate_encoding_generator(input_values, step_duration, min_rate, max_rate):
    rates = np.interp(input_values, (-1, 1), (min_rate, max_rate))
    times = np.arange(0.1, step_duration * len(input_values) + 0.1, step_duration)
    generator = nest.Create('inhomogeneous_poisson_generator')
    nest.SetStatus(generator, {'rate_times': times, 'rate_values': rates})

    return generator


def set_input_to_step_encoder(input_signal, encoding_generator, step_duration, min_value, max_value, start=0.):
    values = np.interp(input_signal, (-1, 1), (min_value, max_value))
    times = start + np.arange(0.1, step_duration * len(input_signal) + 0.1, step_duration)

    if encoding_generator.model == 'inhomogeneous_poisson_generator':
        nest.SetStatus(encoding_generator, {'rate_times': times, 'rate_values': values})
    elif encoding_generator.model == 'step_current_generator':
        nest.SetStatus(encoding_generator, {'amplitude_times': times, 'amplitude_values': values})
    else:
        raise ValueError(f'device_type "{encoding_generator.model}" is not supported.')


def _get_XOR_positions(binary_string):
    interpolated_XOR_values = np.array([0.75*int(binary_string[0]), 0.25*int(binary_string[1])])
    interpolated_XOR_values[interpolated_XOR_values==0] = np.nan

    return interpolated_XOR_values


def set_XOR_input_to_gaussian_spatial_encoder(input_values, encoding_generator, step_duration, min_value, max_value, std, start=0.):

    # convert input values (0 ... 15) to binary strings ('0000' ... '1111')
    input_binary_strings = [f'{iv:b}'.rjust(2,'0') for iv in input_values]

    # convert binary strings ('0000' ... '1111') to 2D ndarray of positions
    # we have only the possible positions -0.75, -0.25, 0.25 and 0.75 for the 4 different input streams of the XORXOR task
    # inactive input streams (0 instead of 1) become np.nan
    array2d_input_positions = np.array([_get_XOR_positions(ibs) for ibs in input_binary_strings])

    # interpolated_input_values = np.interp(input_values, (-1, 1), (0, len(encoding_generator)))
    xvals = np.arange(0, len(encoding_generator)).reshape(len(encoding_generator), 1)

    final_values = np.zeros([len(encoding_generator), len(input_values)])
    for input_dim_index in range(array2d_input_positions.shape[1]):
        values = np.zeros([len(encoding_generator), len(input_values)])
        interpolated_input_values = np.interp(array2d_input_positions[:, input_dim_index], (-1, 1), (0, len(encoding_generator)))

        # we set the inactive inputs (0) to a small value that is far out of the range of input neurons
        # then we can calculate the pdf in the normal way, without changing the values for the deactivated inputs
        interpolated_input_values[np.isnan(interpolated_input_values)] = -999999

        values[:] += scipy.stats.norm.pdf(xvals, interpolated_input_values[:], std)
        values /= values.max()
        values *= (max_value - min_value)
        values += min_value
        final_values[:] += values[:]

    times = start + np.arange(0.1, step_duration * len(input_values) + 0.1, step_duration)

    if encoding_generator.model[0] == 'inhomogeneous_poisson_generator':
        nest.SetStatus(encoding_generator, [{'rate_times': times, 'rate_values': vals} for vals in final_values])
    elif encoding_generator.model[0] == 'step_current_generator':
        nest.SetStatus(encoding_generator, [{'amplitude_times': times, 'amplitude_values': vals} for vals in final_values])
    else:
        raise ValueError(f'device_type "{encoding_generator}" is not supported.')


def _get_XORXOR_positions(binary_string):
    interpolated_XORXOR_values = np.array([0.75*int(binary_string[0]), 0.25*int(binary_string[1]), -0.25*int(binary_string[2]), -0.75*int(binary_string[3])])
    interpolated_XORXOR_values[interpolated_XORXOR_values==0] = np.nan

    return interpolated_XORXOR_values


def set_XORXOR_input_to_gaussian_spatial_encoder(input_values, encoding_generator, step_duration, min_value, max_value, std, start=0.):

    # convert input values (0 ... 15) to binary strings ('0000' ... '1111')
    input_binary_strings = [f'{iv:b}'.rjust(4,'0') for iv in input_values]

    # convert binary strings ('0000' ... '1111') to 2D ndarray of positions
    # we have only the possible positions -0.75, -0.25, 0.25 and 0.75 for the 4 different input streams of the XORXOR task
    # inactive input streams (0 instead of 1) become np.nan
    array2d_input_positions = np.array([_get_XORXOR_positions(ibs) for ibs in input_binary_strings])

    # interpolated_input_values = np.interp(input_values, (-1, 1), (0, len(encoding_generator)))
    xvals = np.arange(0, len(encoding_generator)).reshape(len(encoding_generator), 1)

    final_values = np.zeros([len(encoding_generator), len(input_values)])
    for input_dim_index in range(array2d_input_positions.shape[1]):
        values = np.zeros([len(encoding_generator), len(input_values)])
        interpolated_input_values = np.interp(array2d_input_positions[:, input_dim_index], (-1, 1), (0, len(encoding_generator)))

        # we set the inactive inputs (0) to a small value that is far out of the range of input neurons
        # then we can calculate the pdf in the normal way, without changing the values for the deactivated inputs
        interpolated_input_values[np.isnan(interpolated_input_values)] = -999999

        values[:] += scipy.stats.norm.pdf(xvals, interpolated_input_values[:], std)
        if values.max() > 0:
            values /= values.max()
            values *= (max_value - min_value)
            values += min_value
            final_values[:] += values[:]

    times = start + np.arange(0.1, step_duration * len(input_values) + 0.1, step_duration)

    if encoding_generator.model[0] == 'inhomogeneous_poisson_generator':
        nest.SetStatus(encoding_generator, [{'rate_times': times, 'rate_values': vals} for vals in final_values])
    elif encoding_generator.model[0] == 'step_current_generator':
        nest.SetStatus(encoding_generator, [{'amplitude_times': times, 'amplitude_values': vals} for vals in final_values])
    else:
        raise ValueError(f'device_type "{encoding_generator}" is not supported.')


def set_input_to_gaussian_spatial_encoder(input_values, encoding_generator, step_duration, min_value, max_value, std, start=0.):

    interpolated_input_values = np.interp(input_values, (-1, 1), (0, len(encoding_generator)))
    xvals = np.arange(0, len(encoding_generator)).reshape(len(encoding_generator), 1)

    values = np.zeros([len(encoding_generator), len(input_values)])
    values[:] = scipy.stats.norm.pdf(xvals, interpolated_input_values[:], std)
    values /= values.max()
    values *= (max_value - min_value)
    values += min_value

    times = start + np.arange(0.1, step_duration * len(input_values) + 0.1, step_duration)

    if encoding_generator.model[0] == 'inhomogeneous_poisson_generator':
        nest.SetStatus(encoding_generator, [{'rate_times': times, 'rate_values': vals} for vals in values])
    elif encoding_generator.model[0] == 'step_current_generator':
        nest.SetStatus(encoding_generator, [{'amplitude_times': times, 'amplitude_values':vals} for vals in values])
    else:
        raise ValueError(f'device_type "{encoding_generator}" is not supported.')
