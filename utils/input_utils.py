import numpy as np
import nest
import scipy.stats


def add_poisson_noise(population_list, rate, weight):
    poisson_generator = nest.Create('poisson_generator', {'rate': rate})
    for pop in population_list:
        nest.Connect(poisson_generator, pop, 'all_to_all', {'weight': weight})

    return poisson_generator


def add_repeating_noise(population_list, rate, weight, loop_duration):
    poisson_generator = nest.Create('poisson_generator', {'rate': rate, 'stop': loop_duration})
    all_parrots = nest.NodeCollection()
    for pop in population_list:
        parrots = nest.Create('parrot_neuron', n=len(pop))
        nest.Connect(poisson_generator, parrots, 'all_to_all')
        nest.Connect(parrots, parrots, 'one_to_one', {'delay': loop_duration})
        nest.Connect(parrots, pop, 'one_to_one', {'weight': weight})
        all_parrots += parrots

    return poisson_generator, all_parrots


def get_rate_encoding_generator(input_values, step_duration, min_rate, max_rate):
    rates = np.interp(input_values, (-1, 1), (min_rate, max_rate))
    times = np.arange(0.1, step_duration * len(input_values) + 0.1, step_duration)
    generator = nest.Create('inhomogeneous_poisson_generator')
    nest.SetStatus(generator, {'rate_times': times, 'rate_values': rates})

    return generator


def get_step_encoding_device(device_type, input_signal, step_duration, min_value, max_value):

    values = np.interp(input_signal, (-1, 1), (min_value, max_value))
    times = np.arange(0.1, step_duration * len(input_signal) + 0.1, step_duration)

    generator = nest.Create(device_type)

    if device_type == 'inhomogeneous_poisson_generator':
        nest.SetStatus(generator, {'rate_times': times, 'rate_values': values})
    elif device_type == 'step_current_generator':
        nest.SetStatus(generator, {'amplitude_times': times, 'amplitude_values': values})
    else:
        raise ValueError(f'device_type "{device_type}" is not supported.')

    return generator


def get_gaussian_spatial_encoding_device(device_type, input_values, num_devices, step_duration, min_value, max_value, std):
    interpolated_input_values = np.interp(input_values, (-1, 1), (0, num_devices))
    xvals = np.arange(0, num_devices).reshape(num_devices, 1)

    values = np.zeros([num_devices, len(input_values)])
    values[:] = scipy.stats.norm.pdf(xvals, interpolated_input_values[:], std)
    values /= values.max()
    values *= (max_value - min_value)
    values += min_value

    times = np.arange(0.1, step_duration * len(input_values) + 0.1, step_duration)

    generators = nest.Create(device_type, n=num_devices)

    if device_type == 'inhomogeneous_poisson_generator':
        nest.SetStatus(generators, [{'rate_times': times, 'rate_values': vals} for vals in values])
    elif device_type == 'step_current_generator':
        nest.SetStatus(generators, [{'amplitude_times': times, 'amplitude_values':vals} for vals in values])
    else:
        raise ValueError(f'device_type "{device_type}" is not supported.')

    return generators
