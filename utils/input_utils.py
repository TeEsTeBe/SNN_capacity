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


def set_input_to_step_encoder(input_signal, encoding_generator, step_duration, min_value, max_value, start=0.):
    values = np.interp(input_signal, (-1, 1), (min_value, max_value))
    times = start + np.arange(0.1, step_duration * len(input_signal) + 0.1, step_duration)

    if encoding_generator.model == 'inhomogeneous_poisson_generator':
        nest.SetStatus(encoding_generator, {'rate_times': times, 'rate_values': values})
    elif encoding_generator.model == 'step_current_generator':
        nest.SetStatus(encoding_generator, {'amplitude_times': times, 'amplitude_values': values})
    else:
        raise ValueError(f'device_type "{encoding_generator.model}" is not supported.')


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
