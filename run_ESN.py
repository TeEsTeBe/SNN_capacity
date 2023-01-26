#!/usr/env python

import os
import argparse

import numpy as np
import yaml

from ESN import esn

esn.test_loading()


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default=None, help='Path to input. If not set, the input is generated and stored to the run path.')
    parser.add_argument('--runname', type=str, default='defaultname', help='Name of the current run. A folder with this name and the parameters will be created.')
    parser.add_argument('--groupname', type=str, default='defaultgroup', )
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--steps', type=int, default=100000, help="Number of inputs. Only used if --input file does not exist.")
    parser.add_argument('--nodes', type=int, default=50)
    parser.add_argument('--input_scaling', type=float, default=0.5)
    parser.add_argument('--spectral_radius', type=float, default=0.95)
    parser.add_argument('--n_warmup', type=int, default=500)
    parser.add_argument('--init_normal', action='store_true')
    parser.add_argument('--orthogonalize', action='store_true')
    parser.add_argument('--ortho_density_denominator', type=int, default=1)
    parser.add_argument('--use_relu', action='store_true')
    parser.add_argument('--relu_slope', type=float, default=1)
    parser.add_argument('--relu_start', type=float, default=0)
    parser.add_argument('--use_linear_activation', action='store_true')
    parser.add_argument('--results_file', type=str, default=None)
    parser.add_argument('--figures_path', type=str, default='figures')
    # parser.add_argument('--use_scipy', action='store_true')
    parser.add_argument('--recurrence_density', type=float, default=None,
                        help="For orthogonalization use --ortho_density_denominator!")
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None)

    return parser.parse_args()


def get_full_runnname(args, steps):
    full_runname = f'{args.runname}_steps={steps}_nodes={args.nodes}_inpscaling={args.input_scaling}_specrad={args.spectral_radius}_nwarmup={args.n_warmup}'

    if args.recurrence_density is not None:
        full_runname += f'_recdens={args.recurrence_density}'
    if args.init_normal:
        full_runname += '_init-normal'
    if args.orthogonalize:
        full_runname += '_orthogonalized'
    if args.ortho_density_denominator != 1:
        full_runname += f'_orthdensdenom={args.ortho_density_denominator}'
    if args.use_relu:
        full_runname += '_relu'
    if args.use_linear_activation:
        full_runname += '_linear'

    full_runname += f'_trial={args.trial}'

    return full_runname


def main():
    args = parse_cmd()
    if args.seed is None:
        seed = np.random.randint(2**32-1)
    else:
        seed = args.seed
    np.random.seed(seed)

    # load inputs and add warmup inputs, which are not used for the capacity computation
    warmup_input = 2.0 * np.random.rand(args.n_warmup, 1) - 1.0
    if args.input is None:
        print('Input does not exist yet and will be created now')
        inputs = np.random.uniform(-1, 1, args.steps)
    else:
        inputs = np.load(args.input)
        print(f'Inputs loaded from {args.input}')
    steps = inputs.size

    full_runname = get_full_runnname(args, steps)
    runpath = os.path.join(args.data_path, args.groupname, full_runname)
    os.makedirs(runpath, exist_ok=True)
    with open(os.path.join(runpath, 'parameters.yaml'), 'w') as params_file:
        params = vars(args)
        params['seed'] = seed
        params['runpath'] = runpath
        yaml.dump(params, params_file)

    if args.input is None:
        input_path = os.path.join(runpath, 'inputs.npy')
        np.save(input_path, inputs)
        print(f'\t Input saved to {input_path}')

    inputs = inputs.reshape(steps, 1)
    inputs = np.vstack((warmup_input, inputs))

    # initialize weights
    if args.init_normal:
        I2R = esn.CM_Initialise_Normal(1, args.nodes, scale=args.input_scaling)
        R2R = esn.CM_Initialise_Normal(args.nodes, args.nodes)
    else:
        I2R = esn.CM_Initialise_Uniform(1, args.nodes, scale=args.input_scaling)
        R2R = esn.CM_Initialise_Uniform(args.nodes, args.nodes)
    if args.orthogonalize:
        if args.ortho_density_denominator > 1:
            R2R = esn.CM_Initialise_Sparse_Orthogonal(args.nodes, args.nodes,
                                                      density_denominator=args.ortho_density_denominator)
        else:
            R2R = esn.CM_Initialise_Orthogonal(args.nodes, args.nodes)

    if args.ortho_density_denominator == 1 and args.recurrence_density is not None and args.recurrence_density < 1.0:
        # if args.orthogonalize:
        #     R2R = ESN.set_abs_small_vals_to_zero(R2R, density=args.recurrence_density)
        # else:
        #     R2R = ESN.set_randomly_to_zero(R2R, density=args.recurrence_density)
        R2R = esn.set_randomly_to_zero(R2R, density=args.recurrence_density)

    # is_ortho = lambda mat: np.allclose(mat.T @ mat, np.eye(mat.shape[0]))
    # asdf = ESN.CM_Initialise_Sparse_Orthogonal(args.nodes, args.nodes, 20)
    #
    # ortho = is_ortho(asdf)
    # density = np.count_nonzero(asdf) / asdf.size

    R2R = esn.CM_scale_specrad(R2R, args.spectral_radius)
    B2R = esn.CM_Initialise_Normal(1, args.nodes, scale=0)  # no input bias is used

    # define nonlinearity
    if args.use_relu:
        nonlinearity = lambda x: esn.ReLu(x, slope=args.relu_slope, slope_start=args.relu_start)
    elif args.use_linear_activation:
        nonlinearity = esn.Lin
    else:
        nonlinearity = np.tanh

    # create and run the ESN
    reservoir = esn.ESN(I2R=I2R, B2R=B2R, R2R=R2R, nonlinearity=nonlinearity)
    states = reservoir.Batch(inputs)[args.n_warmup:, :]

    states_path = os.path.join(runpath, 'states.npy')
    np.save(states_path, states)
    print(f'ESN states stored to {states_path}')


if __name__ == '__main__':
    main()