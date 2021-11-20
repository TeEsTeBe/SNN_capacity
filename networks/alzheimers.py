from networks.brunel import BrunelNetwork


class AlzheimersNetwork(BrunelNetwork):

    # default_neuron_params = {}
    neuron_model = 'iaf_psc_exp_ps'

    default_neuron_params = {
        'C_m': 250.,
        'E_L': 0.0,
        'V_reset': 0.0,
        'V_th': 15.,
        't_ref': 2.,
        'tau_syn_ex': 2.,
        'tau_syn_in': 2.,
        'tau_m': 20.,
    }

    def __init__(self, neuron_model='iaf_psc_exp_ps', neuron_params=None, N=1250, g=6., J=32.29, density=0.1, delay=1., CEE=100):
        super().__init__(neuron_model=neuron_model, neuron_params=neuron_params, N=N, g=g, J=J, density=density, delay=delay, CEE=CEE)

    def add_spiking_noise(self, rate=None, weight=None, loop_duration=None):
        if rate is None:
            rate = 2500.

        return super().add_spiking_noise(rate=rate, weight=weight, loop_duration=loop_duration)

