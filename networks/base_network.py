from abc import ABC, abstractmethod

from utils import state_utils


class BaseNetwork(ABC):
    populations = None
    state_multimeter = None
    filter_state_multimeter = None


    @abstractmethod
    def connect_net(self):
        pass

    @abstractmethod
    def add_spiking_noise(self):
        pass

    @abstractmethod
    def get_state_populations(self):
        pass

    @abstractmethod
    def get_input_populations(self):
        pass

    def set_up_state_multimeter(self, interval):
        self.state_multimeter = state_utils.create_multimeter(self.get_state_populations().values(), interval=interval)

    def set_up_spike_filtering(self, filter_tau, interval):
        self.filter_state_multimeter = state_utils.create_spike_filtering_multimeter(self.get_state_populations().values(), interval=interval, filter_tau=filter_tau)

    def get_statematrix(self):
        if self.state_multimeter is None:
            raise ValueError('You have to call set_up_state_multimeter before you can get the state matrix.')

        return state_utils.get_statematrix(self.state_multimeter)

    def get_filter_statematrix(self):
        if self.filter_state_multimeter is None:
            raise ValueError('You have to call set_up_spike_filtering before you can get the state matrix of the filtered spikes.')

        return state_utils.get_statematrix(self.filter_state_multimeter)


