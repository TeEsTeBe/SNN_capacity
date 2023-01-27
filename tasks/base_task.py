from abc import ABC, abstractmethod


class BaseTask(ABC):

    input_dimension = None
    input = None
    target = None

    def __init__(self):
        pass

    @abstractmethod
    def _check_input(self, inputs):
        pass

    @abstractmethod
    def _generate_input(self):
        pass

    @abstractmethod
    def _generate_target(self):
        pass

    @abstractmethod
    def get_default_evaluator(self):
        pass

    @abstractmethod
    def get_state_filter(self):
        pass