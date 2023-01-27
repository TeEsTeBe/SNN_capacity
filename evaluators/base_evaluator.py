from abc import ABC, abstractmethod


class BaseEvaluator(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, targets, reconstruction):
        pass
