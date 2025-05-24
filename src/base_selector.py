
from abc import ABC, abstractmethod

class BaseFeatureSelector(ABC):
    @abstractmethod
    def select(self, X, y):
        pass
