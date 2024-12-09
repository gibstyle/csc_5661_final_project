from abc import ABC, abstractmethod
import random

class Agent(ABC):
    """
    Abstract base class for all agents.
    """

    def __init__(self, name):
        """
        """
        self.name = name

    @abstractmethod
    def choose_action(self, state):
        """
        Abstract method to be implemented by subclasses.
        Given a state, this function returns an action.
        """
        pass