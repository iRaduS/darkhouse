from abc import ABC, ABCMeta, abstractmethod

class AgentService(ABC, metaclass=ABCMeta):
    @property
    @abstractmethod
    def agent_instance(self):
        pass

    @abstractmethod
    def agent_generate_content(self, requirement: str):
        pass