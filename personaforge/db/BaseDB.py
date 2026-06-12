# BaseDB.py

from abc import ABC, abstractmethod

class BaseDB(ABC):
    
    @abstractmethod
    def init_from_data(self, data):
        pass

    @abstractmethod
    def search(self, query, n_results):
        pass

 

    