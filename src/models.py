import gc
from typing import Dict, Any
from abc import ABC, abstractmethod

from numpy import ndarray
from torch.nn import Module
from torch.cuda import empty_cache


class NotTorchModuleError(ValueError):

    def __init__(self, source_model: Any):
        err = f'Invalid type: {source_model}.'
        super().__init__(err)


class BaseModel(ABC):

    def __init__(self, source_model: Module):
        if not isinstance(source_model, Module):
            raise NotTorchModuleError(source_model)
        self._source_model = source_model

    @abstractmethod
    def predict(self, frame: ndarray) -> Dict[str, ndarray]:
        pass

    def is_cuda(self) -> bool:
        param = next(self._source_model.parameters())
        return param.device.type == 'cuda'

    def release(self):
        if self.is_cuda():
            empty_cache()
            self._source_model.cpu()
        del self._source_model
        gc.collect()

    def __str__(self) -> str:
        return self.__class__.__name__
