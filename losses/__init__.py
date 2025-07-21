# losses/__init__.py

from .EVD.LoRA import EVDLoRA
from .EVD.RQ   import EVDRQ
from .SVD.LoRA import SVDLoRA
from .SVD.RQ   import SVDRQ

__all__ = ["EVDLoRA", "EVDRQ", "SVDLoRA", "SVDRQ"]