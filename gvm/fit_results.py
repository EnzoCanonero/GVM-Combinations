from dataclasses import dataclass
import numpy as np


@dataclass
class FitResult:
    mu: float
    thetas: np.ndarray
    nll: float

