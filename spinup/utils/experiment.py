from dataclasses import dataclass, field, astuple
from typing import Tuple, List


@dataclass
class EpochResult:
    epoch: int
    loss: float
    score: float


@dataclass
class ExperimentInfo:
    env: str
    epochs: int
    hidden_layers: Tuple[int,...]

    def to_str(self):
        return f"env: '{self.env}', ep: {self.epochs}, hl: {self.hidden_layers}"

    def to_short_str(self):
        return str(astuple(self))


@dataclass
class ExperimentResult:
    info: ExperimentInfo
    epochs: List[EpochResult] = field(default_factory=list)


@dataclass
class CyclicLearningRateConfig:
    step_size: int = 10
    learning_rate: Tuple[float, float] = (1e-3, 5e-2)
    const_lr_decay: float = .5,
    max_lr_decay: float = .7
