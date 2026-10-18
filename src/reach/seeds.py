"""Run seeds for this project (fresh stream; EML-GP's SEEDS are used only for regression)."""
import random as _random

_master = _random.Random(20261012)
SEEDS = [_master.randint(0, 2 ** 31 - 1) for _ in range(100)]
