"""A collection of classes to help create a neural network for whether screening factors are important."""

from dataclasses import dataclass

import numpy as np

from pynucastro.screening import ScreenFactors
from .utils import DummyPlasmaState, ArrayLike, ScreeningFunction, z_to_a

__all__ = ["ScreeningFactorData"]

@np.vectorize(excluded=[0], signature="(9)->()")
def screening_factors(screen_func: ScreeningFunction, args: np.ndarray) -> float:
    """Function for computing screening factors from a set of parameters.
    
    screen_func: the screening function to use.

    Elements of args:
        log_temp: log10 of the plasma state's temperature.
        log_dens: log10 of the plasma state's density.
        abar: the average mass number of the composition.
        zbar, z2bar: the average atomic number, atomic number squared of the composition.
        z1, a1: the atomic and mass numbers of first screening nuclei.
        z2, a2: the atomic and mass numbers of second screening nuclei.
    """

    temp, dens = 10**args[:2]
    state = DummyPlasmaState(temp, dens, *args[2:5])
    scn_fac = ScreenFactors(*args[5:])

    return screen_func(state, scn_fac)

@dataclass
class ScreeningFactorData:
    """Generates training and testing data to make a neural network for a given screening function.
    
    Keyword arguments:
        `screen_func`: the screening function to use.
        `threshold`: the threshold after which a screening factor is considered important.
        `size`: the number of data points to have in the training and testing data.
        `rng`: the seed used for random number generation.
    """

    screen_func: ScreeningFunction
    threshold: float = 1.01
    size: int = 10**6
    seed: int | None = None

    def __post_init__(self) -> None:
        """Generates the training, validation, and testing data using the parameters supplied."""

        # converts the seed supplied into a numpy random Generator
        self.rng: np.random.Generator = np.random.default_rng(self.seed)

        # generates the data
        log_temp, log_dens = self.rng.uniform([7, 4], [10, 8], (3*self.size, 2)).T

        zbar = 118**(1 - self.rng.uniform(0, 1, 3*self.size))
        z2bar = zbar**2 + np.abs(self.rng.normal(0, 20*(zbar + 1)/118, 3*self.size))
        abar = z_to_a(zbar, rng=self.rng, size=3*self.size)

        z1, z2 = _z = self.rng.uniform(1, 118, size=(2, 3*self.size))
        a1, a2 = z_to_a(_z, rng=self.rng, size=(2, 3*self.size))

        # collects the data into a dictionary and an array
        self._x = {
            "log_temp": log_temp, "log_dens": log_dens,
            "abar": abar, "zbar": zbar, "z2bar": z2bar,
            "z1": z1, "a1": a1,
            "z2": z2, "a2": a2
        }
        self.x = np.column_stack(tuple(self._x.values()))

        # computes the screening factors and whether they're important
        self.f = screening_factors(self.screen_func, self.x).reshape(3*self.size, 1)
        self.y = self.screening_indicator(factors=self.f, threshold=self.threshold)

        # stores fraction of 1s in self.y
        self.frac_pos = np.count_nonzero(self.y)/(3 * self.size)

        # splits the data into training, validation, and testing data
        self.x = self.split_data(self.x)
        self.f = self.split_data(self.f)
        self.y = self.split_data(self.y)

    @staticmethod
    def split_data(data: np.ndarray) -> dict[str, np.ndarray]:
        """Splits array into train, validation, and test datasets."""
        # pylint: disable=unbalanced-tuple-unpacking

        train, validate, test = np.split(data, 3)
        return {"train": train, "validate": validate, "test": test}

    @staticmethod
    def screening_indicator(factors: ArrayLike, threshold: float) -> ArrayLike:
        """Indicator function for whether a screening factor is important.
        
        Keyword arguments:
            factors: the screening factors to check.
            threshold: the threshold over which the screening factors are relevant.
        """

        return (factors > threshold).astype(int)
