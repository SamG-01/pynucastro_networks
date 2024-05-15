"""Some helper functions for making the network function"""

from typing import Callable

import numpy as np

from pynucastro.screening import PlasmaState, ScreenFactors


__all__ = ["ScreeningFunction"]

# type hinting support for screening functions
ScreeningFunction = Callable[[PlasmaState, ScreenFactors], float]
ArrayLike = np.ndarray[float] | float

# helper functions
def z_to_a(z: ArrayLike, rng: np.random.Generator, size: tuple) -> ArrayLike:
    """For a random value of z, returns an appropriate random value of a to go along with it.
    
    Keyword arguments:
        `z`: an `ArrayLike` object
        `rng`: the source of random number generation
        `size`: how many samples to draw from the normal distribution
    """

    return 2*z + np.sqrt(z) * rng.standard_normal(size)

def DummyPlasmaState(temp: float, dens: float, abar: float, zbar: float, z2bar: float) -> PlasmaState:
    """Creates a dummy PlasmaState object used for neural network training.
    
    Keyword arguments:
        `temp`, `dens`: the temperature and density.
        `abar`: the average mass number.
        `zbar`, `z2bar`: the average atomic number and squared atomic number.
    """

    # Creates fake Ys array for PlasmaState.__init__()
    ytot = 1/abar
    Ys = ytot * np.array([1/2, 1/2])

    # To find Zs such that sum(Zs * Ys)/ytot = zbar
    # and sum(Zs**2 * Ys)/ytot = z2bar,
    # we solve (x + y)/2 = zbar, (x^2 + y^2)/2 = z2bar for x,y
    Zs = zbar + np.sqrt(z2bar - zbar**2) * np.array([1, -1])
    state = PlasmaState(temp, dens, Ys, Zs)

    return state
