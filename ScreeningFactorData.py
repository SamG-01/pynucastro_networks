"""A collection of classes to help create a neural network for whether screening factors are important."""

from dataclasses import dataclass
from keras.utils import to_categorical
import numpy as np

import pynucastro as pyna
from pynucastro.screening import chugunov_2009

default_rng = np.random.default_rng()

@dataclass
class ExpLogData:
    """Stores and generates random temperature or density data in both log and linear form.
    
    data_range -- defines the bounds of the data (min: float, max: float)
    """

    data_range: tuple
    scaled: np.ndarray = None
    unscaled: np.ndarray = None
    size: int = None
    rng: np.random.Generator = None

    def __post_init__(self) -> None:
        self.log_range = np.log10(self.data_range)

    @classmethod
    def from_uniform(cls, data_range: tuple, scaled: np.ndarray = None, size: int = 200, rng: np.random.Generator = default_rng):
        """Constructs ExpLogData from an array of scaled data.
        
        Keyword arguments:
        data_range -- defines the bounds of the data (min: float, max: float)
        scaled -- the scaled data to construct the class with. If None, will randomly generate it.
        size -- the number of data points to generate the data if scaled is None
        rng -- the Generator object to use to generate the data if scaled is None
        """

        self = cls(data_range)

        if scaled is None:
            scaled = rng.uniform(size=size)
            self.rng = rng

        self.scaled = scaled
        self.unscaled = self.uniform_to_exp(self.scaled)
        self.size = size

        return self

    @classmethod
    def from_exp(cls, data_range: tuple, unscaled: np.ndarray):
        """Constructs ExpLogData from an array of unscaled data.
        
        data_range -- defines the bounds of the data (min: float, max: float)
        unscaled -- the unscaled data to construct the class with
        """

        self = cls(data_range)
        self.unscaled = unscaled
        self.scaled = self.exp_to_uniform(self.unscaled)
        self.size = len(unscaled)

        return self

    def uniform_to_exp(self, log_data: np.ndarray | float) -> np.ndarray | float:
        """Converts a uniform distribution on [0, 1) to data on a log range.
        
        Keyword arguments:
        log_data -- the data on [0, 1)
        """

        return 10**(self.log_range[0] + (self.log_range[1] - self.log_range[0]) * log_data)

    def exp_to_uniform(self, exp_data: np.ndarray | float) -> np.ndarray | float:
        """Converts data on a log range to data on [0, 1), i.e. inverts uniform_to_exp.
        
        Keyword arguments:
        data_range -- the exponential data to map to [0, 1)
        """

        return (np.log10(exp_data) - self.log_range[0])/(self.log_range[1] - self.log_range[0])

@dataclass
class MassFractionData:
    """Stores and generates random mass fraction data.
    
    data -- the array of mass fraction data.
    """

    data: np.ndarray = None
    num_nuclei: int = None
    size: int = None
    rng: np.random.Generator = None

    @classmethod
    def from_dirichlet(cls, num_nuclei: int, alpha: list = None, size: int = 2000, rng: np.random.Generator = default_rng):
        """Constructs MassFractionData from a dirichlet distribution.
        
        Keyword arguments:
        num_nuclei -- the number of nuclei in the composition.
        alpha -- the parameters used to generate dirichlet distributions. Defaults to a flat distribution.
        size -- the number of mass fraction lists to generate.
        rng 
        """

        if alpha is None:
            alpha = [1] * num_nuclei

        data = rng.dirichlet(alpha, size)
        self = cls(data, num_nuclei, size, rng)

        return self

    @classmethod
    def from_static(cls, mass_fractions: np.ndarray | list, size: int):
        """Constructs MassFractionData from a set mass fraction array.
        
        Keyword arguments:
        mass_fractions -- the array of mass fractions to use for every row.
        size -- the number of rows to generate.
        """

        num_nuclei = len(mass_fractions)
        data = np.tile(mass_fractions, size).reshape(size, num_nuclei)
        self = cls(data, num_nuclei, size)

        return self

@dataclass
class ScreeningFactorData:
    """Stores and generates random screening factors from temperature, density, and mass fraction data for a given composition.
    
    Keyword arguments
    temperatures -- an `ExpLogData` object containing the temperature data.
    densities -- an `ExpLogData` object containing the density data.
    mass_fractions -- a `MassFractionData` object containing the mass fraction data.

    comp -- a `pynucastro.Composition` object containing the nuclei in the system
    reactants -- a list of reactants (strings)
    threshold - the threshold for when screening becomes important to consider
    """

    temperatures: ExpLogData
    densities: ExpLogData
    mass_fractions: MassFractionData

    comp: pyna.Composition
    reactants: list

    threshold: float = 1.01

    # Class-Wide ReacLibLibrary
    reaclib_library = pyna.ReacLibLibrary()

    def __post_init__(self) -> None:
        """Creates a input and output data."""

        self.inputs = np.column_stack((
                self.temperatures.scaled,
                self.densities.scaled,
                self.mass_fractions.data
        ))

        # Defines Reaction Library and Screening Factors
        rfilter = pyna.RateFilter(self.reactants)
        r = self.reaclib_library.filter(rfilter).get_rates()[0]

        self.scn_fac = pyna.make_screen_factors(r.ion_screen[0], r.ion_screen[1])

        # Generates training and testing output data
        self.factors = self.screening_factors()
        self.indicators = self.screening_indicator(self.factors, self.threshold)

    def _screening_factor(self, temperature: float, density: float, mass_fractions: np.ndarray) -> float:
        """Computes the screening factor for a given temperature, density, and mass fraction distribution."""

        #self.comp.set_array(mass_fractions)
        for i, k in enumerate(self.comp.X):
            self.comp.X[k] = mass_fractions[i]

        plasma = pyna.make_plasma_state(temperature, density, self.comp.get_molar())
        return chugunov_2009(plasma, self.scn_fac)

    def screening_factors(self) -> np.ndarray:
        """Vectorization of self._screening_factor."""

        return np.array([
            self._screening_factor(temp, dens, fracs)
            for temp, dens, fracs
            in zip(self.temperatures.unscaled, self.densities.unscaled, self.mass_fractions.data)
        ])

    @staticmethod
    def screening_indicator(factors: np.ndarray, threshold: float) -> np.ndarray:
        """Indicator function for whether a screening factor is important.
        
        Keyword arguments:
        factors -- the screening factors to check.
        threshold -- the threshold over which the screening factors are relevant.
        """

        return to_categorical((factors > threshold).astype(int))
