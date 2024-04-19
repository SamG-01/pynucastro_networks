"""A collection of classes to help create a neural network for whether screening factors are important."""

from dataclasses import dataclass
from keras.utils import to_categorical
import numpy as np

import pynucastro as pyna
from pynucastro.screening import chugunov_2009

@dataclass
class CompositionData:
    """Stores and generates random temperature, density, and mass fraction data for a given composition.
    
    Keyword arguments:
    temperature_range - defines the bounds of the temperature data (min_temperature: float, max_temperature: float) 
    density_range - defines the bounds of the density data (min_density: float, max_density: float)
    num_nuclei - number of nuclei in the composition
    alpha - the parameters used to generate dirichlet distributions for the mass fractions
    size - the number of (temp, dens, mass_fracs) data points to generate
    seed - seed for random number generation
    """

    temperature_range: tuple
    density_range: tuple

    num_nuclei: int
    alpha: list = None

    size: int = 2000
    seed: int = None

    def __post_init__(self) -> None:
        """Generates and stores temperature, density, and mass fraciton data."""

        # Defines rng
        self.rng = np.random.default_rng(self.seed)

        # defaults to a flat Dirichlet distribution
        if self.alpha is None:
            self.alpha = [1] * self.num_nuclei

        # Generates neural network inputs
        self.temperatures = self.generate_inputs(self.temperature_range)
        self.densities = self.generate_inputs(self.density_range)
        self.mass_fractions = self.rng.dirichlet(self.alpha, self.size)

        # Stores input in a dictionary
        self.x = {}
        self.x["scaled"] = np.column_stack((
            self.temperatures["scaled"],
            self.densities["scaled"],
            self.mass_fractions
        ))
        self.x["actual"] = list(zip(
            self.temperatures["actual"],
            self.densities["actual"],
            self.mass_fractions
        ))

    def generate_inputs(self, data_range: tuple) -> np.ndarray:
        """Generates random temperature or density data.
        
        Keyword arguments:
        data_range -- the range over which to generate the inputs
        """

        log_data = self.rng.uniform(size=self.size)
        data = {
            "scaled": log_data,
            "actual": self.uniform_to_exp(log_data, data_range)
        }
        return data

    @staticmethod
    def uniform_to_exp(log_data: np.ndarray | float, data_range: tuple) -> np.ndarray | float:
        """Converts a uniform distribution on [0, 1) to data on a log range.
        
        Keyword arguments:
        data_range -- the interval to map the exponential distribution to
        """

        log_range = np.log10(data_range)
        return 10**(log_range[0] + (log_range[1] - log_range[0]) * log_data)

    @staticmethod
    def exp_to_uniform(exp_data: np.ndarray | float, data_range: tuple) -> np.ndarray | float:
        """Converts data on a log range to data on [0, 1), i.e. inverts uniform_to_exp.
        
        Keyword arguments:
        data_range -- the interval of exponential data to map to [0, 1)
        """

        log_range = np.log10(data_range)
        return (np.log10(exp_data) - log_range[0])/(log_range[1] - log_range[0])

@dataclass
class ScreeningFactorData:
    """Stores and generates random screening factors from temperature, density, and mass fraction data for a given composition.
    
    Keyword arguments
    comp -- a pynucastro `Composition` object
    reactants -- a list of reactants (strings)
    temperature_range - defines the bounds of the temperature data (min_temperature: float, max_temperature: float) 
    density_range - defines the bounds of the density data (min_density: float, max_density: float)
    size - the number of (temp, dens, mass_fracs) input data points to generate
    threshold - the threshold for when screening becomes important to consider
    seed - seed for random number generation
    """

    comp: pyna.Composition
    reactants: list

    temperature_range: tuple
    density_range: tuple

    size: int = 2000
    threshold: float = 1.01
    seed: int = None

    # Class-Wide ReacLibLibrary
    reaclib_library = pyna.ReacLibLibrary()

    def __post_init__(self) -> None:
        """Creates a dictionary of input and output data."""

        # Defines rng
        self.rng = np.random.default_rng(self.seed)

        # Defines Reaction Library and Screening Factors
        rfilter = pyna.RateFilter(self.reactants)
        r = self.reaclib_library.filter(rfilter).get_rates()[0]

        self.scn_fac = pyna.make_screen_factors(r.ion_screen[0], r.ion_screen[1])

        self.training = self.generate_data()
        self.testing = self.generate_data()

    def generate_data(self, X: CompositionData=None) -> dict:
        """Generates input and rate data.
        
        Keyword arguments:
        X -- a preexisting `CompositionData` object. If excluded, the method will generate its own random one.
        """

        if X is None:
            X = CompositionData(
                self.temperature_range,
                self.density_range,
                num_nuclei=len(self.comp.get_nuclei()),
                size=self.size,
                seed=self.seed
            )
        Y = self.screening_factor(X.x["actual"])
        Z = self.screening_indicator(Y)

        return {"input": X, "factors": Y, "indicator": Z}

    def _screening_factor(self, temperature: float, density: float, mass_fractions: np.ndarray) -> float:
        """Computes the screening factor for a given temperature, density, and mass fraction distribution."""

        #self.comp.set_array(mass_fractions)
        for i, k in enumerate(self.comp.X):
            self.comp.X[k] = mass_fractions[i]

        plasma = pyna.make_plasma_state(temperature, density, self.comp.get_molar())
        return chugunov_2009(plasma, self.scn_fac)

    def screening_factor(self, x_unscaled: list) -> np.ndarray:
        """Vectorization of self._screening_factor."""

        return np.array([self._screening_factor(*p) for p in x_unscaled])

    def screening_indicator(self, rates: np.ndarray) -> np.ndarray:
        """Indicator function for whether a screening factor is important."""

        return to_categorical((rates > self.threshold).astype(int))
