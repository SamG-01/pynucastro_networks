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
    temperature_range -- defines the bounds of the temperature data (min_temperature: float, max_temperature: float) 
    density_range -- defines the bounds of the density data (min_density: float, max_density: float)
    num_nuclei -- the number of nuclei in the composition. If a list is supplied, it will be used as the mass fractions.
    alpha - the parameters used to generate dirichlet distributions for the mass fractions (defaults to flat ones).
    size -- the number of (temp, dens, mass_fracs) data points to generate
    seed -- seed for random generation.
    """

    temperature_range: tuple
    density_range: tuple
    num_nuclei: int | list

    alpha: list = None
    size: int = 2000
    seed: int = None

    def __post_init__(self) -> None:
        """Generates and stores temperature, density, and mass fraction data."""

        # Defines rng
        self.rng = np.random.default_rng(self.seed)

        # Generates neural network inputs
        temperatures = self.generate_inputs(self.temperature_range)
        densities = self.generate_inputs(self.density_range)
        
        if isinstance(self.num_nuclei, list):
            mass_fractions = np.array(self.num_nuclei)/np.sum(self.num_nuclei)
            self.num_nuclei = len(mass_fractions)
            mass_fractions = np.tile(mass_fractions, self.size).reshape(self.size, self.num_nuclei)
        else:
            if self.alpha is None:
                self.alpha = [1] * self.num_nuclei
            mass_fractions = self.rng.dirichlet(self.alpha, self.size)

        self.x = np.column_stack((
            temperatures["scaled"],
            densities["scaled"],
            mass_fractions
        ))

        self.x_unscaled = {
            "temperatures": temperatures["actual"],
            "densities": densities["actual"],
            "mass_fractions": mass_fractions
        }

    def normalize_inputs(self, temp: float | np.ndarray, dens: float | np.ndarray, mass_frac: np.ndarray) -> np.ndarray:
        """Converts unnormalized temperature, density, and mass fraction data to a normalized array.
        
        Keyword arguments:
        temp -- a single temperature or a sequence of temperatures.
        dens -- a single density or a sequence of densities.
        mass_frac -- a single mass fraction distribution or a two-dimensional array of mass fraction distributions.
        """

        temp_scaled = self.exp_to_uniform(temp, self.temperature_range)
        dens_scaled = self.exp_to_uniform(dens, self.density_range)

        try:
            x = np.column_stack((temp_scaled, dens_scaled, mass_frac))
        except ValueError:
            x = np.array([temp_scaled, dens_scaled, *mass_frac])
            x = np.reshape(x, (1, x.shape[0]))
        return x

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
    inputs -- a `CompositionData` object with input data
    comp -- a `pynucastro.Composition` object containing the nuclei in the system
    reactants -- a list of reactants (strings)
    threshold - the threshold for when screening becomes important to consider
    """

    inputs: CompositionData

    comp: pyna.Composition
    reactants: list

    threshold: float = 1.01

    # Class-Wide ReacLibLibrary
    reaclib_library = pyna.ReacLibLibrary()

    def __post_init__(self) -> None:
        """Creates a dictionary of input and output data."""

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
            self._screening_factor(temperature, density, mass_fractions)
            for temperature, density, mass_fractions
            in zip(*self.inputs.x_unscaled.values())
        ])

    @staticmethod
    def screening_indicator(rates: np.ndarray, threshold: float) -> np.ndarray:
        """Indicator function for whether a screening factor is important."""

        return to_categorical((rates > threshold).astype(int))
