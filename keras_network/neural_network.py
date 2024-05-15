"""Class for training a `keras` neural network from screening factor data."""

import numpy as np
import pynucastro as pyna
import keras

from .data_generation import ScreeningFactorData

__all__ = ["ScreeningFactorNetwork"]

@np.vectorize(excluded=[0], signature="(),(),(),(2)->()")
def _predict(
        network,
        temp: float, dens: float,
        comp: pyna.Composition,
        nuclei: tuple[pyna.Nucleus, pyna.Nucleus]
    ) -> float:
    """Returns a model's prediction for the screening factor given a temperature, density, and Composition.

    Keyword arguments:
        `network`: the `ScreeningFactorNetwork` to perform the prediction for.
        `temp`, `dens`: the temperature and density.
        `comp`: the `Composition` to consider.
        `nuclei`: the pair of nuclei to predict screening for.
    """

    log_temp, log_dens = np.log10((temp, dens))

    plasma = pyna.make_plasma_state(temp, dens, comp.get_molar())
    scn_fac = pyna.make_screen_factors(*nuclei)

    x = np.array([
        log_temp, log_dens,
        plasma.abar, plasma.zbar, plasma.z2bar,
        scn_fac.z1, scn_fac.a1,
        scn_fac.z2, scn_fac.z2
    ]).reshape(1, 9)

    return network.model.predict(x, verbose=0).item()

class ScreeningFactorNetwork:
    """Contains a `keras` neural network trained to identify the importance
    of screening for a given temperature, density, and composition.

    https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    """

    def __init__(self, data: ScreeningFactorData, seed: int = None) -> None:
        """Defines the model's layers.
    
        Keyword arguments:
            `data`: the `ScreeningFactorData` object containing the training and testing data
            `seed`: used to seed `keras` random number generation
        """

        self.data = data

        # Sets rng
        if seed is not None:
            keras.utils.set_random_seed(seed)

        # Sets up model framework
        self.score = []

        self.model = keras.Sequential(
            [
                keras.layers.BatchNormalization(axis=-1, scale=False, center=False),
            ]
        )

        self.metrics = []

        self.loss = []

        self.callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_prc',
                verbose=1,
                patience=10,
                mode='max',
                restore_best_weights=True
            )
        ]

    def compile(self) -> None:
        """Compiles the model."""

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=self.loss,
            metrics=self.metrics
        )

    def fit_model(self, verbose=0) -> keras.callbacks.History:
        """Fits the model to the data and computes its score.
        
        Keyword arguments:
            `verbose`: how verbose `self.model.fit` should be
        """

        x, y = self.data.x, self.data.y

        self.model.fit(
            x=x["train"],
            y=y["train"],
            batch_size=2048,
            epochs=100,
            verbose=verbose,
            callbacks=self.callbacks,
            validation_data=(x["validate"], y["validate"]),
            class_weight=self.class_weight
        )

        self.score = self.model.evaluate(
            x=x["test"],
            y=y["test"],
            verbose=0
        )

    def predict(
            self, temp: float, dens: float,
            comp: pyna.Composition,
            nuclei: tuple[pyna.Nucleus, pyna.Nucleus]
        ) -> bool:
        """Returns a model's prediction for the screening factor
        of a pair of nuclei given a temperature, density, and Composition.

        Keyword arguments:
            `temp`, `dens`: the temperature and density.
            `comp`: the `Composition` to consider.
            `nuclei`: the pair of nuclei to predict screening for.
        """
        return _predict(self, temp, dens, comp, nuclei)
