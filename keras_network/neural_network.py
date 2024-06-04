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
    ) -> bool:
    """Returns a model's prediction for how important screening is for a given temperature, density, and Composition.

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

    return not bool(network.model.predict(x, verbose=0).item())

class ScreeningFactorNetwork:
    """Contains a `keras` neural network trained to identify the importance
    of screening for a given temperature, density, and composition.

    https://keras.io/examples/structured_data/imbalanced_classification/
    """

    def __init__(self, data: ScreeningFactorData, seed: int = None) -> None:
        """Defines the model's layers.
    
        Keyword arguments:
            `data`: the `ScreeningFactorData` object containing the training, validation, and testing data
            `seed`: used to seed `keras` random number generation
        """

        self.data = data

        # Sets rng
        if seed is not None:
            keras.utils.set_random_seed(seed)

        # Computes output and class bias
        pos = self.data.frac_pos
        neg = 1 - pos

        self.class_weight = {0: 5*0.5/neg, 1: 0.5/pos}
        #self.initial_bias = np.log(pos/neg)
        #self.initial_bias = keras.initializers.Constant(self.initial_bias)

        # Defines threshold for defining false positives/negatives
        self.confidence = 0.5

        # Sets up model framework
        self.score = []

        # Normalization layer
        self.normalization = keras.layers.Normalization(axis=-1)
        self.normalization.adapt(self.data.x["train"])

        # Model layers
        self.model = keras.Sequential(
            [
                self.normalization,
                keras.layers.Dense(512, activation="relu"),
                keras.layers.Dense(512, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(512, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        self.metrics = [
            keras.metrics.FalsePositives(name="fp"),
            keras.metrics.FalseNegatives(name="fn"),
            keras.metrics.TruePositives(name="tp"),
            keras.metrics.TrueNegatives(name="tn"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")
        ]

        self.loss = [
            keras.losses.BinaryCrossentropy()
        ]

        self.callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='fp',
                verbose=1,
                patience=25,
                mode='min',
                restore_best_weights=True
            )
        ]

    def compile(self, learning_rate: float = 5e-4) -> None:
        """Compiles the model."""

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self.loss,
            metrics=self.metrics
        )

    def fit_model(self, verbose: int = 0) -> keras.callbacks.History:
        """Fits the model to the data and computes its score.
        
        Keyword arguments:
            `verbose`: how verbose `self.model.fit` should be
        """

        x, y = self.data.x, self.data.y

        self.model.fit(
            x=x["train"],
            y=y["train"],
            batch_size=2048,
            epochs=500,
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
        """Returns a model's prediction for how important screening is for a given temperature, density, and Composition.

        Keyword arguments:
            `temp`, `dens`: the temperature and density.
            `comp`: the `Composition` to consider.
            `nuclei`: the pair of nuclei to predict screening for.
        """
        return _predict(self, temp, dens, comp, nuclei)
