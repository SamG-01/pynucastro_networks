"""Class for training a `keras` neural network from screening factor data."""

from ScreeningFactorData import dataclass, ScreeningFactorData, np
import keras

@dataclass
class ScreeningFactorNetwork:
    """Contains a keras neural network trained to identify the importance
    of screening for a given temperature, density, and composition.
    
    Keyword arguments:
    training -- a `ScreeningFactorData` object containing the data to train on
    testing -- a `ScreeningFactorData` object containing the data to test the network on
    """

    training: ScreeningFactorData
    testing: ScreeningFactorData

    seed: int = None

    def __post_init__(self) -> None:
        """Defines the model's layers and compiles it."""

        # Sets rng
        if self.seed is not None:
            keras.utils.set_random_seed(self.seed)

        self.input_shape = self.training.inputs.shape[1:]

        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=self.input_shape),
                keras.layers.Dense(800, activation="relu"),
                keras.layers.Dropout(0.4),
                keras.layers.Dense(600, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(2, activation="softmax")
            ]
        )

        self.model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

        self.callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
        ]

        self.score = [None, None]
        self.loss_value = self.accuracy = None

    def fit_model(self, verbose=0) -> keras.callbacks.History:
        """Fits the model to the data and computes its score.
        
        Keyword arguments:
        verbose -- how verbose `self.model.fit` should be
        """

        self.model.fit(
            x=self.training.inputs,
            y=self.training.indicators,
            batch_size=200,
            epochs=20,
            validation_split=0.15,
            callbacks=self.callbacks,
            verbose=verbose
        )

        self.score = self.model.evaluate(
            x=self.testing.inputs,
            y=self.testing.indicators,
            verbose=0
        )
        self.loss_value, self.accuracy = self.score

    def plot_model(self) -> None:
        """Plots the layers of the model."""

        keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=True, dpi=100)

    def predict(self, x: ScreeningFactorData) -> np.ndarray:
        """Predicts whether screening is important for a given `ScreeningFactorData` object.

        Keyword arguments:
        x -- the input data fed to the network.
        """
    
        return self.model.predict(x.inputs)
