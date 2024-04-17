"""Class for training a `keras` neural network from screening factor data."""

from ScreeningFactorData import dataclass, ScreeningFactorData
import keras

@dataclass
class ScreeningFactorNetwork:
    """Contains a keras neural network trained to identify the importance
    of screening for a given temperature, density, and composition.
    
    Keyword arguments:
    data -- a `ScreeningFactorData` object containing the data to train
    batch_size -- the batch size.
    epochs -- the number of epochs to train with.
    """

    data: ScreeningFactorData
    batch_size: int = 200
    epochs: int = 20

    def __post_init__(self) -> None:
        """Defines the model's layers, compilation arguments, and callbacks."""

        self.input_shape = self.data.training["input"].x["scaled"].shape[1:]

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
            self.data.training["input"].x["scaled"],
            self.data.training["indicator"],
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.15,
            callbacks=self.callbacks,
            verbose=verbose
        )

        self.score = self.model.evaluate(
            self.data.testing["input"].x["scaled"],
            self.data.testing["indicator"],
            verbose=0
        )
        self.loss_value, self.accuracy = self.score

    def plot_model(self) -> None:
        """Plots the layers of the model."""

        keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=True, dpi=100)
