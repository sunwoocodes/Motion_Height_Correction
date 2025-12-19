import tensorflow as tf
from tensorflow.keras import layers, models

def build_mlp(input_dim=99, output_dim=99):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(output_dim)        # height-refined output
    ])
    return model
