import tensorflow as tf
from tensorflow.keras import layers
from .constraints import ClipConstraint


class SGBlend(layers.Layer):
    """SGBlend Activation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Build the layer's weights."""
        # Learnable parameters
        self.alpha = self.add_weight(
            name="alpha",
            shape=(1,),
            initializer="zeros",  # Start with GELU (alpha=0)
            constraint=tf.keras.constraints.MinMaxNorm(0.0, 1.0)
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(1,),
            initializer="glorot_uniform",
            constraint=ClipConstraint(0.1, 10.0)
        )
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1,),
            initializer="zeros"
        )

        super().build(input_shape)

    def call(self, inputs):
        """Forward pass of the layer."""
        # GELU approximation: x * σ(1.702x)
        gelu_term = inputs * tf.nn.sigmoid(1.702 * inputs)
        # SSwish: x * σ(beta x) - gamma
        sswish_term = inputs * tf.nn.sigmoid(self.beta * inputs) - self.gamma
        # Blend using alpha (sigmoid to enforce [0,1])
        alpha = tf.nn.sigmoid(self.alpha)
        return alpha * sswish_term + (1 - alpha) * gelu_term


class SSwish(layers.Layer):
    """Symmetric Swish."""

    def build(self, input_shape):
        """Build the layer's weights."""
        self.beta = self.add_weight(
            name="beta",
            shape=(1,),
            initializer="glorot_uniform",
            constraint=ClipConstraint(0.1, 10)
        )
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1,),
            initializer="zeros"
        )
        super().build(input_shape)

    def call(self, inputs):
        """Forward pass of the layer."""
        return inputs * tf.nn.sigmoid(self.beta * inputs) - self.gamma


class RELU(layers.Layer):
    """Rectified Linear Unit."""

    def call(self, inputs):
        """Forward pass of the layer."""
        return tf.nn.relu(inputs)


class GELU(layers.Layer):
    """Gaussian Error Linear Unit."""

    def call(self, inputs):
        """Forward pass of the layer."""
        return tf.nn.gelu(inputs)


class Mish(layers.Layer):
    """Mish Activation."""

    def call(self, inputs):
        """Forward pass of the layer."""
        return inputs * tf.math.tanh(tf.math.softplus(inputs))
