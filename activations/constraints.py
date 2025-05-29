import tensorflow as tf
from tensorflow.keras import constraints


class ClipConstraint(constraints.Constraint):
    """Clips weights between min_val and max_val."""

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_val, self.max_val)

    def get_config(self):
        return {'min_val': self.min_val, 'max_val': self.max_val}