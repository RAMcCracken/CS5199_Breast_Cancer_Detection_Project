import tensorflow as tf

class Rotate90Randomly(tf.keras.layers.experimental.preprocessing.PreprocessingLayer):
    def __init__(self):
        super(Rotate90Randomly, self).__init__()

    def call(self, x):
        # def random_rotate():
        rotation_factor = tf.random.uniform([], minval=0,
                                            maxval=4, dtype=tf.int32)
        rotated = tf.image.rot90(x, k=rotation_factor)

    
        rotated.set_shape(rotated.shape)
        return rotated