import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)


(train_images, train_labels), (test_images, test_labels) = load_data(path="./mnist.npz")

# train_image.shape:(60000, 28, 28)
# train_labels.shape:(60000,)
# test_image.shape:(10000, 28, 28)
# test_labels.shape:(10000,)
train_images = train_images.reshape(60000, 784).astype('float32')/255
test_images = test_images.reshape(10000, 784).astype('float32')/255

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(100)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.batch(100)


class InputLayer(layers.Layer):
    def __init__(self, units=784):
        super(InputLayer, self).__init__()
        self.units = units

    def call(self, inputs, **kwargs):
        return inputs


class HiddenLayer(layers.Layer):
    def __init__(self, units):
        super(HiddenLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'), trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(self.units, ), dtype='float32'), trainable=True)

    def call(self, inputs, **kwargs):
        output = tf.matmul(inputs, self.w) + self.b
        regularizer_L2 = tf.keras.regularizers.L2(.01)
        self.add_loss(regularizer_L2(self.w))
        return output


class OutputLayer(layers.Layer):
    def __init__(self, units):
        super(OutputLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'), trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=self.units,), dtype='float32', trainable=True)

    def call(self, inputs, **kwargs):
        output = tf.matmul(inputs, self.w) + self.b
        regularizer_L2 = tf.keras.regularizers.L2(.001)
        self.add_loss(regularizer_L2(self.w))
        return output


class MLPBlock(tf.keras.Model):
    def __init__(self):
        super(MLPBlock, self).__init__()
        self.input_layer = InputLayer(784)
        self.hidden_layer = HiddenLayer(500)
        self.output_layer = OutputLayer(10)
        self.softmax = layers.Softmax()

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        x = self.hidden_layer(x)
        x = tf.nn.relu(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
mlpmodel = MLPBlock()


@tf.function
def train_step(train_images, train_labels):
    with tf.GradientTape() as tape:
        predictions = mlpmodel(train_images)
        loss = loss_fn(train_labels,predictions)
        total_loss = loss + sum(mlpmodel.losses)

    gradients = tape.gradient(total_loss, mlpmodel.trainable_variables)
    optimizer.apply_gradients(zip(gradients, mlpmodel.trainable_variables))
    train_loss(loss)
    train_accuracy(train_labels, predictions)


@tf.function
def test_step(test_images, test_labels):
    predictions = mlpmodel(test_images)
    loss = loss_fn(test_labels, predictions)
    test_loss(loss)
    test_accuracy(test_labels, predictions)


EPOCHES = 100
for epoch in range(EPOCHES):
    for train_images, train_labels in train_dataset:
        train_step(train_images, train_labels)

    for test_images, test_labels in test_dataset:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1, train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))