import tensorflow as tf
import keras
from keras import models, layers
from keras.layers import Conv2D, Dense, GlobalAveragePooling2D, MaxPooling2D
from keras.applications.vgg19 import VGG19
from keras.models import Model
import tensorflow_addons as tfa
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


train_dir = 'output/train'
val_dir = 'output/val'
test_dir = 'output/test'

image_files = os.listdir(train_dir)
image_file = image_files[0]
image_path = os.path.join(train_dir, image_file)
img = mpimg.imread(image_path)
plt.imshow(img)
plt.axis('off')
# plt.show()

def transform_images_to_arrays(directory):
    image_arrays = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpeg'):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = np.array(img)
            image_arrays.append(img_array)
    return image_arrays

# Transform images from each directory to NumPy arrays
train_arr = transform_images_to_arrays(train_dir)
test_arr = transform_images_to_arrays(test_dir)
val_arr = transform_images_to_arrays(val_dir)

# Convert the lists of arrays to a single NumPy array
train_arr = np.array(train_arr)
test_arr = np.array(test_arr)
val_arr = np.array(val_arr)

# Print the shapes of the arrays
print(train_arr.shape) # (1040, 512, 512, 3)
print(test_arr.shape) # (130, 512, 512, 3)
print(val_arr.shape) # (130, 512, 512, 3)


classes = [
    'crop',
    'weed' # (2,1)
]



target_size = 32
representation_dim = 512
projection_units = 128
num_clusters = 20
k_neighbours = 5
tune_encoder_during_clustering = False


data_preprocessing = keras.Sequential(
    [
        layers.Resizing(target_size, target_size), # (1, 512, 512, 3)
        layers.Normalization(), # (1, 512, 512, 3)
    ]
)
# Compute the mean and the variance from the data for normalization.
data_preprocessing.layers[-1].adapt(train_arr)


data_augmentation = keras.Sequential(
    [
        layers.RandomTranslation(
            height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode="nearest" # (1, 512, 512, 3)
        ),
        layers.RandomFlip(mode="horizontal"), # (1, 512, 512, 3)
        layers.RandomRotation(factor=0.15, fill_mode="nearest"), # (1, 512, 512, 3)
        layers.RandomZoom(
            height_factor=(-0.3, 0.1), width_factor=(-0.3, 0.1), fill_mode="nearest" # (1, 512, 512, 3)
        ),
    ]
)


def create_encoder_with_layers(representation_dim):
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

    x = base_model.output # (None, 16, 16, 512) ==> MaxPooling layer
    # x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x) # (None, 14, 14, 64)
    # x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x) # (None, 12, 12, 128)
    # x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(x) # (None, 10, 10, 256)
    # x = MaxPooling2D()(x)                                   # (None, 5, 5, 256)
    # x = Dense(units=representation_dim, activation='relu')(x)         # (None, 5, 5, 512)

    model = Model(inputs=base_model.input, outputs=x) # (None, 16, 16, 512)

    return model

class RepresentationLearner(keras.Model): # (1, 512, 512, 3)
    def __init__(
        self,
        encoder, # base model or feature extractor that transforms the input data into a meaningful representation.
        projection_units,
        num_augmentations, # augmented versions of each input sample used during training
        temperature=1.0, # sharpness of the predicted probability distributions
        dropout_rate=0.1,
        l2_normalize=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        # further transform and refine these features
        self.projector = VGG19(
            include_top=False,
            weights=None,
            pooling="avg",
            input_shape=(None, encoder.output_shape[-1], 3), # (None, None, 512)
        )

        self.num_augmentations = num_augmentations
        self.temperature = temperature
        self.l2_normalize = l2_normalize
        self.loss_tracker = keras.metrics.Mean(name="loss")



    # defines a property metrics which returns a list containing the loss_tracker metric
    @property
    def metrics(self):
        return [self.loss_tracker]

    # loss function commonly used in unsupervised learning tasks, particularly in contrastive learning.
    # It aims to encourage similar instances to be closer together in the embedding space while pushing dissimilar instances apart.
    # The idea behind contrastive loss is to maximize the similarity between positive pairs (instances that should be similar) and minimize the similarity between negative pairs (instances that should be dissimilar).
    def compute_contrastive_loss(self, feature_vectors, batch_size):
        num_augmentations = tf.shape(feature_vectors)[0] // batch_size
        if self.l2_normalize:
            feature_vectors = tf.math.l2_normalize(feature_vectors, -1)
        # The logits shape is [num_augmentations * batch_size, num_augmentations * batch_size].
        logits = (
            tf.linalg.matmul(feature_vectors, feature_vectors, transpose_b=True)
            / self.temperature
        )
        # Apply log-max trick for numerical stability.
        logits_max = tf.math.reduce_max(logits, axis=1)
        logits = logits - logits_max
        # The shape of targets is [num_augmentations * batch_size, num_augmentations * batch_size].
        # (2,2)
        # targets is a matrix consists of num_augmentations submatrices of shape [batch_size * batch_size].
        # Each [batch_size * batch_size] submatrix is an identity matrix (diagonal entries are ones).
        targets = tf.tile(tf.eye(batch_size), [num_augmentations, num_augmentations])
        # Compute cross entropy loss
        return keras.losses.categorical_crossentropy(
            y_true=targets, y_pred=logits, from_logits=True
        )

    # main entry point of the model
    def call(self, inputs):
        preprocessed = data_preprocessing(inputs)
        augmented = []
        for _ in range(self.num_augmentations):
            augmented.append(data_augmentation(preprocessed))
        augmented = layers.Concatenate(axis=0)(augmented)
        features = self.encoder(augmented)
        return self.projector(features)

    def train_step(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # Run the forward pass and compute the contrastive loss
        with tf.GradientTape() as tape:
            feature_vectors = self(inputs, training=True)
            loss = self.compute_contrastive_loss(feature_vectors, batch_size)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update loss tracker metric
        self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        batch_size = tf.shape(inputs)[0]
        feature_vectors = self(inputs, training=False)
        loss = self.compute_contrastive_loss(feature_vectors, batch_size)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


encoder = create_encoder_with_layers(representation_dim) # (2, 16, 16, 512)
representation_learner = RepresentationLearner(
    encoder, projection_units, num_augmentations=1, temperature=0.1
)

lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001, decay_steps=500, alpha=0.1
)

representation_learner.compile(
    optimizer=tfa.optimizers.AdamW(learning_rate=lr_scheduler, weight_decay=0.0001),
)

history = representation_learner.fit(
    train_arr,
    batch_size=1,
    epochs=5
)

# input shape (None, 512, 512, 3)
# output shape (None, 16, 16, 512) len 4
# input shape of projector : (None, None, 512, 3) len 4
# output shape of projector (None,512)
