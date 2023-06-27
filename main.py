import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10
)

# Define the directories for train and validation data
train_dir = 'data/train'
test_dir = 'data/test'
val_dir = 'data/val'

# Set up the DirectoryIterator for train and validation data
train_generator = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
test_generator = datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
val_generator = datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')



def vgg19(input_shape=(224, 224, 3), num_classes=1000):
    model = tf.keras.Sequential(name='VGG19')

    # Block 1
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 2
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 3
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 4
    model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Block 5
    model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Classification layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return model


# Create an instance of VGG19 model
model = vgg19()

# Print the model summary
model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)



history = model.fit(
    #otherwise img_gen keeps generating new images
    train_dir,
    #batch
    batch_size=1,
    validation_data=val_dir,
    verbose=1,
    epochs=2,
)

