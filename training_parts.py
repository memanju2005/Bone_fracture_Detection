import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# load images to build and train the model
#                       ....                                     /    img1.jpg
#             test      Hand            patient0000   positive  --   img2.png
#           /                /                         \    .....
#   Dataset   -         Elbow  ------   patient0001
#           \ train               \         /                           img1.png
#                       Shoulder        patient0002     negative --      img2.jpg
#                       ....                   \
#
def load_path(path):
    """
    Iterates through the dataset directory structure to collect image file paths
    and their corresponding body part labels (Elbow, Hand, or Shoulder).
    Returns a list of dictionaries containing 'label' and 'image_path'.
    """
    dataset = []
    for folder in os.listdir(path):
        folder = path + '/' + str(folder)
        if os.path.isdir(folder):
            for body in os.listdir(folder):
                path_p = folder + '/' + str(body)
                for id_p in os.listdir(path_p):
                    patient_id = id_p
                    path_id = path_p + '/' + str(id_p)
                    for lab in os.listdir(path_id):
                        if lab.split('_')[-1] == 'positive':
                            label = 'fractured'
                        elif lab.split('_')[-1] == 'negative':
                            label = 'normal'
                        path_l = path_id + '/' + str(lab)
                        for img in os.listdir(path_l):
                            img_path = path_l + '/' + str(img)
                            dataset.append(
                                {
                                    'label': body,
                                    'image_path': img_path
                                }
                            )
    return dataset


# ---------------------------------------------------------
# 1. DATA LOADING AND PREPARATION
# ---------------------------------------------------------
# Define the root dataset path relative to this script
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
image_dir = THIS_FOLDER + '/Dataset'

# Parse the dataset directory to collect all file paths and labels
data = load_path(image_dir)
labels = []
filepaths = []

# add labels for dataframe for each category 0-Elbow, 1-Hand, 2-Shoulder
Labels = ["Elbow", "Hand", "Shoulder"]
for row in data:
    labels.append(row['label'])
    filepaths.append(row['image_path'])

# Convert the lists into Pandas Series objects for dataframe creation
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# Concatenate the two series side-by-side (axis=1) into a unified Pandas DataFrame
images = pd.concat([filepaths, labels], axis=1)

# Create an initial split: Reserve 10% of the data strictly for testing the final model.
# The remaining 90% will be further split downstream into training and validation sets.
train_df, test_df = train_test_split(images, train_size=0.9, shuffle=True, random_state=1)

# each generator to process and convert the filepaths into image arrays,
# and the labels into one-hot encoded labels.
# The resulting generators can then be used to train and evaluate a deep learning model.

# now we have 10% test, 72% training and 18% validation

# ---------------------------------------------------------
# 2. DATA PIPELINE GENERATORS
# ---------------------------------------------------------
# ImageDataGenerator processes images in batches (preventing out-of-memory errors).
# It also applies ResNet50's specific preprocessing to format pixel math correctly.
# The train_generator reserves 20% of the 90% training data for validation.
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    validation_split=0.2)

# The test generator only needs preprocessing, no validation splitting.
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=64,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=64,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# ---------------------------------------------------------
# 3. MODEL ARCHITECTURE (TRANSFER LEARNING)
# ---------------------------------------------------------
# Load the pre-trained ResNet50 model. We exclude its top prediction layer (include_top=False)
# because we only want to use it as a feature extractor. We apply global average pooling.
pretrained_model = tf.keras.applications.resnet50.ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg')

# Freeze the ResNet50 weights so they aren't destroyed during our training.
# This saves massive amounts of computation time and prevents overfitting.
pretrained_model.trainable = False

# Build our custom classification "head" on top of the ResNet50 base.
inputs = pretrained_model.input
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(50, activation='relu')(x)
# Final output layer uses 'softmax' to output probabilities for our 3 specific categories
outputs = tf.keras.layers.Dense(len(Labels), activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
print(model.summary())

# ---------------------------------------------------------
# 4. TRAINING AND COMPILATION
# ---------------------------------------------------------
# Compile the model with the Adam optimizer using a low learning rate to make careful adjustments.
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define EarlyStopping callback to halt training if our validation loss doesn't improve
# for 3 consecutive epochs. This secures the best epoch weights to prevent overfitting.
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Execute the training loop using the generators
history = model.fit(train_images, validation_data=val_images, epochs=25,
                    callbacks=[callbacks])

# Save the fully trained model to disk so it can be loaded later by the GUI interface.
model.save(THIS_FOLDER + "/weights/ResNet50_BodyParts.h5")

# ---------------------------------------------------------
# 5. EVALUATION AND VISUALIZATION
# ---------------------------------------------------------
# Evaluate the completely unseen 10% test data to get our true final accuracy.
results = model.evaluate(test_images, verbose=0)
print(results)
print(f"Test Accuracy: {np.round(results[1] * 100, 2)}%")


# create plots for accuracy and save it
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# create plots for loss and save it
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
