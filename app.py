import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Paths to the dataset directories
train_dir = 'train'
valid_dir = 'valid'
test_dir = 'test'

# Image dimensions
img_height, img_width = 224, 224
batch_size = 8

# Function to update and overwrite class_names.json with current classes
def update_and_overwrite_class_names():
    # Get the list of current class names from the directories
    all_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    # Create a new class names dictionary
    class_names = {cls: i for i, cls in enumerate(all_classes)}

    # Write the updated class names to class_names.json, overwriting if it exists
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f, indent=4)

# Function to create necessary folders
def create_folders(class_name):
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

# Data augmentation
def augment_image(image, count=30):
    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    augmented_images = []

    for _ in range(count):
        augmented_image = datagen.flow(image, batch_size=1)[0]
        augmented_image = array_to_img(augmented_image[0])
        augmented_images.append(augmented_image)
    
    return augmented_images

# Save images to respective folders
def save_images(images, class_name):
    train_folder = os.path.join(train_dir, class_name)
    valid_folder = os.path.join(valid_dir, class_name)
    test_folder = os.path.join(test_dir, class_name)
    
    for i, img in enumerate(images):
        if i < 3:
            img.save(os.path.join(test_folder, f'image_{i}.jpg'))
        elif i < 9:
            img.save(os.path.join(valid_folder, f'image_{i}.jpg'))
        else:
            img.save(os.path.join(train_folder, f'image_{i}.jpg'))

# Preprocess image for prediction
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match the model input
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Retrain the model
def retrain_model():
    # Update and overwrite class names before retraining
    update_and_overwrite_class_names()

    with open('class_names.json', 'r') as f:
        class_names = json.load(f)

    num_classes = len(class_names)
    
    # Prepare data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Load MobileNetV2 model with pre-trained weights and without top layers
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Define the model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(
        train_generator,
        epochs=5,
        validation_data=valid_generator
    )

    # Save the retrained model
    model.save('staff_mobilenet_v2_model.h5')

# Streamlit app layout
st.title("Staff Image Recognition")

st.write("Upload an image of a staff member to get predictions and add new images to the dataset.")

# User input for class name
class_name = st.text_input("Enter the class name for the new images:")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None and class_name:
    # Create folders if they don't exist
    create_folders(class_name)

    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Augment and save images
    augmented_images = augment_image(image)
    save_images(augmented_images, class_name)

    # Retrain the model
    retrain_model()

    # Load the updated model for prediction
    model = tf.keras.models.load_model('staff_mobilenet_v2_model.h5')

    # Preprocess and predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)

    predicted_class = list(class_names.keys())[predicted_class_index]

    # Display prediction
    st.write(f"Prediction: {predicted_class} with probability {np.max(predictions):.2f}")
