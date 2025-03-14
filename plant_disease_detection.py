# Import necessary libraries
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Function to count total files in a folder
def total_files(folder_path):
    return len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

# Dataset paths
dataset_path = 'Dataset'

train_files_healthy = os.path.join(dataset_path, 'Train', 'Healthy')
train_files_powdery = os.path.join(dataset_path, 'Train', 'Powdery')
train_files_rust = os.path.join(dataset_path, 'Train', 'Rust')

test_files_healthy = os.path.join(dataset_path, 'Test', 'Healthy')
test_files_powdery = os.path.join(dataset_path, 'Test', 'Powdery')
test_files_rust = os.path.join(dataset_path, 'Test', 'Rust')

valid_files_healthy = os.path.join(dataset_path, 'Validation', 'Healthy')
valid_files_powdery = os.path.join(dataset_path, 'Validation', 'Powdery')
valid_files_rust = os.path.join(dataset_path, 'Validation', 'Rust')

# Print the number of images in each dataset category
def print_file_counts():
    print("Number of healthy leaf images in training set", total_files(train_files_healthy))
    print("Number of powdery leaf images in training set", total_files(train_files_powdery))
    print("Number of rusty leaf images in training set", total_files(train_files_rust))
    print("========================================================")
    print("Number of healthy leaf images in test set", total_files(test_files_healthy))
    print("Number of powdery leaf images in test set", total_files(test_files_powdery))
    print("Number of rusty leaf images in test set", total_files(test_files_rust))
    print("========================================================")
    print("Number of healthy leaf images in validation set", total_files(valid_files_healthy))
    print("Number of powdery leaf images in validation set", total_files(valid_files_powdery))
    print("Number of rusty leaf images in validation set", total_files(valid_files_rust))

# Display sample images
def display_images(image_paths):
    for image_path in image_paths:
        with open(image_path, 'rb') as f:
            display_image = Image.open(f)
            display_image.show()

# Create image data generators
def create_data_generators():
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    return train_datagen, test_datagen

# Create data generators
def get_data_generators(train_datagen, test_datagen):
    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_path, 'Train'),
        target_size=(225, 225),
        batch_size=32,
        class_mode='categorical'
    )
    validation_generator = test_datagen.flow_from_directory(
        os.path.join(dataset_path, 'Validation'),
        target_size=(225, 225),
        batch_size=32,
        class_mode='categorical'
    )
    return train_generator, validation_generator

# Build the CNN model
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(225, 225, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Plot training history
def plot_history(history):
    sns.set_theme()
    sns.set_context("poster")
    plt.figure(figsize=(25, 25), dpi=100)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Preprocess image for prediction
def preprocess_image(image_path, target_size=(225, 225)):
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    return x

# Predict the label of an image
def predict_image(model, image_path, labels):
    x = preprocess_image(image_path)
    predictions = model.predict(x)
    predicted_label = labels[np.argmax(predictions)]
    return predicted_label

# Main function
def main():
    # Print file counts
    print_file_counts()

    # Display sample images
    display_images([
        os.path.join(train_files_healthy, '8ce77048e12f3dd4.jpg'),
        os.path.join(train_files_rust, '8bc27bdbdf8092a0.jpg')
    ])

    # Create data generators
    train_datagen, test_datagen = create_data_generators()
    train_generator, validation_generator = get_data_generators(train_datagen, test_datagen)

    # Build and train model
    model = build_model()
    history = model.fit(
        train_generator,
        epochs=5,
        validation_data=validation_generator
    )

    # Plot training history
    plot_history(history)

    # Save the model
    model.save("plant_disease_detection_model.h5")

    # Load class labels
    labels = train_generator.class_indices
    labels = {v: k for k, v in labels.items()}

    # Predict an image
    predicted_label = predict_image(model, os.path.join(test_files_rust, '82f49a4a7b9585f1.jpg'), labels)
    print("Predicted label:", predicted_label)

# Execute main function
if __name__ == "__main__":
    main()
