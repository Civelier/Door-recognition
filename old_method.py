import os
from time import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
from colorama import init, Fore, Back, Style


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Initialize colorama
init(autoreset=True)

SIZE=(256, 256)



class ColoredConsoleHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            level = record.levelname

            if level == 'DEBUG':
                color = Fore.BLUE
            elif level == 'INFO':
                color = Fore.GREEN
            elif level == 'WARNING':
                color = Fore.YELLOW
            elif level == 'ERROR':
                color = Fore.RED
            elif level == 'CRITICAL':
                color = Fore.WHITE + Back.RED + Style.BRIGHT
            else:
                color = ''

            formatted_msg = f"{color}{msg}{Style.RESET_ALL}"
            print(formatted_msg)
        except Exception:
            self.handleError(record)

# Create a logger
log_pre = logging.getLogger('Preprocess')
log_pre.setLevel(logging.DEBUG)
log_main = logging.getLogger('Main')
log_main.setLevel(logging.DEBUG)

# Create and configure the custom colored console handler
colored_console_handler = ColoredConsoleHandler()
colored_console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s [%(levelname)s] %(message)s')
colored_console_handler.setFormatter(formatter)

# Add the custom handler to the logger
log_pre.addHandler(colored_console_handler)
log_main.addHandler(colored_console_handler)

# Define the label-to-color mapping
label_to_color = {
    'door': (0, 0, 128),
    'door2': (0, 128, 0),
    'background': (0, 0, 0),
}

# Define the paths to the Images and Annotations directories
images_dir = 'Images/'
annotations_dir = 'Annotations/'

# Define the checkpoint directory
checkpoint_dir = 'checkpoints/'

# Create the checkpoint directory if it doesn't exist
os.makedirs('cache/', exist_ok=True)

# Function to load and preprocess the dataset
def load_and_preprocess_data(images_dir, annotations_dir):
    log_pre.info("Loading and preprocessing.")
    image_data = []
    labels = []
    one_hot_labels = []
    
    count = len(os.listdir(images_dir))
    last_percent = -1
    num_classes = len(label_to_color)  # Calculate the number of classes based on the mapping

    for i, filename in enumerate(os.listdir(images_dir)):
        percent = i * 100 // count
        if last_percent != percent:
            log_pre.info(f"[{percent}%] {i}/{count}.")
            last_percent = percent
        if filename.endswith('.png'):
            if '(' in filename:
                log_pre.warn(f"Skipped {filename}.")
                continue
            image_path = os.path.join(images_dir, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, SIZE)

            annotation_filename = filename.replace('.png', '.png')
            annotation_path = os.path.join(annotations_dir, annotation_filename)
            annotation = cv2.imread(annotation_path)
            annotation = cv2.resize(annotation, SIZE)

            label_mask = np.zeros_like(annotation, dtype=np.uint8)
            for label, color in label_to_color.items():
                label_mask[np.all(annotation == color, axis=-1)] = list(label_to_color.keys()).index(label)

            # cv2.imshow("Debug", label_mask*255)
            # cv2.waitKey(1)
            # while True:
            #     k = cv2.waitKey(100)
            #     if k == ord('d'):
            #  
            one_hot_label = np.zeros(label_mask.shape[:2] + (num_classes,), dtype=np.uint8)
            for i, (_, color) in enumerate(label_to_color.items()):
                mask = np.all(label_mask == color, axis=-1)
                one_hot_label[..., i] = mask
            one_hot_labels.append(one_hot_label)

            image_data.append(image)
            labels.append(label_mask)

    image_data = np.array(image_data)
    labels = np.array(labels)

    log_pre.info("Done!")
    return image_data, labels, np.array(one_hot_labels)

# Function to create and compile the model
def create_model(input_shape, num_classes):
    log_main.info("Creating model.")
    model = keras.Sequential([
        # Add your convolutional layers, pooling layers, and fully connected layers here
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')  # Adjust the number of units to match the number of classes
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    log_main.info("Model created!")
    return model

def create_segmentation_model(input_shape):
    log_main.info("Creating model.")
    inputs = keras.layers.Input(shape=input_shape)
    
    # Encoder
    conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    # Add more encoder layers as needed

    # Decoder
    up1 = keras.layers.UpSampling2D(size=(2, 2))(pool2)
    concat1 = keras.layers.Concatenate()([conv2, up1])
    conv3 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(concat1)
    up2 = keras.layers.UpSampling2D(size=(2, 2))(conv3)
    concat2 = keras.layers.Concatenate()([conv1, up2])
    segmentation_output = keras.layers.Conv2D(3, 1, activation='softmax')(concat2)  # Output with 2 channels (door and background)

    model = keras.models.Model(inputs=inputs, outputs=segmentation_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    log_main.info("Model created!")
    return model

# Load or create the dataset
if os.path.exists('cache/image_data.npy') and os.path.exists('cache/labels.npy') and os.path.exists('cache/one_hot_labels.npy'):
    log_main.info("Preprocessing already done, loading from 'cache/image_data.npy' and 'cache/labels.npy'.")
    image_data = np.load('cache/image_data.npy')
    labels = np.load('cache/labels.npy')
    one_hot_labels = np.load('cache/one_hot_labels.npy')
else:
    image_data, labels, one_hot_labels = load_and_preprocess_data(images_dir, annotations_dir)
    np.save('cache/image_data.npy', image_data)
    np.save('cache/labels.npy', labels)
    np.save('cache/one_hot_labels.npy', one_hot_labels)
    log_main.info("Saved as 'cache/image_data.npy', 'cache/labels.npy' and 'cache/one_hot_labels.npy'.")

# Split the dataset into training and testing sets
log_main.info("Splitting dataset.")
from sklearn.model_selection import train_test_split
train_images, test_images, train_labels, test_labels = train_test_split(
    image_data, labels, test_size=0.2, random_state=42
)

# Create or load the model
input_shape = (*SIZE, 3)  # Adjust the input shape to match your images
num_classes = len(label_to_color)  # Calculate the number of classes based on the mapping
# model = create_model(input_shape=input_shape, num_classes=num_classes)
model = create_segmentation_model(input_shape)

# Define callbacks, including a ModelCheckpoint callback
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    'cache/model_checkpoint.h5',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)
prog_bar = keras.callbacks.ProgbarLogger()

# Train the model
log_main.info("Training (this may take a while).")
train_start = time()
model.fit(train_images, train_labels, epochs=100, validation_data=(test_images, test_labels), callbacks=[checkpoint_callback, prog_bar])
train_end = time()
log_main.info(f"Done in {round(train_end-train_start, 2)}s.")

# Save the final trained model
model.save('cache/final_model.h5')
log_main.info("Saved at 'cache/final_model.h5'")

# You can stop the program here if needed and run it again to continue

# Load the trained model
loaded_model = keras.models.load_model('cache/final_model.h5')

# Use the trained model for inference
# ...

# Continue with any additional processing or tasks