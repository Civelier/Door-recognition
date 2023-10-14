import multiprocessing
import os
from pathlib import Path
from unittest.main import MODULE_EXAMPLES
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from cinput import *
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
import pandas as pd
import logging
from colorama import init, Fore, Back, Style
from tqdm import tqdm, tqdm_gui

import tkinter as tk
from tkinter import filedialog

metrics = tf.keras.metrics
losses = tf.keras.losses
optimizers = tf.keras.optimizers

# Constants
NUM_WORKERS = multiprocessing.cpu_count()  # Number of CPU cores to use for parallel processing
SIZE = (128,128)
BATCH_SIZE = 128
BUFFER_SIZE = 1000
EPOCHS = 50
VAL_SUBSPLITS = 5
MODEL_PATH = 'cache/final_model.h5'
DATASET_CACHE_PATH = 'cache/dataset'
DATASET_PATH = 'deepdoors2'
IMAGES_PATH = 'deepdoors2/image'
MASKS_PATH = 'deepdoors2/segmentation_mask'
CACHE_PATH = "cache/benchmark"
IS_REMOTE = True
# IS_REMOTE = False

LABELS = {
    1: (0,0,0),
    2: (0,0,128),
    3: (0,128,0),
}
OUTPUT_CLASSES = 1
# OUTPUT_CLASSES = len(LABELS)

# Initialize colorama
init(autoreset=True)

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

# Data
model = None
normalization_layer = tf.keras.layers.Rescaling(1./255)
sample_image = None
sample_mask = None
test_batches = None
train_batches = None
train = None
val = None


def load_image_from_file(image_path):
    # Load and decode image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    return image


# Recreate the 'info' dictionary manually
info = {
    'name': 'DeepDoors2',
    'description': 'Segmentation dataset of doors',
    'homepage': 'https://github.com/gasparramoa/DeepDoors2',
    'data_url': 'https://drive.google.com/drive/folders/1SxVKeJ9RBcoJXHSHw-LWaLGG07BZT-b5?usp=sharing',
    'version': 'Dataset version number or date',

    'splits': {
        'train': {
            'name': 'train',
            'num_examples': 3000,  # Total number of examples in your dataset
        },
        # Add splits for validation, test, or other subsets if applicable
    },

    'features': {
        'image': {
            'shape': (640, 480, 3),  # Replace with actual image dimensions
            'dtype': tf.uint8,  # Replace with the data type of your images
        },
        'segmentation_mask': {
            'shape': (640, 480, 3),  # Replace with actual mask dimensions
            'dtype': tf.uint8,  # Replace with the data type of your masks
        },
        'label': {
            'num_classes': 3,  # Number of segmentation classes (door, otherdoor, background)
        },
    },
}

TRAIN_LENGTH = int(3000*0.8)
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

# Core functions
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    # input_mask 
    return input_image, input_mask

def load_image(image):
    return tf.cast(image, tf.float32) / 255.0

def load_mask(image):
    return image - 1


def test(*args, **kwargs):
    return None

def display1(display_list, output="display.png", forceSave=False):
    fig = plt.figure(figsize=(15, 15))

    title = []
    for i in range(len(display_list)):
        title.append(f"Image {i}")

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    if IS_REMOTE or forceSave:
        fig.savefig(output)
    else:
        plt.show()
    

def display(display_list, output="display.png", forceSave=False):
    fig = plt.figure(figsize=(15, 15))

    title = ["Input Image", "True Mask"]
    for i in range(len(display_list) -2):
        title.append(f"Predicted Mask{i}")

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    if IS_REMOTE or forceSave:
        fig.savefig(output)
    else:
        plt.show()

def load_and_preprocess_data():
    count = len(os.listdir(IMAGES_PATH))
    x_list = []
    y_list = []
    for i, filename in tqdm(enumerate(os.listdir(IMAGES_PATH)), "Preprocessing", count, unit='images'):
        if filename.endswith('.png'):
            if '(' in filename:
                log_pre.warn(f"Skipped {filename}.")
                continue
            img_path = os.path.join(IMAGES_PATH, filename)
            mask_path = os.path.join(MASKS_PATH, filename)
    
    
            img_file = tf.io.read_file(img_path)
            image = tf.image.decode_png(img_file, channels=3, dtype=tf.uint8)
            mask_file = tf.io.read_file(mask_path)
            mask = tf.image.decode_png(mask_file, channels=3, dtype=tf.uint8)
            
            input_image = tf.image.resize(image, SIZE)
            input_mask = tf.image.resize(mask, SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            
            label_mask = np.ones((*SIZE, 1), dtype=np.bool_)
            # for id, label in LABELS.items():
            #     label_mask[np.all(input_mask == label, axis=-1)] = id
            label_mask[np.all(input_mask == (0,0,0), axis=-1)] = False

            input_image, input_mask = normalize(input_image, input_mask)
            # display1([input_image, input_mask, label_mask])

            x_list.append(input_image)
            y_list.append(label_mask)
    return x_list, y_list

    


# Augmentation
# flip both image and mask identically
def flip_hori(img, mask):
    img = tf.image.flip_left_right(img)
    mask = tf.image.flip_left_right(mask)
    return img, mask


    
def create_mask(pred_mask):
    return (pred_mask > 0.5).astype(np.uint8)[0]
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset=None, models=[model], num=1, output="display.png", forceSave=False):
    if dataset:
        for image, mask in dataset.take(num):
            pred_masks = []
            for m in models:
                pred_masks.append(create_mask(m.predict(image)))
            display([image[0], mask[0], *pred_masks], output=output, forceSave=forceSave)
    else:
        for m in models:
            pred_mask = (create_mask(m.predict(sample_image[tf.newaxis, ...])))
            display([sample_image, sample_mask, pred_mask], output=output, forceSave=forceSave)

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels



VALIDATION_STEPS = (3000 - TRAIN_LENGTH) // BATCH_SIZE

def load(*args):
    global sample_image
    global sample_mask
    global test_batches
    global train_batches
    global train
    global val
    p = argparse.ArgumentParser('load')
    p.add_argument('-f', '--force', action="store_true", help="force reloads the dataset")
    def onexit(*args, **kwargs):
        global on_exit
        on_exit = True
    p.exit = onexit
    a = p.parse_args(args)
    if on_exit:
        return
    
    ld = not os.path.exists(f"{DATASET_CACHE_PATH}/train")
    if a.force:
        ld = True

    if ld:
        x, y = load_and_preprocess_data()
        
        for i in range(len(x)):
            xx, yy = flip_hori(x[i], y[i])
            x.append(xx)
            y.append(yy)

        train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.2, random_state=0)

        # develop tf Dataset objects
        train_x = tf.data.Dataset.from_tensor_slices(train_x)
        val_x = tf.data.Dataset.from_tensor_slices(val_x)

        train_y = tf.data.Dataset.from_tensor_slices(train_y)
        val_y = tf.data.Dataset.from_tensor_slices(val_y)

        # zip images and masks
        train = tf.data.Dataset.zip((train_x, train_y))
        val = tf.data.Dataset.zip((val_x, val_y))
        train.save(f"{DATASET_CACHE_PATH}/train")
        val.save(f"{DATASET_CACHE_PATH}/val")
    else:
        train = tf.data.Dataset.load(f"{DATASET_CACHE_PATH}/train")
        val = tf.data.Dataset.load(f"{DATASET_CACHE_PATH}/val")
    
    if train_batches == None or ld:
        
        train_batches = (
            train.cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .repeat()
            # .map(Augment())
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        test_batches = val.batch(BATCH_SIZE)

        


def sample(*args):
    global sample_image
    global sample_mask
    p = argparse.ArgumentParser('sample')
    def onexit(*args, **kwargs):
        global on_exit
        on_exit = True
    p.exit = onexit
    a = p.parse_args(args)
    if on_exit:
        return
    
    # Dependency
    load()
    for images, masks in test_batches.shuffle(BUFFER_SIZE).take(1):
        sample_image, sample_mask = images[0], masks[0]
        display([sample_image, sample_mask])
    
    

def unet_model1(output_channels: int):
    
    # base_model = tf.keras.applications.MobileNetV2(
    #     input_shape=[128, 128, 3], include_top=False
    # )
    base_model = tf.keras.applications.DenseNet121(
        input_shape=[128, 128, 3], include_top=False, weights='imagenet'
    )

    # # Use the activations of these layers
    layer_names = ['conv1/relu', # size 64*64
                'pool2_relu',  # size 32*32
                'pool3_relu',  # size 16*16
                'pool4_relu',  # size 8*8
                'relu'        # size 4*4
                ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    ]

    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2, padding="same"
    )  # 64x64 -> 128x128

    x = last(x)
    
    
    m = tf.keras.Model(inputs=inputs, outputs=x)

    # m.compile(
    #     optimizer="adam",
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=[tf.keras.metrics.CategoricalAccuracy(), ],
    # )
    # m.compile(
    #     optimizer=tf.optimizers.Adam(),
    #     loss=tf.keras.losses.MeanAbsolutePercentageError(),
    #     metrics=[
    #         tf.metrics.mse,
    #         # No good:
    #         # tf.metrics.CategoricalCrossentropy()
    #     ])


    return m

def unet_model2(output_channels: int):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=[128, 128, 3], include_top=False
    )

    # Use the activations of these layers
    layer_names = [
        "block_1_expand_relu",  # 64x64
        "block_3_expand_relu",  # 32x32
        "block_6_expand_relu",  # 16x16
        "block_13_expand_relu",  # 8x8
        "block_16_project",  # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    ]
    
    
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2, padding="same"
    )  # 64x64 -> 128x128

    x = last(x)

    m = tf.keras.Model(inputs=inputs, outputs=x)

    return m

def unet_model3(output_channels: int):
    #Build the model
    inputs = tf.keras.layers.Input((*SIZE, 3))
    # s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    #Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    m = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return m

on_exit = False
def create_model(*args):
    global model
    global on_exit
    on_exit = False
    p = argparse.ArgumentParser("create_model")
    p.add_argument('-o', '--overwrite', action="store_true", help="rebuild the model if it exists")
    def onexit(*args, **kwargs):
        global on_exit
        on_exit = True
    p.exit = onexit
    a = p.parse_args(args)
    if on_exit:
        return
    
    if sample_image is None:
        sample()
    else:
        load()
    
    create = not os.path.exists(MODEL_PATH)
    if a.overwrite:
        create = True
    if create:
        log_main.debug("Creating model.")
        model = unet_model1(output_channels=OUTPUT_CLASSES)
        log_main.debug("Compiling model.")
        model.compile(
            optimizer=optimizers.Adam(),
            # loss=losses.MeanAbsoluteError(),
            loss=losses.BinaryCrossentropy(),
            
            metrics=[
                # metrics.MeanSquaredError(),
                # metrics.CosineSimilarity(),
                # metrics.CategoricalAccuracy(),
                
                'accuracy',
                # metrics.MeanAbsoluteError(),
                # No good:
                # tf.metrics.CategoricalCrossentropy(from_logits=True, axis=1)
        ])
        model.save(MODEL_PATH)
        log_main.info(f"Saved model at {MODEL_PATH}.")
    elif model == None:
        log_main.debug("Loading model.")
        model = tf.keras.models.load_model(MODEL_PATH)
        log_main.info(f"Loaded model from {MODEL_PATH}.")
        

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(models=[model])
        print("\nSample Prediction after epoch {}\n".format(epoch + 1))
        
class FileSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, mdl:tf.keras.Model=model):
        super().__init__()
        self.mdl = mdl
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(models=[self.mdl],output="train_preview.png", forceSave=True)
        print("\nSample Prediction after epoch {}\n".format(epoch + 1))

def draw_model(*args):
    create_model()
    tf.keras.utils.plot_model(model, show_shapes=True)
    log_main.info("Saved model plot to model.png")
    
def train_ai(*args):
    p = argparse.ArgumentParser("train")
    p.add_argument('-d', '--display', action="store_true", help="diplay the prediction at every epoch")
    p.add_argument('-e', '--epochs', type=int, default=EPOCHS, required=False, help="number of epochs to train for")
    p.add_argument('-p', '--plot', action="store_true", help="when training done, plot the history graph")
    def onexit(*args, **kwargs):
        global on_exit
        on_exit = True
    p.exit = onexit
    a = p.parse_args(args)
    if on_exit:
        return
    create_model()
    
    cbs = [tf.keras.callbacks.ProgbarLogger('steps'), FileSaveCallback(model), tf.keras.callbacks.ModelCheckpoint(MODEL_PATH)]
    if a.display:
        cbs.append(DisplayCallback())
    model_history = model.fit(train_batches, epochs=a.epochs,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_batches,
                          callbacks=cbs)
    model.save(MODEL_PATH)
    if a.plot:
        loss = model_history.history["loss"]
        val_loss = model_history.history["val_loss"]

        fig = plt.figure()
        plt.plot(model_history.epoch, loss, "r", label="Training loss")
        plt.plot(model_history.epoch, val_loss, "bo", label="Validation loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.ylim([0, 1])
        plt.legend()
        if IS_REMOTE:
            fig.savefig("display.png")
        else:
            plt.show()

def predict(*args):
    p = argparse.ArgumentParser("predict")
    p.add_argument('-s', '--select', action='store_true', help="select file with dialog")
    p.add_argument('-u', '--url', type=str, help="from url")
    p.add_argument('-c', '--count', type=int, default=1, help="number of images to predict")
    
    def onexit(*args, **kwargs):
        global on_exit
        on_exit = True
    p.exit = onexit
    a = p.parse_args(args)
    if on_exit:
        return
    
    create_model()
    
    if a.select:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Predict from image")
        root.destroy()
        
        img_file = tf.io.read_file(file_path)
        image = tf.image.decode_png(img_file, channels=3, dtype=tf.uint8)
        image = tf.image.resize(image, SIZE)
        image, _ = normalize(image, None)
        pred_mask = (create_mask(model.predict(image[tf.newaxis, ...])))
        display1([image, pred_mask], forceSave=True)
    elif a.url:
        pass
    else:
        show_predictions(train_batches, [model], a.count)
    
    
    
def benchmark_models(*args):
    p = argparse.ArgumentParser("benchmark")
    p.add_argument('-e', '--epochs', type=int, default=EPOCHS, required=False, help="number of epochs to train for")
    p.add_argument('-r', '--retry', action='store_true', help="train the successfull models again without recompiling them")
    def onexit(*args, **kwargs):
        global on_exit
        on_exit = True
    p.exit = onexit
    a = p.parse_args(args)
    if on_exit:
        return
    
    sample()
    
    mdls = [unet_model1, unet_model2, unet_model3]
    
    opts:List[optimizers.Optimizer] = [
        optimizers.Adam,
        optimizers.RMSprop,
        optimizers.SGD,
        optimizers.Adadelta,
        optimizers.Adamax,
        optimizers.Ftrl,
    ]
    
    lss:List[losses.Loss] = [
        # losses.KLDivergence(),
        # losses.LogCosh(),
        # losses.CosineSimilarity(),
        # losses.MeanSquaredLogarithmicError(),
        # losses.MeanSquaredError(),
        # losses.CategoricalHinge(),
        losses.MeanAbsolutePercentageError,
        losses.MeanAbsoluteError,
        losses.CategoricalCrossentropy,
        losses.SparseCategoricalCrossentropy,
        losses.Poisson,
        losses.BinaryCrossentropy,
        losses.BinaryFocalCrossentropy,
    ]
    
    mts:List[metrics.Metric] = [
        metrics.MeanSquaredError,
        metrics.Accuracy,
        metrics.CosineSimilarity,
        metrics.CategoricalCrossentropy,
        metrics.Mean,
        metrics.LogCoshError,
        metrics.CategoricalAccuracy,
        metrics.MeanAbsoluteError,
        metrics.MeanTensor,
        metrics.SparseCategoricalAccuracy,
        metrics.BinaryAccuracy,
        metrics.BinaryCrossentropy,
        metrics.BinaryIoU,
        metrics.AUC,
    ]
    
    permutations:List[Tuple(Callable[[int], tf.keras.Model], optimizers.Optimizer, losses.Loss, metrics.Metric, str)] = []
    for i, cm in enumerate(mdls):
        mdp = '/'.join([CACHE_PATH, f"Model{i+1}"])
        for opt in opts:
            opt = opt()
            optp = '/'.join([mdp, opt.name])
            for ls in lss:
                ls = ls()
                lsp = '/'.join([optp, ls.name])
                for mt in mts:
                    mt = mt()
                    mtp = '/'.join([lsp, mt.name])
                    os.makedirs(mtp, exist_ok=True)
                    if os.path.exists(f"{mtp}/failed.log"):
                        continue
                    if a.retry and os.path.exists(f"{mtp}/trained.h5"):
                        os.remove(f"{mtp}/trained.h5")
                    permutations.append((cm, opt, ls, mt, mtp))
    count = len(permutations)
    
    def create_model_from(modelType:Callable[[int], tf.keras.Model], optimizer:optimizers.Optimizer, loss:losses.Loss, metric:metrics.Metric, path:str):
        modelPath = '/'.join([path,'model.h5'])
        compiledPath = '/'.join([path,'compiled.h5'])
        trainedPath = '/'.join([path,'trained.h5'])
        plotPath = '/'.join([path,'plot.png'])
        resultPath = '/'.join([path,'result.png'])
        failPath = '/'.join([path, 'failed.log'])
        name = path.removeprefix(f"{CACHE_PATH}/")
        
        if os.path.exists(failPath):
            log_main.warning(f"Skipping {name}")
            return None
        
        def crt():
            mdl:tf.keras.Model = modelType(OUTPUT_CLASSES)
            mdl.save(modelPath)
            return mdl
        
        def comp(mdl:tf.keras.Model):
            try:
                mdl.compile(optimizer=optimizer, loss=loss, metrics=[metric])
                mdl.save(compiledPath)
            except KeyError or ValueError as e:
                p = Path(failPath)
                p.touch()
                with p.open('w') as wr:
                    print('Failed at compile', file=wr, flush=True)
                    log_main.error(f"Failed to compile {name}")
                    print(str(e), file=wr, flush=True)
                return None
            return mdl
            
        def trn(mdl:tf.keras.Model):
            cbs = [tf.keras.callbacks.ProgbarLogger('steps'), FileSaveCallback(mdl)]
            try:
                model_history = mdl.fit(train_batches, epochs=a.epochs,
                                    steps_per_epoch=STEPS_PER_EPOCH,
                                    validation_steps=VALIDATION_STEPS,
                                    validation_data=test_batches,
                                    callbacks=cbs)
                mdl.save(trainedPath)
                h_loss = model_history.history["loss"]
                val_loss = model_history.history["val_loss"]

                fig = plt.figure()
                plt.plot(model_history.epoch, h_loss, "r", label="Training loss")
                plt.plot(model_history.epoch, val_loss, "bo", label="Validation loss")
                plt.title("Training and Validation Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss Value")
                plt.ylim([0, 1])
                plt.legend()
                fig.savefig(plotPath)
                
                show_predictions(models=[mdl], output=resultPath)
            except KeyError or ValueError as e:
                p = Path(failPath)
                p.touch()
                with p.open('w') as wr:
                    print('Failed at training', file=wr, flush=True)
                    log_main.error(f"Failed to train {name}")
                    print(str(e), file=wr, flush=True)
                return None
            
            
        paths = [modelPath, compiledPath, trainedPath]
        checks = list([os.path.exists(p) for p in paths])
        actions = [crt, comp, trn]
        
        mdl = None
        for i, pth, check, action in tqdm(zip(range(len(paths)), paths, checks, actions), f"Benchmarking {name}", len(paths), unit='action'):
            if check: # Model path found
                if i < len(paths)-1:
                    if checks[i + 1]:
                        continue
                else:
                    continue
                mdl = tf.keras.models.load_model(pth)
            else: # Model path not found, execute action
                if mdl == None: # Before creating model, there are no arguments
                    mdl = action()
                else:
                    mdl = action(mdl)
                if mdl == None: # Action failed
                    return None
    
    
    def cleanup():
        unique:List[str]=[]
        for mdln in os.listdir(CACHE_PATH):
            mdlp = f"{CACHE_PATH}/{mdln}"
            for optn in os.listdir(mdlp):
                optp = f"{mdlp}/{optn}"
                for lsn in os.listdir(optp):
                    lsp = f"{optp}/{lsn}"
                    for mtn in os.listdir(lsp):
                        mtp = f"{lsp}/{mtn}"
                        unique.append(mtp)
        for p in tqdm(unique, "Cleaning up", len(unique), unit='dir'):
            pp = [permut[-1] for permut in permutations]
            if p not in pp:
                for f in os.listdir(p):
                    os.remove(f"{p}/{f}")
                os.removedirs(p)
    
    for inf in tqdm(permutations, "Benchmarking", count):
        create_model_from(*inf)
    cleanup()
        
def benchmark_results(*args):
    p = argparse.ArgumentParser("results")
    p.add_argument('-a', '--all', action='store_true', help="show all permutations, even ones that failed")
    p.add_argument('-p', '--predict', action='store_true', help="predict with a new sample image")
    def onexit(*args, **kwargs):
        global on_exit
        on_exit = True
    p.exit = onexit
    a = p.parse_args(args)
    if on_exit:
        return
    shape = (0,0,0)
    sample()
    
    
    unique:List[Tuple(str,str,str,str, cv2.Mat)]=[]
    choices:Dict[str, Dict[str, Dict[str, Dict[str, cv2.Mat]]]] = {}
    for mdln in os.listdir(CACHE_PATH):
        mdlp = f"{CACHE_PATH}/{mdln}"
        mdlns = {}
        for optn in os.listdir(mdlp):
            optp = f"{mdlp}/{optn}"
            optns = {}
            for lsn in os.listdir(optp):
                lsp = f"{optp}/{lsn}"
                lsns = {}
                for mtn in os.listdir(lsp):
                    mtp = f"{lsp}/{mtn}"
                    im = cv2.imread(f"{mtp}/result.png")
                    if im is not None:
                        shape = im.shape
                        unique.append((mdln, optn, lsn, mtn, mtp))
                    elif not a.all:
                        continue
                    lsns[mtn] = im
                if len(lsns) == 0 and not a.all:
                    continue
                optns[lsn] = lsns
            if len(optns) == 0 and not a.all:
                continue
            mdlns[optn] = optns
        if len(mdlns) == 0 and not a.all:
            continue
        choices[mdln] = mdlns
    
    if a.predict:
        for u in tqdm(unique, "Predicting", len(unique)):
            pth = u[-1]
            mdl:tf.keras.Model = tf.keras.models.load_model(f"{pth}/trained.h5")
            show_predictions(models=[mdl], output=f"{pth}/result.png")
                    
    def show_result(mdl:str, opt:str, ls:str, mt:str, choices:Dict[str, Dict[str, Dict[str, Dict[str, cv2.Mat]]]], selected:int):
        img = choices[mdl][opt][ls][mt]
        font = cv2.FONT_HERSHEY_SIMPLEX 
        # fontScale 
        fontScale = 1
        # Line thickness of 2 px 
        thickness = 2
        color = (0, 255, 0) 
        if img is None:
            img = np.full(shape, 255, np.uint8)
            
            # Blue color in BGR 
            color = (0, 0, 255) 
            
            
            # Using cv2.putText() method 
            img = cv2.putText(img, 'FAILED', (50,100), font,  
                            fontScale, color, thickness, cv2.LINE_AA) 
        txt=""
        
        for i, lbl in enumerate([mdl,opt,ls,mt]):
            if selected == i:
                txt+=f"<(wasd)-{lbl}>"
            else:
                txt+=f"|{lbl}|"
        else:
            img = img.copy()
        img = cv2.putText(img, txt, (50, 50), font,  
                        fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow("Preview", img)
        
    def control(choices:Dict[str, Dict[str, Dict[str, Dict[str, cv2.Mat]]]], unique:List[Tuple[str,str,str,str,cv2.Mat]], showAll:bool):
        def inc(v:int, count:int)->int:
            if count == 0:
                return 0
            return (v + 1) % count
        
        def decr(v:int, count:int)->int:
            if count == 0:
                return 0
            if v == 0:
                return count-1
            return v-1
        
        params = [0,0,0,0]
        index = 0
        uid = 0
        while True:
            if showAll:
                mdls = list(choices.keys())
                opts = list(choices[mdls[params[0]]].keys())
                lss = list(choices[mdls[params[0]]][opts[params[1]]].keys())
                mts = list(choices[mdls[params[0]]][opts[params[1]]][lss[params[2]]].keys())
                names = [mdls, opts, lss, mts]
                counts = list([len(ss) for ss in [mdls, opts, lss, mts]])
                
                k = cv2.waitKey(100)
                
                if k == ord('w'):
                    params[index] = inc(params[index], counts[index])
                if k == ord('s'):
                    params[index] = decr(params[index], counts[index])
                if k == ord('d'):
                    index = inc(index, 4)
                if k == ord('a'):
                    index = decr(index, 4)
                if k == ord('q'):
                    cv2.destroyAllWindows()
                    return
                
                values = list([names[i][ii] for i, ii in enumerate(params)])
                
            else:
                k = cv2.waitKey(100)
                
                if k == ord('w'):
                    uid = inc(uid, len(unique))
                if k == ord('s'):
                    uid = decr(uid, len(unique))
                if k == ord('q'):
                    cv2.destroyAllWindows()
                    return
                
                values = unique[uid][:4]
            
            show_result(*values, choices=choices, selected=index)
    
    control(choices, unique, a.all)
            
        
    
menus:Menu = {
    'load' : load,
    'sample' : sample,
    'create' : create_model,
    'plot' : draw_model,
    'train' : train_ai,
    'predict' : predict,
    'benchmark' : benchmark_models,
    'results' : benchmark_results,
}


if __name__ == "__main__":
    menuArgsInput(menus, "Main menu")
    # benchmark_results()
    # benchmark_models('-e 20')