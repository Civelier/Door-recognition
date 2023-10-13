import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('cache/final_model.h5')
tf.keras.utils.plot_model(model, show_shapes=True)

# Load the image you want to segment
image_path = 'Images/Door0016.png'  # Replace with the path to your image
image = cv2.imread(image_path)

# Preprocess the image (resize to match the model's input shape)
input_shape = (256, 256)
orig_size = image.shape[:2]
image = cv2.resize(image, input_shape)

# Normalize the image (if necessary)
# image = image / 255.0  # Assuming pixel values are in the range [0, 255]

# Make a prediction using the model
prediction = model.predict(np.expand_dims(image, axis=0))

# Assuming your model outputs pixel-wise one-hot encoded labels, reshape the prediction
prediction = prediction.reshape(input_shape + (-1,))

# Convert the prediction to a segmentation mask
segmentation_mask = np.argmax(prediction, axis=-1)

# Define colors for each class (door, door2, background)
class_colors = {
    0: [0, 0, 128],    # door (blue)
    1: [0, 128, 0],    # door2 (green)
    2: [0, 0, 0]       # background (black)
}

# Create a colored overlay based on the segmentation mask
colored_overlay = np.zeros_like(image)
for class_id, color in class_colors.items():
    mask = segmentation_mask == class_id
    colored_overlay[mask] = color

# Overlay the colored segmentation on the original image
alpha = 0.5  # Adjust the alpha blending factor as needed
output_image = cv2.addWeighted(image, 1 - alpha, colored_overlay, alpha, 0)

# Display the original image and the overlay
cv2.imshow('Original Image', image)
cv2.imshow('Segmentation Overlay', output_image)
cv2.imshow('Colored Overlay', colored_overlay)
# output_image = cv2.resize(output_image, orig_size)
cv2.imwrite('result.jpg', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
