import tensorflow as tf
import numpy as np
import cv2

class_labels = {'Healthy': 0, 'Late_blight': 1, 'Early_blight': 2}
labels = ["Healthy","Late_blight","Early_blight"]

#convert tensorflow model to tflite model

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("tflite_model.tflite","wb") as f:
    f.write(tflite_model)

## Inference
# Load TFLite model and allocate tensors.

interpreter = tf.lite.Interpreter(model_path="./tflite_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = cv2.resize(cv2.imread("./test/Early_Blight/042135e2-e126-4900-9212-d42d900b8125___RS_Early.B 8791.JPG"),(224,224))
input_data = np.expand_dims(input_data,axis=0)
print(input_data.shape)

# Convert the input_data to FLOAT32 and normalize it to the range [0, 1]
input_data = input_data.astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(labels[np.argmax(output_data)])