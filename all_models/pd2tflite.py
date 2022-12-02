import tensorflow as tf

# Convert the model
saved_model_dir = "/home/cyhuiae/PycharmProjects/tpu_facenet/TPU/all_models/yolov5_ppe/best_saved_model"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('yolov5_ppe/model.tflite', 'wb') as f:
  f.write(tflite_model)
