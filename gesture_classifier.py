import numpy as np
import tensorflow as tf


class GestureClassifier(object):
    def __init__(
        self,
        model_path='./gesture_classifier.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):

        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_details_tensor_index, landmark_list)
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)
        probabilities = result # Convert to probabilities


        result_index = np.argmax(probabilities)
        confidence = probabilities[0][result_index]  # Confidence of predicted gesture


        return result_index, confidence