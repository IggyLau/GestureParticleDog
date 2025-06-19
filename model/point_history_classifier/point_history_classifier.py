#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class PointHistoryClassifier(object):
    def __init__(
        self,
        model_path='model/point_history_classifier/point_history_classifier.tflite',
        score_th=0.5,
        invalid_value=0,
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.score_th = score_th
        self.invalid_value = invalid_value
        self.last_confidence = 0.0  # Store confidence from last prediction
        self.last_threshold = 0.0   # Store threshold used for last prediction
        
        # Class-specific thresholds - higher for clockwise/counter-clockwise
        self.class_thresholds = {
            0: 0.3,  # Stop - lower threshold (easier to detect)
            1: 0.8,  # Clockwise - higher threshold (harder to detect)
            2: 0.8,  # Counter-clockwise - higher threshold (harder to detect)
            3: 0.6,
            4: 0.6,
            5: 0.6   # Move - medium threshold
        }

    def __call__(
        self,
        point_history,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([point_history], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))
        self.last_confidence = float(np.squeeze(result)[result_index])  # Store confidence

        # Use class-specific thresholds instead of global threshold
        class_threshold = self.class_thresholds.get(result_index, self.score_th)
        
        if self.last_confidence < class_threshold:
            result_index = self.invalid_value

        self.last_threshold = class_threshold  # Store the threshold used for the last prediction

        return result_index

    def get_confidence(self):
        """Get the confidence score from the last prediction"""
        return self.last_confidence

    def get_last_threshold(self):
        """Get the threshold used for the last prediction"""
        return self.last_threshold
