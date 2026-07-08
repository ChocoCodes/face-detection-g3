import os
import numpy as np
import tensorflow as tf
import cv2 
from keras_facenet import FaceNet
from deepface import DeepFace

class FaceAligner:
    def __init__(
            self, 
            detector_weights = "../models/face_detection_yunet_2023mar.onnx", 
            recognizer_weights="../models/face_recognition_sface_2021dec.onnx", 
            input_size=(320, 320)
        ):
        self.input_size = input_size 
        self.detector = cv2.FaceDetectorYN.create(detector_weights, "", self.input_size)
        self.recognizer = cv2.FaceRecognizerSF.create(recognizer_weights, "")
    
    def align(self, bgr_image):
        h, w = bgr_image.shape[:2]
        self.detector.setInputSize((w, h))
        retval, faces = self.detector.detect(bgr_image)

        return None if not retval or faces is None else self.recognizer.alignCrop(bgr_image, faces[0])

class MobileNetV2CNN:
    def __init__(self):
        print("Initializing MobileNetV2 CNN...")
        self.input_size = (224, 224)
        self.model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224,224,3),
            pooling='avg'
        )
        self.model.trainable = False
    
    def get_embedding(self, bgr_face):
        rgb_face = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_face, self.input_size)
        tensor = tf.keras.applications.mobilenet_v2.preprocess_input(resized)
        batch = tf.expand_dims(tensor, axis=0)

        embedding = self.model.predict(batch, verbose=0).flatten()
        norm = np.linalg.norm(embedding)
        return embedding if norm == 0 else (embedding / norm)
    
    def similarity(self, vec_a, vec_b):
        """ Euclidean Distance (L2) """
        return float(np.linalg.norm(vec_a - vec_b))

class KerasFaceNet:
    def __init__(self):
        print("[Initializing] FaceNet (Inception-ResNet VGGFace2)...")
        self.embedder = FaceNet()

    def get_embedding(self, bgr_face):
        rgb_face = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
        batch = np.expand_dims(rgb_face, axis=0)
        embedding = self.embedder.embeddings(batch)[0]
        return embedding.flatten()

    def similarity(self, vec_a, vec_b):
        """ Bounded Normalized Euclidean Distance (L2) """
        return float(np.linalg.norm(vec_a - vec_b))

class ArcFace:
    def __init__(self):
        print("[Initializing] ArcFace (Geodesic Angular Manifold)...")
        self.model_name = "ArcFace"
    
    def get_embedding(self, bgr_face):
        result = DeepFace.represent(
            img_path=bgr_face,
            model_name=self.model_name,
            detector_backend="skip",
            enforce_detection=False
        )
        return np.array(result[0]['embedding'])

    def similarity(self, vec_a, vec_b):
        """ 
        Angular Cosine Distance: 1.0 - Cosine Similarity.
        Maps perfectly between 0.0 (identical) and 1.0 (perpendicular).
        """
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        cosine_similarity = dot_product / (norm_a * norm_b)
        return float(1.0 - cosine_similarity)
    
class SFace:
    def __init__(self, w_recognizer="../models/face_recognition_sface_2021dec.onnx"):
        print("[Initializing] SFace via OpenCV DNN Runtime Engine...")
        self.recognizer = cv2.FaceRecognizerSF.create(w_recognizer, "")

    def get_embedding(self, bgr_face):
        """ Image color space should be BGR. """
        embedding = self.recognizer.feature(bgr_face)
        return embedding.flatten()

    def similarity(self, vec_a, vec_b):
        """ 
        Natively leverages OpenCV's SFace Match metrics.
        Returns the optimized NormL2 distance matching 1.128 LFW standard threshold.
        """
        v1 = vec_a.reshape(1, -1).astype(np.float32)
        v2 = vec_b.reshape(1, -1).astype(np.float32)
        return float(self.recognizer.match(v1, v2, cv2.FaceRecognizerSF_FR_NORM_L2))