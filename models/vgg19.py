from keras.applications.vgg19 import VGG19, preprocess_input
from scipy.spatial import distance
import keras.utils as image
import numpy as np


class FeatureExtractor:
    def __init__(self):
        self.model = VGG19(weights='imagenet', include_top=False)

    def compare_features(self, f1, f2):
        try:
            if f1 is None or f2 is None:
                print("[VGG19] Uma ou ambas as características são nulas.")
                return None
            return distance.cosine(f1, f2)
        except Exception as e:
            print(f"[VGG19] Erro ao comparar características: {e}")
            return None

    def extract_features(self, img):
        try:
            img = image.load_img(img, target_size=(224, 224))
            img = image.img_to_array(img)
            x = np.expand_dims(img, axis=0)
            x = preprocess_input(x)
            features = self.model.predict(x)
            features = features.flatten()
            return features
        except Exception as e:
            print(f"[VGG19] Erro ao extrair características da imagem: {e}")
            return None
