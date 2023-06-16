# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import cv2
import os


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        model = tf.saved_model.load("./magenta_arbitrary-image-stylization-v1-256_2")

    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
        scale: float = Input(
            description="Factor to scale image by", ge=0, le=10, default=1.5
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.style_transfer(processed_image, scale)
        return image

    def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0): 
        """Return a sharpened version of the image, using an unsharp mask."""
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        
        return sharpened

    def style_transfer(content_img, style_image, style_weight = 1, content_weight = 1, style_blur=False):
        content_img = unsharp_mask(content_img,amount=1)
        content_img = tf.image.resize(tf.convert_to_tensor(content_img,tf.float32)[tf.newaxis,...] / 255.,(512,512),preserve_aspect_ratio=True)
        style_img = tf.convert_to_tensor(style_image,tf.float32)[tf.newaxis,...] / 255.
        if style_blur:
            style_img=  tf.nn.avg_pool(style_img, [3,3], [1,1], "VALID")
        style_img = tf.image.adjust_contrast(style_img, style_weight)        
        content_img = tf.image.adjust_contrast(content_img,content_weight)     
        content_img = tf.image.adjust_saturation(content_img, 2)        
        content_img = tf.image.adjust_contrast(content_img,1.5)        
        stylized_img = model(content_img, style_img)[0]
        
        return Image.fromarray(np.uint8(stylized_img[0]*255))


