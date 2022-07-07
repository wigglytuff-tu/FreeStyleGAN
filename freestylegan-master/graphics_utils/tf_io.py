import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO

#==============================================

def writeImageToTensorboard(img, writer, it, res=None, tag="images"):
    img = Image.fromarray((img * 255).astype(np.uint8))
    if res is not None:
        img = img.resize(res, Image.ANTIALIAS)
    output = BytesIO()
    img.save(output, format='PNG')
    img_sum = tf.Summary.Image(encoded_image_string=output.getvalue())
    output.close()
    img_sum_value = tf.Summary.Value(tag=tag, image=img_sum)
    writer.add_summary(tf.Summary(value=[img_sum_value]), it)