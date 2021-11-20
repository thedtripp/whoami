from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import tensorflow as tf
import numpy as np

tfVersion=tf.version.VERSION.replace(".", "")# can be used as savename
print(tf.version.VERSION)

# pretrained_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# converter = tf.lite.TFLiteConverter.from_keras_model(pretrained_model)
converter = tf.lite.TFLiteConverter.from_keras_model_file('my_model.h5')


folderpath='./All_croped_images/'

def prepare(img):
    img = np.expand_dims(img,0).astype(np.float32)
    img = preprocess_input(img, version=2)
    return img
      
repDatagen=tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=prepare)
datagen=repDatagen.flow_from_directory(folderpath,target_size=(224,224),batch_size=1)

def representative_dataset_gen():
  for _ in range(10):
    img = datagen.next()
    yield [img[0]]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.experimental_new_converter = True

converter.target_spec.supported_types = [tf.int8]
quantized_tflite_model = converter.convert()

open('quant_model.tflite' , "wb").write(quantized_tflite_model)
