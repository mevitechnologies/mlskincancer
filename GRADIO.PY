from tensorflow.keras.models import load_model
import tensorflow as tf
import keras
import numpy as np
#Prediction
import numpy as np
from keras.utils import load_img, img_to_array
import gradio as gr
from gtts import gTTS
from playsound import playsound
import tensorflow as tf


labels={0:'actinic keratosis',
          1:'basal cell carcinoma',
          2:'dermatofibroma',
          3:'melanoma',
          4:'nevus',
          5:'pigmented benign keratosis',
          6:'seborrheic keratosis',
          7:'squamous cell carcinoma',
          8:'vascular lesion',}

def predict(inp):
    skincancer = tf.keras.models.load_model('model1.h5')
    test_image = inp.reshape(200,200,3) #loading the image
    test_image =img_to_array(test_image) #converting image to array
    test_image = np.expand_dims(test_image, axis = 0) #expanding the dimension
    result = skincancer.predict(test_image) #predicting img
    print('predicted_sign', result)
    predicted_sign = result.argmax(axis=-1)[0]
    print(predicted_sign)
    predicted_sign1 = labels[predicted_sign]
    print(predicted_sign1)
    voice_output = gTTS(text="The predicted class is " +predicted_sign1, lang='en', slow=False)
    voice_output.save("output.mp3")
    playsound('output.mp3')
    return predicted_sign1

output_text = gr.outputs.Textbox(label="Label")
iface = gr.Interface(fn=predict, 
             inputs=gr.Image(shape=(200,200)),
             outputs=output_text,
             interpretation='default',
             title="Skin Cancer Classification",
description="Recognize and Classify the images of Skin cancer Images ")
iface.launch()