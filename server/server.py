import numpy as np
from flask import Flask, render_template,request,jsonify,url_for,json
import pickle
from PIL import Image
import gtts
import json
# from playsound import playsound
# import os
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
import requests # to get image from the web
import shutil # to save it locally


app = Flask(__name__, static_folder="static")
model=load_model('model.h5')
@app.route('/',methods=['POST','GET'])
def predict():
  
    if(request.method=="POST"):
        img = Image.open(request.files['imageFile'].stream)
        # img.show()
        # img = load_img("img.jpg")
        # img = img.resize((64, 64))
        img.save("img.jpg")
        # img = img_to_array(img) 
        # result=model.predict(img)
        test_image = tf.keras.utils.load_img('img.jpg', target_size = (64, 64))
        test_image = tf.keras.utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        if result[0][0] == 1:
            prediction = 'Fire Not Detected'
        else:
            prediction = 'Fire Detected'
        
        
        print("Prediction: ",prediction)
        print("Probablity: ",result[0][0])
                    
        # img.save("img.jpeg")
       
    return "Hello from server"

if __name__=='__main__':
    app.run(port=8000,host='0.0.0.0')

