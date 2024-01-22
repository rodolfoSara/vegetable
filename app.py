import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st


data_cat = ['apple', 'banana',  'beetroot',  'bell pepper',  'cabbage', 'capsicum', 'carrot', 'cauliflower',
 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic',  'ginger', 'grapes', 'jalepeno',  'kiwi',  'lemon',  'lettuce',
 'mango',  'onion',  'orange',  'paprika',  'pear',  'peas',  'pineapple',  'pomegranate',  'potato',  'raddish',  'soy beans',
 'spinach',  'sweetcorn',  'sweetpotato',  'tomato',  'turnip',  'watermelon']

model = load_model('C:/Users/Usuario/Desktop/new data/tutos individuales/verduras/.ipynb_checkpoints/verduras-checkpoint.ipynb')


image = 'banana.jpg'
image = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
img_arr = tf.keras.utils.array_to_img(image)
img_bat = tf.expand_dims(img_arr, 0)

predict = model.predict(img_bat)


score = tf.nn.softmax(predict)

st.image(image)
st.write('Veg/Fruit in image is {} ' +data_cat[np.argmax(score)])
st.write('with accuracy of ' + np.max(score)*100)
