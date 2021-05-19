from flask import Flask
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
import numpy as np


Classifier=Sequential()

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/predict-setup')
def predict():
    model = load_model('model_Classifier.h5')

    img = image.load_img('./testpic.jpg', target_size=(224, 224))
    test_image = image.img_to_array(img)
    test_image = np.expand_dims(test_image, axis=0)
    # result = Classifier.predict(test_image)
    result=model.predict(test_image)
    print(result)
# 10 100 20 200 5 50
    return 'done'

if __name__ == '__main__':
    app.run()
