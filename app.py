from flask import Flask, request,jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub
from google.cloud import vision
import numpy as np
import webcolors
import threading
from translate import Translator
from PIL import Image, ImageOps


app = Flask(__name__)

model = load_model('./ml_models/keras_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})

@app.route('/', methods = ['POST'])
def hello_world():
    return jsonify({'Test':'Hello World!'})

# detect currency using the Model created for a single frame
def detect_currency_frame(i,output):
    s = 'image'
    s += str(i)
    img = Image.open(request.files[s])
    img.save(s+'.jpg')
    img = Image.open(s+'.jpg')

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img

    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    
    # Load the image into the array
    data[0] = normalized_image_array
    
    # run the inference
    print('Started predicting')
    prediction = model.predict(data)
    print(prediction)
    print('Predicting done')
    result = np.argmax(prediction)
    print(result)
    switcher = {
                 0:"خمسة جنيهات", 
                 1:"عشرة جنيهات",
                 2:"عشرون جنيه",
                 3:"خمسون جنيه",
                 4:"مائة جنيه",
                 5:"مائتي جنيه"
        }
    s=switcher.get(result, "Not Maching")
    output.append(s)
    print(s)

@app.route('/predict-currency', methods = ['POST'])
def detect_currency():
    print(request.remote_addr)

    # create output list to hold all 4 results of the 4 frames
    output = list()

    for i in range(1, 5):
        detect_currency_frame(i, output)

    # Print all outputs in the console
    print(output)

    # create a map to save count of each result
    results = dict()

    # For loop to count the number of occurrences of each currency if different currencies
    for res in output:
        if res in results.keys():
            results[res] += 1
        else:
            results[res] = 1

    # Return the value with maximum occurrences
    max = -1
    maxRes = ""
    for k in results.keys():
        if results[k] > max:
            max = results[k]
            maxRes = k
    if max < 2:
        for k in results.keys():
            if len(k) > max:
                max = results[k]
                maxRes = k
    result = maxRes
    return jsonify({'value': result})

# function to get the closest color with a name and return the name
def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

# returns the actual name and closest name of a coloro using the previous function and webcolors Library
def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

# function to detect color using google cloud services API for a single frame
def detect_color_frame(i,output):
    client = vision.ImageAnnotatorClient()
    s = 'image'
    s += str(i)
    content = request.files[s].read()
    image = vision.Image(content=content)

    response = client.image_properties(image=image)
    props = response.image_properties_annotation
    print('Properties:')
    # get the color in RGB format, and get its name
    for color in props.dominant_colors.colors:
        print('fraction: {}'.format(color.pixel_fraction))
        print('\tr: {}'.format(color.color.red))
        print('\tg: {}'.format(color.color.green))
        print('\tb: {}'.format(color.color.blue))
        print('\ta: {}'.format(color.color.alpha))
        _, result = get_colour_name((int(color.color.red), int(color.color.green), int(color.color.blue)))
        print(result)
        output.append(result)
        # Break after first iteration because we only need the most dominant color in the picture
        break

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

@app.route('/detect-color', methods = ['POST'])
def detect_color():
    print(request.remote_addr)

    # create jobs array for threading
    jobs = []
    # create list to carry all the outputs
    output = list()
    for i in range(4, 0, -1):
        thread = threading.Thread(target=detect_color_frame(i,output))
        jobs.append(thread)

    # Start all Jobs
    for j in jobs:
        j.start()
    # Make sure all Jobs has finished
    for j in jobs:
        j.join()

    # Print all outputs in the console
    print(output)

    results = dict()
    # For loop to count the number of occurrences of each color if different colors
    for res in output:
        if res in results.keys():
            results[res] += 1
        else:
            results[res] = 1
    # Return the color with maximum occurrences
    max = -1
    maxRes = ""
    for k in results.keys():
        if results[k] > max:
            max = results[k]
            maxRes = k
    # get the color in arabic
    translator = Translator(from_lang="english", to_lang="arabic")
    result = translator.translate(maxRes)
    print(result.replace('color',''))
    result = result.replace('color','')
    return jsonify({'color':result})


# function to apply OCR on a single frame
def detect_text_frame(i,output):
    """Detects text in the file."""

    client = vision.ImageAnnotatorClient()

    # [START vision_python_migration_text_detection]

    s = 'image'
    s += str(i)
    content = request.files[s].read()
    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')
    result = ""
    for text in texts:
        print('\n"{}"'.format(text.description))
        result += text.description
        break

    output.append(result)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
        # [END vision_python_migration_text_detection]
# [END vision_text_detection]

@app.route('/detect-text', methods = ['POST'])
def detect_text():
    print(request.remote_addr)
    # create jobs array for threading
    jobs = []
    # create list to carry all the outputs
    output = list()
    for i in range(4, 0, -1):
        thread = threading.Thread(target=detect_text_frame(i, output))
        jobs.append(thread)
    # Start all Jobs
    for j in jobs:
        j.start()
    # Make sure all Jobs has finished
    for j in jobs:
        j.join()

    # Print all outputs in the console
    print(output)

    results = dict()
    # For loop to count the number of occurrences of each text if different texts
    for res in output:
        if res in results.keys():
            results[res] += 1
        else:
            results[res] = 1
    # Return the text with maximum occurrences
    max = -1
    maxRes = ""
    for k in results.keys():
        if results[k] > max:
            max = results[k]
            maxRes = k
    # if there is no text occurred more than 1 time, then choose the largest text.
    if max < 2:
        for k in results.keys():
            if len(k) > max:
                max = results[k]
                maxRes = k

    result = maxRes
    # Library to detect the language of the extracted text from the picture
    # to use the correct voice from the frontend
    from langdetect import detect
    try:
        lang = detect(result)
        print(lang)
    except:
        lang = 'ar'
    if lang == 'de':
        lang = 'en'
    # count number of words by counting spaces and new lines
    words = len(result.split(' ')) + len(result.split('\n'))
    return jsonify({'extracted':result,'lang':lang, 'words':words})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)