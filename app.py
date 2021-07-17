from flask import Flask, request,jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from google.cloud import vision
import io
import numpy as np
import base64
import webcolors
import threading
from translate import Translator


#Classifier=Sequential()

app = Flask(__name__)

model = load_model('./ml_models/model_Classifier.h5')

@app.route('/', methods = ['POST'])
def hello_world():
    return jsonify({'Test':'Hello World!'})

@app.route('/predict-currency')
def predict():
    print(request.remote_addr)
    img = image.load_img('./testpic.jpg', target_size=(224, 224))
    test_image = image.img_to_array(img)
    test_image = np.expand_dims(test_image, axis=0)
    # result = Classifier.predict(test_image)
    result=model.predict(test_image)
    print(result)
# 10 100 20 200 5 50
    if result[0][0] == 1:
        result = '10'
    elif result[0][1] == 1:
        result = '100'
    elif result[0][2] == 1:
        result = '20'
    elif result[0][3] == 1:
        result = '200'
    elif result[0][4] == 1:
        result = '5'
    elif result[0][5] == 1:
        result = '50'
    return jsonify({'value':result})

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

def detect_color_frame(i,output):
    client = vision.ImageAnnotatorClient()
    s = 'image'
    s += str(i)
    content = request.files[s].read()
    image = vision.Image(content=content)

    response = client.image_properties(image=image)
    props = response.image_properties_annotation
    print('Properties:')

    for color in props.dominant_colors.colors:
        print('fraction: {}'.format(color.pixel_fraction))
        print('\tr: {}'.format(color.color.red))
        print('\tg: {}'.format(color.color.green))
        print('\tb: {}'.format(color.color.blue))
        print('\ta: {}'.format(color.color.alpha))
        result = ""
        _, result = get_colour_name((int(color.color.red), int(color.color.green), int(color.color.blue)))
        print(result)
        output.append(result)
        break

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

@app.route('/detect-color', methods = ['POST'])
def detect_color():
    print(request.remote_addr)
    jobs = []
    output = list()
    for i in range(1,5):
        thread = threading.Thread(target=detect_color_frame(i,output))
        jobs.append(thread)

    for j in jobs:
        j.start()
    for j in jobs:
        j.join()

    print(output) #Print all outputs in the console

    results = dict()
    # For loop to count the number of occurences of each color if different colors
    for res in output:
        if res in results.keys():
            results[res] += 1
        else:
            results[res] = 1
    # Return the color with maximum occurences
    max = -1
    maxRes = ""
    for k in results.keys():
        if results[k] > max:
            max = results[k]
            maxRes = k
    # get the color in arabic
    translator = Translator(from_lang="english", to_lang="arabic")
    result = translator.translate(maxRes)
    return jsonify({'color':result})


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
        vertices = (['({},{})'.format(vertex.x, vertex.y)
                     for vertex in text.bounding_poly.vertices])
        print('bounds: {}'.format(','.join(vertices)))
    output.append(result)
    # response = client.document_text_detection(image=image)
    # for page in response.full_text_annotation.pages:
    #     for block in page.blocks:
    #         print('\nBlock confidence: {}\n'.format(block.confidence))
    #
    #         for paragraph in block.paragraphs:
    #             print('Paragraph confidence: {}'.format(
    #                 paragraph.confidence))
    #
    #             for word in paragraph.words:
    #                 word_text = ''.join([
    #                     symbol.text for symbol in word.symbols
    #                 ])
    #                 print('Word text: {} (confidence: {})'.format(
    #                     word_text, word.confidence))
    #                 # result = word_text
    #
    #                 for symbol in word.symbols:
    #                     print('\tSymbol: {} (confidence: {})'.format(
    #                         symbol.text, symbol.confidence))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

@app.route('/detect-text', methods = ['POST'])
def detect_text():
    print(request.remote_addr)
    jobs = []
    output = list()
    for i in range(1, 5):
        thread = threading.Thread(target=detect_text_frame(i, output))
        jobs.append(thread)

    for j in jobs:
        j.start()
    for j in jobs:
        j.join()

    print(output)  # Print all outputs in the console

    results = dict()
    # For loop to count the number of occurences of each color if different colors
    for res in output:
        if res in results.keys():
            results[res] += 1
        else:
            results[res] = 1
    # Return the color with maximum occurences
    max = -1
    maxRes = ""
    for k in results.keys():
        if len(k) > max:
            max = results[k]
            maxRes = k

    result = maxRes
    from langdetect import detect
    lang = detect(result)
    print(lang)
    return jsonify({'extracted':result,'lang':lang})
    # [END vision_python_migration_text_detection]
# [END vision_text_detection]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005,threaded=False,processes=4)