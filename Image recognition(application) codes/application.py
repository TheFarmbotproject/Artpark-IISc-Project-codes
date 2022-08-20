# -*- coding: utf-8 -*-
"""
Created on Fri June 10 19:06:57 2022
Flask application for local Deployment

"""
from flask import Flask, render_template, session, redirect, url_for, jsonify, make_response
from flask_wtf import FlaskForm
import numpy as np
import tensorflow as tf
from wtforms import SubmitField
from flask_wtf.file import FileField, FileRequired, FileAllowed
from flask_uploads import UploadSet, IMAGES, configure_uploads, patch_request_class
import cv2
import os

basedir = os.path.abspath(os.path.dirname(__file__))

application = Flask(__name__)
application.config['SECRET_KEY'] = 'mysecretkey'
application.config['UPLOADED_PHOTOS_DEST'] = os.path.join(basedir, 'uploads')  # you'll need to create a folder named uploads

photos = UploadSet('photos', IMAGES)
configure_uploads(application, photos)
patch_request_class(application)  # set maximum file size, default 16MB


# Create a WTForm Class
# We will get image input from the user
class ImageForm(FlaskForm):
    photo = FileField('Leaf Image', validators=[FileRequired('File was empty!'), FileAllowed(photos, 'Images Only!')])
    submit = SubmitField('Submit')


def dise_class(tflite_path, image, img_path, url):
    # Classes declared
    classes = ["Healthy", "Boll Rot", "Leafhopper Jassids", "Leaf Redenning", "Rust of Cotton", "Wet Weather Blight",
               "White Flies"]

    # TFLITE SETUP
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create a copy of image for saving with results
    output = image.copy()

    # Image Preprocessing
    image = cv2.resize(image, (224, 224))  # Input shape for the model

    image = image.astype("float") / 255.0  # Scale the pixel values from range 0-255 to range 0-1

    input_shape = input_details[0]['shape']

    input_tensor = np.array(np.expand_dims(image, 0), dtype="float32")

    input_index = interpreter.get_input_details()[0]["index"]

    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()

    output_details = interpreter.get_output_details()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    preds = np.squeeze(output_data)

    result = dict()
    thres = 0.75
    for i in range(7):
        if preds[i] > thres:
            result[classes[i]] = round(float(list(preds)[i]*100), 2)

    if bool(result) == False:
        result = {"No":"Prediction"}

    # idx = pred.argsort()[-3:][::-1]  # Get top 3 classes
    #
    # result = dict(zip([classes[i] for i in idx], [round(pred[i] * 100, 2) for i in idx]))

    # Save the image with prediction
    img_text = max(result.items(), key=lambda k: k[1])
    op_folder = "static/outputs"
    name = os.path.split(img_path)[1]
    X = output.shape[1]
    Y = output.shape[0]
    x, y, w, h = 0, 0, int(X * 0.7), int(Y * 0.075)

    cv2.rectangle(output, (x, y), (X, y + h), (0, 0, 0), -1)
    cv2.putText(output, "Classification : {} with confidence : {} %".format((img_text[0]),
                                                                            (img_text[1])),
                (int(X * 0.04), int(Y * 0.04)), cv2.FONT_HERSHEY_SIMPLEX, Y * 0.0007, (255, 255, 255), int(X * 0.002))
    cv2.imwrite(os.path.join(op_folder, name), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

    # Creating url of saved photo with label
    n_url = os.path.join("..\\", op_folder, name)
    n_url = n_url.replace("\\", "/")

    return result, n_url


# Load the model
model_path = "cnnvgg16_model.tflite"  # path of the tflite model


@application.route('/', methods=['GET', 'POST'])
def index():
    form = ImageForm()  # Create an instance of form

    if form.validate_on_submit():  # If form is valid on submitting
        # Grab the image 

        filename = photos.save(form.photo.data)
        file_url = photos.url(filename)
        filepath = photos.path(filename)
        session['ImgUrl'] = file_url
        session['ImgPath'] = filepath

        return redirect(url_for("classify_disease"))

    return render_template('home.html', form=form)


@application.route('/prediction')
def classify_disease():
    # Get the image
    file_url = session["ImgUrl"]
    imgpath = session['ImgPath']
    img = cv2.imread(imgpath)
    # img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results, n_url = dise_class(tflite_path=model_path, image=img, img_path=imgpath, url=file_url)

    return render_template('thankyou.html', results=results, file_url=n_url)


if __name__ == '__main__':
    application.run('127.0.0.1', port=5000, debug=True)
