import os
import urllib.request
from app import app
from flask import Flask, flash, request, redirect, render_template, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from classifier import *
import cv2
import numpy as np

# opencv allowed file extensions
ALLOWED_EXTENSIONS = set(['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpe', 'jp2', 'tiff','tif', 'png', 'jpg', 'jpeg'])

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES = os.path.join(APP_ROOT, 'class_names.npy')
MODEL_TRANSFER = os.path.join(APP_ROOT, 'model_transfer.pt')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['GET','POST'])
def upload_file():
    target = os.path.join(APP_ROOT, app.config['UPLOAD_FOLDER'])
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # check if file is selected
        if file.filename == '':
            return redirect(request.url)
        # save and display uploaded image
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            destination = '/'.join([target, filename])
            print(destination)
            file.save(destination)
            # load class names
            class_names = load_class_names(CLASS_NAMES)
            # load model
            model = load_transfer_model(MODEL_TRANSFER)
            # run pipeline
            run_app(destination, class_names, model)
            return render_template('display.html',image_name=filename)
        # check if file has right extension
        # note that some are not supported in OpenCV
        else:
            flash('* Check that you have the right file extension. *')
            return redirect(request.url)

def run_app(img_path, class_names, model):
    """
    Accepts a file path to image, determines whether image contains a human, dog, or neither, and returns dog breed.

    Args:
        img_path: path to image
        class_names: list of class names
        model
    """
    # Check if image contains a human
    is_human = face_detector(img_path)
    # Check if image contains a dog
    is_dog = dog_detector(img_path)

    # Exit program if both human and dog are detected or neither is detected
    if (is_human and is_dog) or (not is_human and not is_dog):
        flash("Error processing image. Please ensure your image contains either a human or a dog.")
        return

    # Predict dog breed
    probs, breeds = predict_breed_transfer(img_path, model, class_names, k=2)

    # Print out top breed for human image
    if is_human:
        flash("Hello Human!")
        flash("You look like a ...")
        flash(breeds[0].replace("_", " "))

    # Print out top breed for dog image
    if is_dog:
        flash("Hello Dog!")
        flash("Your predicted breed is ...")
        # Check if top second breed's probability is within 10% of top breed's probability
        if probs[1] >= (probs[0]-probs[0]*0.1):
            # print out both breeds
            flash("Mixed breed: {} and {}".format(breeds[0].replace("_", " "), breeds[1].replace("_", " ")))
        else:
            # print out the top breed
            flash(breeds[0].replace("_", " "))

@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory('uploads',filename)

@app.errorhandler(413)
def request_entity_too_large(error):
    return 'File too large. Must be less than 16 MB.', 413

if __name__ == "__main__":
    app.run()
