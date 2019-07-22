#!/usr/bin/env python3
import numpy as np
from glob import glob
import os
import sys

from PIL import Image
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as models

############################ Human Detector ####################################
def face_detector(img_path):
    """
    Returns "True" if face is detected in image stored at img_path

    Args:
        img_path: path to image file
    """
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    # read in image
    img = cv2.imread(img_path)
    # convert to grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect faces
    faces = face_cascade.detectMultiScale(gray)
    # returns "True" if face is detected
    return len(faces) > 0

############################ Process Image ####################################
def process_image(img_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model

    Args:
        img_path: path to an image

    Returns:
        Numpy array of image
    """
    # Open image as PIL image
    im = Image.open(img_path)

    # Get width and height of PIL image
    width, height = im.size
    # Calculate aspect ratio
    aspect_ratio = width/height
    # Make shortest side 256 pixels, keeping aspect ratio
    if width < height:
        im.resize((256,int(256*aspect_ratio**-1)))
    else:
        im.resize((int(256*aspect_ratio),256))

    # Center crop to 224 x 224
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    im = im.crop((left, top, right, bottom))

    # Convert PIL image to numpy array
    np_image = np.array(im)

    # Scale color channels to floats 0–1
    np_image_scaled = np.array(np_image)/255

    # Normalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_norm_image = (np_image_scaled - mean)/std

    # Reorder dimensions so that color channel is first, retain order of other two
    np_final_image = np.transpose(np_norm_image, [2,0,1])

    return np_final_image

############################ Load VGG16 model ##################################
def VGG16_predict(img_path):
    """
    Use pre-trained VGG-16 model to obtain index corresponding to
    predicted ImageNet class for image at specified path

    Args:
        img_path: path to an image

    Returns:
        Index corresponding to VGG-16 model's prediction
    """

    ## Load and pre-process an image from the given img_path
    img = process_image(img_path)

    # Convert numpy array image to tensor
    timg = torch.from_numpy(img).type(torch.FloatTensor)

    # Add batch of size 1 to image
    timg.unsqueeze_(0)

    ## Return the *index* of the predicted class for that image

    # define VGG16 model
    VGG16 = models.vgg16(pretrained=True)

    # set device to cpu
    device = torch.device('cpu')

    # set model to cpu
    VGG16.cpu()

    # Set model to evaluation mode
    VGG16.eval()

    # Calculate the class log probabilities for img
    with torch.no_grad():
        timg = timg.to(device)
        output = VGG16.forward(timg)

    # Calculate class probabilities for img
    ps = torch.exp(output)

    # Determine top probability and predicted index
    top_probs, top_indices = ps.topk(1, dim=1)
    top_index = top_indices.cpu().detach().numpy().tolist()[0][0]

    return top_index # predicted class index

############################## Dog Detector ####################################
def dog_detector(img_path):
    ## TODO: Complete the function.
    """
    Use pre-trained VGG-16 model to obtain index corresponding to
    predicted ImageNet class for image at specified path.

    Args:
        img_path: path to an image

    Returns:
        returns "True" if dog is detected in image stored at img_path
    """
    index = VGG16_predict(img_path)

    return 151 <= index <= 268 # true/false

############################## Load Densenet121 ################################
def load_transfer_model(model_pth_path):
    """
    Loads pretrained densenet121 model, adjusts classifier layer, and adds
    weights from best trained model.

    Args:
        model_pth_path: path to model weights

    Returns:
        model for classifying dog breed
    """

    # Load the pretrained model from pytorch
    model_transfer = models.densenet121(pretrained=True)   

    # Freeze parameters
    for param in model_transfer.parameters():
        param.requires_grad = False

    # Set last layer to number of dog breed classes
    n_inputs = model_transfer.classifier.in_features

    last_layer = nn.Linear(n_inputs, 133)

    model_transfer.classifier = last_layer

    # set model to cpu
    model_transfer.cpu()

    # load the model that got the best validation accuracy
    model_transfer.load_state_dict(torch.load(model_pth_path, map_location='cpu'))

    return model_transfer

############################ Load Class Names #################################
def load_class_names(class_names_path):
   """
   Load numpy file of dog breed class names.

   Args:
      class_names_path: path to numpy file of class names

   Returns:
      list of dog breed class names
   """
   # load numpy file
   class_names = np.load("class_names.npy")

   return class_names

############################## Predict Breed ###################################
def predict_breed_transfer(img_path, model, class_names, k=2):
    """
    Load the image and return the predicted breed.

    Args:
        img_path: path to image
        model
        class_names: list of class names
        k: integer of classes to save

    Returns:
        top_probs: list of probabilities for each class
        top_breeds: list of class names ~ breeds
    """
    # Set device to cpu
    device = torch.device('cpu')

    # Set model to cpu
    model.cpu()

    # Set model to evaluation mode
    model.eval()

    # Preprocess image
    img = process_image(img_path)

    # Convert numpy array image to tensor
    timg = torch.from_numpy(img).type(torch.FloatTensor)

    # Add batch of size 1 to image
    timg.unsqueeze_(0)

    # Calculate the class probabilities for img
    with torch.no_grad():
        timg = timg.to(device)
        output = model.forward(timg)

    # Convert probabilities to proper format
    ps = nn.functional.softmax(output, dim=1)

    # Determine top probabilities and predicted classes
    top_probs, top_indices = ps.topk(k, dim=1)
    top_probs = top_probs.cpu().detach().numpy().tolist()[0]
    top_indices = top_indices.cpu().detach().numpy().tolist()[0]

    # Convert indices to classes
    #idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    #top_classes = [idx_to_class[index] for index in top_indices]
    top_dog_breeds = [class_names[cat] for cat in top_indices]

    return top_probs, top_dog_breeds
