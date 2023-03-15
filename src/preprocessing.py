import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

import glob

import imutils

#dataset
#https://www.kaggle.com/code/ruslankl/brain-tumor-detection-v1-0-cnn-vgg-16

# 1. LOAD IMAGES
positive_path = 'raw_data/yes/*.jpg'
negative_path = 'raw_data/no/*.jpg'

positive_imgs = []
negative_imgs = []

# loading 
def load_images():
    positive_imgs = []
    negative_imgs = []
    
    for img in glob.glob(positive_path):
        n = cv2.imread(img)
        positive_imgs.append(n)

    #loading negative examples
    for img in glob.glob(negative_path):
        n = cv2.imread(img)
        negative_imgs.append(n)
        
    return positive_imgs, negative_imgs

positive_imgs, negative_imgs = load_images()

print('# of positive images: ', len(positive_imgs))
print('# of negative images: ', len(negative_imgs))

# DISPLAY IMAGES
def display_examples(negative_imgs, positive_imgs, rows, samples):
    
    #display negative examples
    fig, ax = plt.subplots(rows,samples,figsize=(10,5))
    fig.suptitle('Brain Scans - Tumor: NO', size=16)

    for i in range(rows):
        for j in range(samples):
            rand = np.random.randint(90)
            ax[i,j].imshow(negative_imgs[rand],interpolation='nearest',cmap='gray')
            ax[i,j].axis('off')
    fig.show()
    fig.waitforbuttonpress()
    
    # display positive examples
    fig1, ax1 = plt.subplots(3,10,figsize=(10,5))
    fig1.suptitle('Brain Scans - Tumor: YES', size=16)

    for i in range(rows):
        for j in range(samples):
            rand = np.random.randint(90)
            ax1[i,j].imshow(positive_imgs[rand],interpolation='nearest',cmap='gray')
            ax1[i,j].axis('off')
    fig1.show()
    fig1.waitforbuttonpress()
    
#display_examples(negative_imgs, positive_imgs, rows = 3, samples = 10)  

# 3. PREPROCESS IMAGES (CROP, RESIZE AND AUGMENTATION)

# display step by step cropping strategy
def crop_example(positive_imgs):
    
    img1 = positive_imgs[9]
    
    image = img1.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
    
    img2 = image.copy()
    
    # determine the most extreme points along the contour
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # draw the outline of the object, then draw each of the
    # extreme points, where the left-most is red, right-most
    # is green, top-most is blue, and bottom-most is teal
    cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
    cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
    cv2.circle(image, extRight, 8, (0, 255, 0), -1)
    cv2.circle(image, extTop, 8, (255, 0, 0), -1)
    cv2.circle(image, extBot, 8, (255, 255, 0), -1)
    
    img3 = image.copy()
    
    img4 = img1.copy()
    # Crop
    ADD_PIXELS = 0
    img4 = img4[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()

    #Show steps
    plt.figure(figsize=(15,6))
    plt.subplot(141)
    plt.imshow(img1)
    plt.xticks([])
    plt.yticks([])
    plt.title('Step 1. Get the original image')
    plt.subplot(142)
    plt.imshow(img2)
    plt.xticks([])
    plt.yticks([])
    plt.title('Step 2. Find the biggest contour')
    plt.subplot(143)
    plt.imshow(img3)
    plt.xticks([])
    plt.yticks([])
    plt.title('Step 3. Find the extreme points')
    plt.subplot(144)
    plt.imshow(img4)
    plt.xticks([])
    plt.yticks([])
    plt.title('Step 4. Crop the image')
    plt.show()
    
#crop_example(positive_imgs)

#https://pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/

# Crop all images
def crop_images(positive_imgs, negative_imgs):
    
    pos_images = []
    neg_images = []
    
    imgs = [positive_imgs,negative_imgs]
    
    for i in range(0,2):
        for j in range(len(imgs[i])):
            
            image = (imgs[i][j])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            # threshold the image, then perform a series of erosions +
            # dilations to remove any small regions of noise
            thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # find contours in thresholded image, then grab the largest one
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            # cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
            
            # determine the most extreme points along the contour
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])
            
            # Crop
            ADD_PIXELS = 0
            new_img = image[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()

            if i == 1:
                pos_images.append(new_img)
            else:
                neg_images.append(new_img)
                
    return pos_images, neg_images

positive_imgs,negative_imgs = crop_images(positive_imgs, negative_imgs)

#display_examples(negative_imgs, positive_imgs,rows=3,samples=10)  

def resize_examples(positive_imgs, negative_imgs,img_size) :
    pos_images = []
    neg_images = []

    
    for image in positive_imgs:
        img = cv2.resize(image,
            dsize= img_size,
            interpolation= cv2.INTER_CUBIC)
        pos_images.append(img)
    
    for image in negative_imgs:
        img = cv2.resize(image,
            dsize= img_size,
            interpolation= cv2.INTER_CUBIC)
        neg_images.append(img)
        
    return pos_images, neg_images

positive_imgs, negative_imgs = resize_examples(positive_imgs, negative_imgs, (224,224))

display_examples(negative_imgs, positive_imgs,rows=3,samples=10)  

        
            

