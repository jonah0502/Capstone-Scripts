#!/usr/bin/env python3

import os
import cv2
import numpy as np

input_dir  = input("What is the path to the uncropped photos?: ")
label_dir  = input("What is the path to the uncropped labels?: ")
output_dir  = input("What is the name of the folder you want to export the cropped photos to: ")
isExists = os.path.exists('./' + output_dir)

if not isExists:
    os.mkdir('./' + output_dir)

imageArr = os.listdir(input_dir)
for image in imageArr:
        img = cv2.imread(input_dir + '/' + image, cv2.IMREAD_COLOR)
        txt = open((label_dir + '/' + image).replace(".jpg", ".txt"), "r")
        lines = txt.readlines()
        dimensions = img.shape
        
        converted = []
        label = None
        labelSize = -1
        for line in lines:
            #class, xmid, ymid, width, height
            labelArr = line.split(" ")
            for i in range(len(labelArr)):
                labelArr[i] = float(labelArr[i])
            xMin = labelArr[1] - labelArr[3]/2 
            xMax = labelArr[1] + labelArr[3]/2 
            yMin = labelArr[2] - labelArr[4]/2 
            yMax = labelArr[2] + labelArr[4]/2 
            
            newLabelSize = (xMax - xMin) * (yMax - yMin)

            if(newLabelSize > labelSize): #make labelsiexe
                labelSize = newLabelSize
                label = [xMin, yMin, xMax, yMax]
        
        print(label)
        label[0] = int(label[0] * dimensions[1])
        label[1] = int(label[1] * dimensions[0])
        label[2] = int(label[2] * dimensions[1])
        label[3] = int(label[3] * dimensions[0])
        print(label)
        print(dimensions)

        cropped = img[ label[1]:label[3], label[0]:label[2]]
        # cropped = img[int(label[1] * dimensions[1]):int(label[3] * dimensions[1]), int(label[0] * dimensions[0]):int(label[2] * dimensions[0])]
        #cropped = im[int(label[0] * dimensions[0]):int(label[2] * dimensions[0])], [int(label[1] * dimensions[1]):int(label[3] * dimensions[1])]

        for x in label:
            print(x)
        cv2.imwrite(output_dir + '/' + image, cropped)