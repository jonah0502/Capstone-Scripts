import os
import sys
import random
import shutil

#   UTILIZATION
#   Please organize your files as
#   *parent_dir/
#       images/
#           All of your images (jpg format)
#       labels/
#           All of your labels (Yolo txt format)

#   This program will randomly assign images to your training and validation set in your current directory

#   Run this program with
#   python splitDataScript.py path/to/parent_dir --test-true --percentage percent<-(range 0-1)

#   Check if the input directory exists
if(not os.path.exists(sys.argv[1])):
    print("That directory does not exist")
    exit()

trainPath = 'train' + sys.argv[1]
valPath = 'val' + sys.argv[1]
testPath = 'test' + sys.argv[1]
isTest = False
isPercent = False
#   Check if train/val directories already exist
if(os.path.exists(trainPath) or os.path.exists(valPath)):
    print("Test or validation directory already exists")
    exit()

#   Make the train and test directories
os.mkdir(trainPath)
os.mkdir(valPath)
percentage = 1
if (sys.argv[2] == '--test-true'):
    isTest = True
    os.mkdir(testPath)
    os.mkdir(testPath + '/images/')
    os.mkdir(testPath + '/labels/')
    if len(sys.argv) >= 4:
        print(len(sys.argv))
        if sys.argv[3] == '--percentage':
            percentage = sys.argv[4]
            isPercent = True
elif (sys.argv[2] == '--percentage'):
    percentage = sys.argv[3]
    isPercent = True
#   Make the subdirectories of each
os.mkdir(trainPath + '/images/')
os.mkdir(trainPath + '/labels/')
os.mkdir(valPath + '/images/')
os.mkdir(valPath + '/labels/')

#   Get the file names of the original directory
fNames = os.listdir(sys.argv[1] + '/images/')

newLen = round(len(fNames) * float(percentage))
#   Remove .jpg from each file name
for i in range(newLen):
    fNames[i] = fNames[i][0:(len(fNames[i]) - 4)]

#   shuffle order and segment
random.shuffle(fNames)
#   70% goes into training
trainFNames = fNames[:((7 * newLen) // 10)]
valFNames = fNames[((7 * newLen) // 10):]

trainData = 0
valData = 0

#   Copy the images to train
for name in trainFNames:
    srcImageFName = './' + sys.argv[1] + '/images/' + name
    srcLabelFName = './' + sys.argv[1] + '/labels/' + name
    if(os.path.exists(srcImageFName + '.jpg') and os.path.exists(srcLabelFName + '.txt')):
        shutil.copyfile(srcImageFName + '.jpg', trainPath + '/images/' + name + '.jpg')
        shutil.copyfile(srcLabelFName + '.txt', trainPath + '/labels/' + name + '.txt')
        if (isTest):
            if(not isPercent and trainData < (len(fNames) * 0.1)):
                shutil.copyfile(srcImageFName + '.jpg', testPath + '/images/' + name + '.jpg')
                shutil.copyfile(srcLabelFName + '.txt', testPath + '/labels/' + name + '.txt')
        trainData += 1
    else:
        print('skipped one file not found in original dataset')

#   Copy the images to validation
for name in valFNames:
    srcImageFName = './' + sys.argv[1] + '/images/' + name
    srcLabelFName = './' + sys.argv[1] + '/labels/' + name
    if(os.path.exists(srcImageFName + '.jpg') and os.path.exists(srcLabelFName + '.txt')):
        shutil.copyfile(srcImageFName + '.jpg', valPath + '/images/' + name + '.jpg')
        shutil.copyfile(srcLabelFName + '.txt', valPath + '/labels/' + name + '.txt')
        valData += 1
    else:
        print('skipped one file not found in original dataset')

print(str(trainData) + ' training data and ' + str(valData) + ' validation data')