from itertools import count
import os
from turtle import st
from typing import Counter

import cv2
import random
import numpy as np


fileListPaths = [".\\video.txt", ".\\1000fps.txt"]
fileList = []
tgtPath = "F:\\img"
testProp = 0.25

counter = 0

# print(fileList)


def processVideo(path, targetPath, stepSize=43, keyFrames=[0, 7, 14, 21, 28, 35, 42], resolution=(960, 540)):
    capture = cv2.VideoCapture(path)
    end = False
    global counter
    
    if capture.isOpened():
        print("Processing", path, "...")
        # targetPath = os.path.join(targetPath, path)
        blurryPath = os.path.join(targetPath, "blurry")
        sharpPath = os.path.join(targetPath, "sharp")
        
        if not os.path.exists(tgtPath):
            os.makedirs(tgtPath)
        if not os.path.exists(blurryPath):
            os.makedirs(blurryPath)
        if not os.path.exists(sharpPath):
            os.makedirs(sharpPath)
        
        # counter = 0
        while not end:
            print("Counter: {0}".format(counter), end="\r")
            imgGroup = []
            for i in range(stepSize):
                ret, img = capture.read()
                if not ret:
                    end = True
                    break
                imgGroup.append(img)

            if end:
                print("Counter: {0}".format(counter))
                break

            # sharpGroupPath = os.path.join(sharpPath, "%d"%counter)
            # if not os.path.exists(sharpGroupPath):
            #     os.makedirs(sharpGroupPath)

            blurImg = np.zeros(imgGroup[0].shape)
            blurImg = cv2.resize(blurImg, resolution, interpolation=cv2.INTER_CUBIC)

            sharpCounter = 1
            for j in range(len(imgGroup)):
                img = imgGroup[j]
                img = cv2.resize(img, resolution, interpolation=cv2.INTER_CUBIC)
                blurImg += img

                if j in keyFrames:
                    cv2.imwrite(os.path.join(sharpPath, "sharp_{:0>6d}_{}.png".format(counter+1, sharpCounter)), img)
                    sharpCounter += 1
                # print("img00:", img[0][0], ", blurry00:", blurImg[0][0])
            blurImg = blurImg / stepSize
            blurImg = cv2.resize(blurImg, resolution, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(blurryPath, "blurry_{:0>6d}.png".format(counter+1)), blurImg)

            counter += 1
        
        print(path, "DONE.\n")

for fileListPath in fileListPaths:
    with open(fileListPath, "r") as f:
        fileList = f.readlines()

    for i in range(len(fileList)):
        fileList[i] = fileList[i].strip()
    
    random.shuffle(fileList)

    testSize = round(len(fileList) * testProp)
    testSet = fileList[:testSize]
    trainSet = fileList[testSize:]

    suffix = fileListPath.split(".")[-1]
    testPath = os.path.join(tgtPath, fileListPath.rstrip(suffix), "test")
    trainPath = os.path.join(tgtPath, fileListPath.rstrip(suffix), "train")

    # print(len(testSet), "\n", testSet, "\n=======================\n", len(trainSet), "\n", trainSet)


    if not os.path.exists(testPath):
        os.makedirs(testPath)

    if not os.path.exists(trainPath):
        os.makedirs(trainPath)
 
    c = 1
    print("Test set:", fileListPath)
    for file in testSet:
        print(c, "of", len(testSet))
        processVideo(file, testPath)
        c += 1

    c = 1
    print("\nTrain set:", fileListPath)
    
    counter = 0
    for file in trainSet:
        print(c, "of", len(trainSet))
        processVideo(file, trainPath)
        c += 1

    print("\n=============================================\n")