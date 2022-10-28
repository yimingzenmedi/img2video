import os

dirPath = ".\\GoPro\\test\\input"
dirPathNew = ".\\GoPro\\test\\input"
dir_ = os.walk(dirPath)

for path, dirList, fileList in dir_:
    for fileName in fileList:
        preName, suffix = fileName.split(".")
        print(preName.split("-"))
        prefix, counter = preName.split("-")
        counter = int(counter)
        print(fileName, preName, prefix, counter, suffix)
        newCounter = counter + 4
        newName = "{prefix}-{newCounter}.{suffix}".format(prefix=prefix, newCounter="%06d" % newCounter, suffix=suffix)
        print(newName)
        filePath = os.path.join(dirPath, fileName)
        newFilePath = os.path.join(dirPathNew, newName)
        os.rename(filePath, newFilePath)
