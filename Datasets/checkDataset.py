import os

inpDirPath = ".\\GoPro\\test\\input"
tarDirPath = ".\\GoPro\\test\\target"

inpDir = os.walk(inpDirPath)
tarDir = os.walk(tarDirPath)

counter1 = 0
for path, dirList, fileList in inpDir:
    for fileName in fileList:
        tarPath = os.path.join(tarDirPath, fileName)
        if not os.path.exists(tarPath):
            inpPath = os.path.join(inpDirPath, fileName)
            print("remove file:", inpPath)
            os.remove(inpPath)
            counter1 += 1
print("Counter:", counter1)

counter2 = 0
for path, dirList, fileList in tarDir:
    for fileName in fileList:
        inpPath = os.path.join(inpDirPath, fileName)
        if not os.path.exists(inpPath):
            tarPath = os.path.join(tarDirPath, fileName)
            print("remove file:", tarPath)
            os.remove(tarPath)
            counter2 += 1
print("Counter: ", counter2)

print("TOTAL:", counter2 + counter1)
