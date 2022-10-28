import os

paths = [".\\video", ".\\1000fps"]
fmts = ["mp4"]
fileTgt = "./files.txt"

for i in range(len(fmts)):
    fmts[i] = fmts[i].lower()


def readDir(path):
    files = os.listdir(path=path)
    filesPath = []
    for file in files:
        file = os.path.join(path, file)
        if not os.path.isdir(file):
            suffix = file.split(".")[-1]
            suffix = suffix.lower()
            print("NOT DIR:", file)
            print(suffix, "in", fmts)
            if suffix in fmts:
                print(suffix, "in", fmts)
                filesPath.append(file)
        else:
            filesPath += readDir(file)
    
    return filesPath


files = []
for p in paths:
    files += readDir(p)

print(files)

with open(fileTgt, "w") as f:
    for i in files:
        f.writelines([i, "\n"])