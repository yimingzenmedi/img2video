import os
import re

dir = "./video_reduced"

test_dir = os.path.join(dir, "test")
train_dir = os.path.join(dir, "train")

test_sharp_dir = os.path.join(test_dir, "sharp")
test_blurry_dir = os.path.join(test_dir, "blurry")
train_sharp_dir = os.path.join(train_dir, "sharp")
train_blurry_dir = os.path.join(train_dir, "blurry")

sum_test_blurry = 1254
sum_train_blurry = 3474

test_sharp_files = []
test_blurry_files = []
train_sharp_files = []
train_blurry_files = []

test_sharp_all = os.listdir(test_sharp_dir)
test_blurry_all = os.listdir(test_blurry_dir)
train_sharp_all = os.listdir(train_sharp_dir)
train_blurry_all = os.listdir(train_blurry_dir)

for i in range(sum_test_blurry):
    blurry_name = 'blurry_%06d.png' % (i+1)
    blurry_path = os.path.join(test_blurry_dir, blurry_name)
    print(f"\r{blurry_path} , exists: {os.path.exists(blurry_path)}", end="")
    if blurry_name in test_blurry_all:
        test_blurry_all.remove(blurry_name)
    if not os.path.exists(blurry_path):
        test_blurry_files.append(blurry_path)
    for j in range(7):
        sharp_name = 'sharp_%06d_%d.png' % ((i+1), (j+1))
        sharp_path = os.path.join(test_sharp_dir, sharp_name)
        print(f"\r{sharp_path}, exists: {os.path.exists(sharp_path)}", end="")
        if sharp_name in test_sharp_all:
            test_sharp_all.remove(sharp_name)
        if not os.path.exists(sharp_path):
            test_sharp_files.append(sharp_path)
print("\ntest done.")

for i in range(sum_train_blurry):
    blurry_name = 'blurry_%06d.png' % (i+1)
    blurry_path = os.path.join(train_blurry_dir, blurry_name)
    print(f"\r{blurry_path} , exists: {os.path.exists(blurry_path)}", end="")
    if blurry_name in train_blurry_all:
        train_blurry_all.remove(blurry_name)
    if not os.path.exists(blurry_path):
        train_blurry_files.append(blurry_path)
    for j in range(7):
        sharp_name = 'sharp_%06d_%d.png' % ((i+1), (j+1))
        sharp_path = os.path.join(train_sharp_dir, sharp_name)
        print(f"\r{sharp_path}, exists: {os.path.exists(sharp_path)}", end="")
        if sharp_name in train_sharp_all:
            train_sharp_all.remove(sharp_name)
        if not os.path.exists(sharp_path):
            train_sharp_files.append(sharp_path)

print("\ntrain done")

print("\n==================================\nNOT EXISTS:\n")
print("> test_sharp_files\n", test_sharp_files)
print("> test_blurry_files\n", test_blurry_files)
print("> train_sharp_files\n", train_sharp_files)
print("> train_blurry_files\n", train_blurry_files)


print("\n=================================\nINVALID:\n")
print("> test sharp:\n", test_sharp_all)
print("> test blurry:\n", test_blurry_all)
print("> train sharp:\n", train_sharp_all)
print("> train blurry:\n", train_blurry_all)


if len(test_sharp_all) + len(test_blurry_all) + len(train_sharp_all) + len(train_blurry_all) > 0:
    cmd = input("\nDelete invalid files? (y/n)\n")
else:
    cmd = "n"

def removeFile(path):
    try:
        os.remove(path)
        print("Removed:", path)
    except:
        print("Remove failed:", path)

if cmd == "y":
    if len(test_sharp_all) > 0:
        for file in test_sharp_all:
            path = os.path.join(test_sharp_dir, file)
            removeFile(path)

    if len(test_blurry_all) > 0:
        for file in test_blurry_all:
            path = os.path.join(test_blurry_dir, file)
            removeFile(path)

    if len(train_sharp_all) > 0:
        for file in train_sharp_all:
            path = os.path.join(train_sharp_dir, file)
            removeFile(path)

    if len(train_blurry_all) > 0:
        for file in train_blurry_all:
            path = os.path.join(train_blurry_dir, file)
            removeFile(path)