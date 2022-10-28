"""
This file refers to https://github.com/nmhkahn/CARN-pytorch and made changes.
"""

import os
import numpy as np
import imageio
import glob
import h5py
import datetime


dataset_type = "test"
size_limit = 0

dirs = ["video", "1000fps"]

totalstarttime = datetime.datetime.now()
for dir in dirs:
    starttime = datetime.datetime.now()
    dataset_dir = dir+"\\{}\\".format(dataset_type)
    h5 = h5py.File("D:\\Study\\ANU-study-stuff\\COMP8755 Personal Research Project\code\\base_method\\datasets\\{}_{}.h5".format(dir, dataset_type), "w")
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))

    print(starttime, "Starting...")

    # sharp_paths = glob.glob(os.path.join(dataset_dir, "sharp", "*.png"))
    blurry_paths = glob.glob(os.path.join(dataset_dir, "blurry", "*.png"))
    blurry_paths.sort()

    sharp_grp = h5.create_group("sharp")
    blurry_grp = h5.create_group("blurry")
    total = len(blurry_paths)

    for i, blurry_path in enumerate(blurry_paths):
        file_index = blurry_path.split("\\")[-1].split(".")[0].split("_")[1]
        filename = "sharp_" + file_index
        sharp_paths = [os.path.join(dataset_dir, "sharp", "{filename}_{idx:d}.png".format(filename=filename, idx=idx)) for idx in range(1, 8)]
        # print(file_index, filename,sharp_paths)
        if size_limit <= 0 or i < size_limit:
            log = "{} of {} \n\tblurry: {}".format(i, total, blurry_path)
            blurry_grp.create_dataset(file_index, data=imageio.imread(blurry_path))
            file_grp = sharp_grp.create_group(file_index)
            counter = 1
            for sharp_path in sharp_paths:
                log += "\n\tsharps: {}".format(sharp_path)
                file_grp.create_dataset(str(counter), data=imageio.imread(sharp_path))
                counter += 1
            log += "\n"
            print(log)

    endtime = datetime.datetime.now()
    print("Time for {}:".format(dir), (endtime - starttime).seconds)

print("Total time:", (endtime - totalstarttime).seconds)

    # for subdir in ["sharp", "blurry"]:

    #     # load images:
    #     if subdir == "sharp":
    #         im_paths = glob.glob(os.path.join(dataset_dir, subdir, "*.png"))
    #         im_paths.sort()
    #         grp = h5.create_group(subdir)
    #         total = len(im_paths)
    #         # build *.h5
    #         for i, path in enumerate(im_paths):
    #             if size_limit <= 0 or i < size_limit:
                    
    #                 im = imageio.imread(path)
    #                 print("%6d"%(i+1), "of", total, "|", path)
    #                 grp.create_dataset(str(i), data=im)

    #     else:
    #         im_paths = glob.glob(os.path.join(dataset_dir, subdir, "*.png"))
    #         im_paths.sort()
    #         grp = h5.create_group(subdir)
    #         total = len(im_paths)
    #         # build *.h5
    #         for i, path in enumerate(im_paths):
    #             if size_limit <= 0 or i < size_limit:
    #                 im = imageio.imread(path)
    #                 print("%6d"%(i+1), "of", total, "|", path)
    #                 grp.create_dataset(str(i), data=im)

        # print(im_paths)
