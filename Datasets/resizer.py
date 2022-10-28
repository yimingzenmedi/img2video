import os
import cv2

dirs = ['1000fps', 'video', 'video_min', 'video_reduced']
# dirs = ["video_min"]
fmt = "png"
resolution = (960, 540)

for dir in dirs:
    print("\ndir:", dir)
    g = os.walk(dir)

    for path, dir_list, file_list in g:
        logs = ""
        counter = 0
        for file_name in file_list:
            file_path = os.path.join(path, file_name)

            file_fmt = file_path[-3:len(file_path)]
            if file_fmt == fmt:
                counter += 1
                img = cv2.imread(file_path)
                if img.shape[1] != resolution[0] or img.shape[0] != resolution[1]:
                    logs += "\nresize: " + file_path
                    img = cv2.resize(img, resolution, interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(file_path, img)
                else:
                    logs += "\nskip: " + file_path
            if counter % 200 == 0:
                print(logs)
                print("counter:", counter)
        print(logs)
        print("total counter:", counter)
