from glob import glob
import re

import cv2

instances_names = [
    "0_test",
    # "p31",
    # "inithx.i.1",
    # "miles250",
    # "p29",
    # "GEOM20a",
    # "inithx.i.2",
    # "GEOM30",
    # "GEOM20",
    # "miles500",
    # "inithx.i.3",
    # "zeroin.i.1",
    # "GEOM40a",
    # "zeroin.i.3",
    # "zeroin.i.2",
    # "GEOM40",
    # "GEOM50",
    # "DSJR500.1",
    # "GEOM80",
    # "mulsol.i.5",
    # "GEOM50a",
    # "GEOM20b",
    # "GEOM60",
    # "GEOM30a",
    # "p32",
    # "GEOM40b",
    # "miles1000",
    # "miles1500",
    # "GEOM60a",
    # "GEOM100",
    # "GEOM70a",
    # "GEOM50b",
    # "GEOM90b",
    # "le450_25b",
    # "le450_25a",
    # "GEOM70b",
    # "GEOM110",
    # "GEOM90",
    # "GEOM60b",
    # "GEOM110b",
    # "GEOM120",
    # "R50_1g",
    # "R50_1gb",
    # "p28",
    # "GEOM30b",
    # "GEOM80b",
    # "GEOM100b",
    # "GEOM70",
    # "GEOM80a",
    # "p26",
    # "wap02a",
    # "wap06a",
    # "wap05a",
    # "wap01a",
    # "p24",
    # "GEOM120b",
    # "p25",
    # "p36",
    # "le450_15a",
    # "wap04a",
    # "wap07a",
    # "wap08a",
    # "GEOM90a",
    # "le450_15b",
    # "p33",
    # "p35",
    # "wap03a",
    # "p21",
    # "p40",
    # "p38",
    # "le450_25c",
]

for instance in instances_names:
    print(instance)
    video_name = f"vid_reduction/{instance}.avi"

    images = glob(f"img_reduction/{instance}_*.png")

    images.sort(key=lambda f: int(re.sub("\D", "", f)))
    # print(images)
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    # print(height, width, layers)

    video = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*"DIVX"), 2, (width, height)
    )

    for image in images[0::]:
        video.write(cv2.imread(image))

    video.write(cv2.imread(images[0]))
    video.write(cv2.imread(images[0]))
    video.write(cv2.imread(images[-1]))
    video.write(cv2.imread(images[-1]))
    video.write(cv2.imread(images[0]))
    video.write(cv2.imread(images[0]))
    video.write(cv2.imread(images[-1]))
    video.write(cv2.imread(images[-1]))

    cv2.destroyAllWindows()
    video.release()
