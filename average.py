import cv2
import numpy as np
from random import randint
from skimage.measure import compare_ssim as ssim
from skimage import img_as_float
import time



def upgrade_average_frame(video, list_of_index, list_of_frames):
    value = randint(init, min(init + duration, total) - 1)
    dif = []
    for i in list_of_index:
        dif.append(abs(i - value))
        if min(dif) < 5 * fps:
            upgrade_average_frame(video, list_of_index, list_of_frames)
        else:
            video.set(cv2.CAP_PROP_POS_FRAMES, value)
            ret, frame = video.read()
            if frame is None:
                upgrade_average_frame(video, list_of_index, list_of_frames)
            else:
                list_of_index.append(value)
                gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                list_of_frames.append(gframe)
                test_average_frame(video, list_of_index, list_of_frames)


def test_average_frame(video, list_of_index, list_of_frames):
    mean1 = np.average(list_of_frames[::-1], axis=0)
    mean1 = np.array(mean1, dtype="uint8")
    mean2 = np.average(list_of_frames[::-2], axis=0)
    mean2 = np.array(mean2, dtype="uint8")
    mean3 = np.average(list_of_frames[::-3], axis=0)
    mean3 = np.array(mean3, dtype="uint8")
    dif = abs(ssim(img_as_float(mean1), img_as_float(mean2)) - ssim(img_as_float(mean2), img_as_float(mean3)))
    if dif > 0.00035:
        upgrade_average_frame(video, list_of_index, list_of_frames)
    else:
        print len(list_of_frames)
        cv2.imwrite("average_frame.jpg", mean1)
        cv2.imshow("Background Model", mean1)
        print ("--- %s seconds ---" % (time.time() - start_time))
        if cv2.waitKey(0) & 0xFF == 27: #maquina 64 bits, se 32, k == 27
            cv2.destroyAllWindows()


def create_average_frame(video):
    index = []
    frames = []
    while len(index) != 20:
        if len(index) == 0:
            value = randint(init, min(init + duration, total) - 1)
            index.append(value)
        else:
            value = randint(init, min(init + duration, total) - 1)
            dif = []
            for i in index:
                dif.append(abs(i - value))
            if min(dif) < 5 * fps:
                pass
            else:
                video.set(cv2.CAP_PROP_POS_FRAMES, value)
                ret, frame = video.read()
                if frame is None:
                    pass
                else:
                    index.append(value)
                    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(gframe)
    print "created"
    test_average_frame(video, index, frames)

start_time = time.time()
starts = [26, 15, 18, 11, 14, 15, 17, 14]
n = 6
inicio = starts [n-1]
minutes = 5
vd = cv2.VideoCapture(str(n) + ".avi")

if vd.isOpened() == False:
    print "Video Not Opened"
    vd.release
else:
    roi_x = 152
    roi_xw = 489
    roi_y = 90
    roi_yh = 379
    fps =  vd.get(cv2.CAP_PROP_FPS)
    total = vd.get(cv2.CAP_PROP_FRAME_COUNT)
    height = vd.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vd.get(cv2.CAP_PROP_FRAME_WIDTH)
    init = inicio * vd.get(cv2.CAP_PROP_FPS)
    duration = minutes * 60 * vd.get(cv2.CAP_PROP_FPS)
    create_average_frame(vd)
    vd.release()