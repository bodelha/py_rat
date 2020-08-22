import cv2
import numpy as np
from random import randint
from skimage.measure import compare_ssim as ssim
from skimage import img_as_float
from scipy import ndimage
from matplotlib import pyplot as plt

def analysis (video):
    a_f = cv2.imread("C:\\Users\\bodel\\Desktop\\output4\\average_frame.jpg",0)
    video.set(cv2.CAP_PROP_POS_FRAMES, init)
    cx = []
    cy = []
    while True:
        ret, frame = vd.read()
        if frame is None:
            pass
        else:
            rframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pos = video.get(cv2.CAP_PROP_POS_MSEC)
            cv2.imwrite("C:\\Users\\bodel\\Desktop\\output4\\frames\\" + str(pos) + ".jpg", frame)
            dif = cv2.absdiff(rframe, a_f)
            cv2.imwrite("C:\\Users\\bodel\\Desktop\\output4\\dif\\"+str(pos)+".jpg", dif)
            m, n = dif.shape
            bdif = cv2.blur(dif, (int(n/50), int(m/50)))
            cv2.imwrite("C:\\Users\\bodel\\Desktop\\output4\\bdif\\" + str(pos) + ".jpg", bdif)
            lim = np.amax(bdif)
            if lim < 5:
                lim = 5
            hist, bins = np.histogram(bdif.ravel(), lim+1, [0, lim+1])
            hist_i = hist [::-1]
            bins = list(reversed(bins))
            sum = 0
            value = 0
            for i in xrange(len(hist_i) - 1):
                if sum < int(m*n/30):
                    sum +=  hist_i[i]
                    value = bins[i+1]
                else:
                    pass
            ret, thresh = cv2.threshold(bdif,value, 255, cv2.THRESH_BINARY)
            cv2.imwrite("C:\\Users\\bodel\\Desktop\\output4\\thresh\\" + str(pos)+str(value) + ".jpg", thresh)
            eroded = cv2.morphologyEx(thresh, cv2.MORPH_ERODE,(int(n/50), int(m/50)))
            closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, (int(n/50), int(m/50)), iterations= 1)
            cv2.imwrite("C:\\Users\\bodel\\Desktop\\output4\\thresh2\\" + str(pos) + ".jpg", closed)
            img, cnt, hier = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            mask = np.zeros(closed.shape, np.uint8)
            cv2.drawContours(mask, max(cnt, key=cv2.contourArea), -1, 255, 2)
            cv2.imwrite("C:\\Users\\bodel\\Desktop\\output4\\cont\\" + str(pos) + ".jpg", mask)
            img, cnt1, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.fillPoly(mask, cnt1, 255)
            cv2.imwrite("C:\\Users\\bodel\\Desktop\\output4\\cont\\" + str(pos) + "2.jpg", mask)
            nframe = frame.copy ()
            cv2.drawContours(nframe, cnt1, -1, (255, 0, 0), 3)
            cv2.imwrite("C:\\Users\\bodel\\Desktop\\output4\\fc\\" + str(pos) + ".jpg", nframe)
            cg = ndimage.measurements.center_of_mass(mask)
            cgx = int(cg[1])
            cgy = int(cg[0])
            cx.append(cgx)
            cy.append(cgy)
            if video.get (cv2.CAP_PROP_FPS)% 5 == 0:
                if len(cx) == video.get (cv2.CAP_PROP_FPS)/5:
                    print cx,cy
                    cx = []
                    cy = []
            elif video.get (cv2.CAP_PROP_FPS)% 4 == 0:
                if len(cx) == video.get (cv2.CAP_PROP_FPS)/4:
                    print cx,cy
                    cx = []
                    cy = []
            elif video.get (cv2.CAP_PROP_FPS)% 2 == 0:
                if len(cx) == video.get (cv2.CAP_PROP_FPS)/2:
                    print cx,cy
                    cx = []
                    cy = []
            else:
                if len(cx) == video.get (cv2.CAP_PROP_FPS):
                    print cx,cy
                    cx = []
                    cy = []
            cv2.circle (frame, (cgx, cgy), int(max(frame.shape)/100), (55, 125, 3), -1 )
            cv2.imwrite("C:\\Users\\bodel\\Desktop\\output4\\end\\" + str(pos) + ".jpg", frame)
            #cv2.imshow("dfv", frame)
            #cv2.waitKey(2)
    #cv2.destroyAllWindows()


def upgrade_average_frame (video, list_of_index, list_of_frames):
    value = randint (init, total-1)
    dif = []
    for i in list_of_index:
        dif.append(abs(i-value))
        if min(dif) < 5*video.get(cv2.CAP_PROP_FPS):
            upgrade_average_frame(video, list_of_index, list_of_frames)
        else:
            vd.set(cv2.CAP_PROP_POS_FRAMES, value)
            ret, frame = vd.read()
            if frame is None:
                upgrade_average_frame(video, list_of_index, list_of_frames)
            else:
                list_of_index.append(value)
                gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                list_of_frames.append(gframe)
                test_average_frame(video, list_of_index, list_of_frames)

def test_average_frame (video, list_of_index, list_of_frames):
    mean1 = np.average(list_of_frames[::-1], axis=0)
    mean1 = np.array(mean1, dtype="uint8")
    global n
    n = n + 1
    cv2.imwrite("C:\\Users\\bodel\\Desktop\\output4\\1\\average_frame"+str(n)+".jpg", mean1)
    mean2 = np.average(list_of_frames[::-2], axis=0)
    mean2 = np.array(mean2, dtype="uint8")
    mean3 = np.average(list_of_frames[::-3], axis=0)
    mean3 = np.array(mean3, dtype="uint8")
    dif =  abs(ssim(img_as_float(mean1), img_as_float(mean2)) - ssim(img_as_float(mean2), img_as_float(mean3)))
    if dif > 0.00035:
        print "upgrading", len(list_of_frames)
        upgrade_average_frame (video, list_of_index, list_of_frames)
    else:   
        print len(list_of_frames)
        cv2.imwrite("C:\\Users\\bodel\\Desktop\\output4\\average_frame.jpg", mean1)
        analysis(video)

def create_average_frame(video):
    index = []
    frames = []
    while len(index) != 3:
        if len (index) == 0:
            value = randint(init, total - 1)
            index.append(value)
        else:
            value = randint(1, total - 1)
            dif = []
            for i in index:
                dif.append(abs (i - value))
            if min(dif) <  5*video.get(cv2.CAP_PROP_FPS):
                pass
            else:
                vd.set(cv2.CAP_PROP_POS_FRAMES, value)
                ret, frame = vd.read()
                if frame is None:
                    pass
                else:
                    index.append(value)
                    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(gframe)
    print "created"
    test_average_frame (video, index, frames)


vd = cv2.VideoCapture("C:\\Users\\bodel\\Desktop\\OF.wmv")

if vd.isOpened() == False:
    print "Video Not Opened"
else:
    total = vd.get(cv2.CAP_PROP_FRAME_COUNT)
    height =  vd.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vd.get(cv2.CAP_PROP_FRAME_WIDTH)
    n = 3
    #init = 0
    init = int(190/vd.get(cv2.CAP_PROP_FPS))
    create_average_frame(vd)
