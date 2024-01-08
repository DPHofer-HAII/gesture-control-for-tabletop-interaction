import time
import os
import cv2
import numpy as np
from mss import mss


def record(name):
    name2 = "Final_Recording/desktop_video"
    with mss() as sct:
        # mon = {'top': 160, 'left': 160, 'width': 200, 'height': 200}
        mon = sct.monitors[2]

        name = name + '.avi'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        desired_fps = 30.0
        recorded = False

        while True:
            img = sct.grab(mon)
            destRGB = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
            # destRGB[200:900, 50:145] = (0, 0, 0)
            #out.write(destRGB)

            # # Process Key (ESC: end) #################################################
            # key = cv2.waitKey(10)
            # if key == 27:  # ESC
            #     break


            if (os.path.exists("C:\\Users\\MMT\\Desktop\\StartScreenRecording.txt") and not recorded):
                out = cv2.VideoWriter(name, fourcc, desired_fps, frameSize=(1920, 1080))
                out2 = cv2.VideoWriter(name2, fourcc, desired_fps, frameSize=(1920, 1080))
                # (mon['width'], mon['height']))
                recorded = True

            if (recorded):
                out.write(destRGB)
                out2.write(destRGB)

            if (recorded and (not os.path.exists("C:\\Users\\MMT\\Desktop\\StartScreenRecording.txt"))):
                out.release()
                out2.release()
                recorded = False

        cv2.destroyAllWindows()
        #out.release()


record("Test_Recordings/desktop_video")