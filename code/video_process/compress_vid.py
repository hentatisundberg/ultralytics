
# Video compression using os

import cv2
from pathlib import Path

def compress_videos(input, outputdir):

    file = Path(input)
    name = file.name
    output = outputdir+name
    print(output)

    cap = cv2.VideoCapture(input)

    if not cap.isOpened():
        print("Error: Could not open the input video file.")
        exit()

    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Change this to your desired codec
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output, fourcc, frame_rate, frame_size, isColor=True)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            out.write(frame)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


compress_videos("vids/orig/BONDEN322062208-3.avi", "vids/compressed/")