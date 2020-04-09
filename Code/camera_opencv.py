import os
import cv2
from base_camera import BaseCamera
import CNN
from CNN import model


class Camera(BaseCamera):
    video_source = 0
    letter_pred = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
        8: "I",
        9: "K",
        10: "L",
        11: "M",
        12: "N",
        13: "O",
        14: "P",
        15: "Q",
        16: "R",
        17: "S",
        18: "T",
        19: "U",
        20: "V",
        21: "W",
        22: "X",
        23: "Y",
    }

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        width = camera.get(3) #get width and height of img
        height = camera.get(4)
        
        crop_image_width = 224 #input size for pytorch pre-trained models
        crop_image_height = 224
        
        startW = int ((width-crop_image_width)/2)
        startH = int ((height - crop_image_height)/2)
        endH = startH + crop_image_height
        endW = startW + crop_image_width

        start_pt = (int(startW),int(startH))
        end_pt = (int(endW), int(endH) )
        color = (255,0,0)
        thickness = 1

        draw_H = startH - 20
        draw_W = int ((startW +endW)/2 )
        draw_pt = (int (draw_W), int(draw_H))
        font = cv2.FONT_HERSHEY_SIMPLEX


        while True:
            # read current frame
            _, img = camera.read()
            #DO ERROR CHECK IF WEBCAM CAMERA NOT DETECTED

            img = cv2.flip (img, 1)

            img = cv2.rectangle(img, start_pt, end_pt, color, thickness=thickness) #draw input rectangle

            cropped_img = img[startH:endH,startW:endW] #crop img to rectangle
            
            index = CNN.predict(model,cropped_img)

            prediction = Camera.letter_pred[index]

            img  = cv2.putText(img, prediction, draw_pt, font, 2, (0,0,255),3)

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()



