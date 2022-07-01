import glob
import time
import cv2

from unet import CustomUnet

unet = CustomUnet()
# unet.train()

font = cv2.FONT_HERSHEY_SIMPLEX
org = (20, 20)
fontScale = 0.75
color = (0, 0, 255)
thickness = 2

# cap = cv2.VideoCapture('video.mp4')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if ret:
        start = time.time()
        res = unet.predict(img)
        txt = f'Time: {(time.time() - start) * 1000:.2f} ms'
        result_img = cv2.hconcat([img, res[1]])
        # if len(res) > 0:
        result_img = cv2.resize(result_img, (1024, 512))
        result_img = cv2.putText(result_img, txt, org, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow(f'result', result_img)
        cv2.waitKey(1)
cv2.destroyAllWindows()
