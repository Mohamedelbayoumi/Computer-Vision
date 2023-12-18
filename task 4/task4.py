import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox

image = cv2.imread("imtest11.jpg")
plt.imshow(image)
plt.show()

box, label, count = cv.detect_common_objects(image, confidence=0.1, model='yolov3')

output = draw_bbox(image, box, label, count)

plt.imshow(output)
plt.show()

print("Numbers of cars are:" +str(label.count('car')))
print("Nuber of trucks are:" +str(label.count('truck')))