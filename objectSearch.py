import tensorflow.keras
from paz.applications import SSD512COCO
import cv2
from paz.abstract.messages import Box2D




detect = SSD512COCO()

image = cv2.imread('test.jpg')
# apply directly to an image (numpy-array)
inferences = detect(image)



# for item in inferences['boxes2D']:
#
#     cv2.rectangle(image, (item.coordinates[0], item.coordinates[1] ), (item.coordinates[2], item.coordinates[3]), (0, 255, 255), 10)
#     print(item)

cv2.imwrite("test3.jpg", image)