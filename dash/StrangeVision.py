import cv2

# from networktables import NetworkTables
from matplotlib import pyplot as plt


# NetworkTables.initialize(server="roborio-TEAM-frc.local")
# table = NetworkTables.getTable("vision")


cameraCap = cv2.VideoCapture(0)

while True:
    ret, frame = cameraCap.read()
    if not ret:
        break


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Creates the environment
# of the picture and shows it
plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()


robot_data = cv2.CascadeClassifier("")
found = robot_data.detectMultiScale(img_gray, minSize=(20, 20))
