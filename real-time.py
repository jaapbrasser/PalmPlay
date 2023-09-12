# This is an update from the repo https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection and updated to start identifying the number of fingers and print on screen
# Also changed 'is' to '==' due to syntax change between python versions
# Still all very much a WIP!

import cv2
import numpy as np
from unified_detector import Fingertips
from hand_detector.detector import SOLO, YOLO

hand_detection_method = "yolo"

if hand_detection_method == "solo":
    hand = SOLO(weights="weights/solo.h5", threshold=0.8)
elif hand_detection_method == "yolo":
    hand = YOLO(weights="weights/yolo.h5", threshold=0.8)
else:
    assert False, (
        "'"
        + hand_detection_method
        + "' hand detection does not exist. use either 'solo' or 'yolo' as hand detection method"
    )

fingertips = Fingertips(weights="weights/fingertip.h5")

cam = cv2.VideoCapture(0)
print("Unified Gesture & Fingertips Detection")

while True:
    ret, image = cam.read()

    if ret is False:
        break
    fingers = 0
    # hand detection
    tl, br = hand.detect(image=image)

    if tl and br is not None:
        cropped_image = image[tl[1] : br[1], tl[0] : br[0]]
        height, width, _ = cropped_image.shape

        # gesture classification and fingertips regression
        prob, pos = fingertips.classify(image=cropped_image)
        pos = np.mean(pos, 0)

        # post-processing
        prob = np.asarray([(p >= 0.5) * 1.0 for p in prob])
        fingers = prob.size
        for i in range(0, len(pos), 2):
            pos[i] = pos[i] * width + tl[0]
            pos[i + 1] = pos[i + 1] * height + tl[1]

        # drawing
        index = 0
        color = [
            (15, 15, 240),
            (15, 240, 155),
            (240, 155, 15),
            (240, 15, 155),
            (240, 15, 240),
        ]
        image = cv2.rectangle(image, (tl[0], tl[1]), (br[0], br[1]), (235, 26, 158), 2)
        for c, p in enumerate(prob):
            if p > 0.5:
                image = cv2.circle(
                    image,
                    (int(pos[index]), int(pos[index + 1])),
                    radius=12,
                    color=color[c],
                    thickness=-2,
                )
            index = index + 2

    if cv2.waitKey(1) & 0xFF == 27:
        break

    cv2.putText(
        image,
        f"{fingers} number of fingers",
        (15, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
    )
    # display image
    cv2.imshow("Unified Gesture & Fingertips Detection", image)

cam.release()
cv2.destroyAllWindows()
