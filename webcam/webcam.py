from PIL import Image
import cv2
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Step 1: Initialize model with the best available weights
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.eval()
model = model.to(device)

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3600)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2_im2 = np.empty([cv2_im.shape[2], cv2_im.shape[0], cv2_im.shape[1]])
    cv2_im2[0, :, :] = cv2_im[:, :, 0]
    cv2_im2[1, :, :] = cv2_im[:, :, 1]
    cv2_im2[2, :, :] = cv2_im[:, :, 2]
    frame = torch.from_numpy(cv2_im2).to(dtype=torch.uint8)
    frame = frame.to(device)

    # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    # Step 3: Apply inference preprocessing transforms
    batch = [preprocess(frame)]
    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(frame,
                              boxes=prediction["boxes"],
                              labels=labels,
                              colors="red",
                              width=4, font_size=30)

    im = to_pil_image(box.detach())
    # im.show()
    im = np.array(im)
    # im = np.asarray(im, dtype=np.float32)
    cv2.imshow('Input', im)
    del frame, batch, box, labels, prediction

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
