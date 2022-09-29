from PIL import Image
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize model with the best available weights
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.eval()
model = model.to(device)

# Initialize the inference transforms
preprocess = weights.transforms()

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(cv2_im).to(dtype=torch.uint8)
    # put it from HWC to CHW format
    frame = frame.permute((2, 0, 1)).contiguous()
    frame = frame.to(device)

    # Apply inference preprocessing transforms
    batch = [preprocess(frame)]
    # Use the model and visualize the prediction
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(frame,
                              boxes=prediction["boxes"],
                              labels=labels,
                              colors="red",
                              width=4, font_size=30)

    im = to_pil_image(box.detach())
    im = np.array(im)
    cv2.imshow('Input', im)
    # delete and release GPU memory
    del frame, batch, box, labels, prediction

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
