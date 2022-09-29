from model_data import BB_model
import torch
from utils import read_image, show_corner_bb
import cv2
from model_data import RoadDataset
import pandas as pd
import numpy as np

model = BB_model().cuda()
model.load_state_dict(torch.load(
    '/home/charleshe/PycharmProjects/torchVisionObjectDetectionFinetuning/road_signs_detection/road_signs_model.pth'))
model.eval()

# resizing test image
im = read_image(
    '/home/charleshe/PycharmProjects/torchVisionObjectDetectionFinetuning/road_signs_detection/data/archive/images/road876.png')
im = cv2.resize(im, (int(1.49 * 300), 300))
cv2.imwrite(
    '/home/charleshe/PycharmProjects/torchVisionObjectDetectionFinetuning/road_signs_detection/data/archive/road_signs_test/road876.jpg',
    cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

# test Dataset
test_ds = RoadDataset(pd.DataFrame([{
    'path': '/home/charleshe/PycharmProjects/torchVisionObjectDetectionFinetuning/road_signs_detection/data/archive/road_signs_test/road876.jpg'}])[
                          'path'],
                      pd.DataFrame([{'bb': np.array([0, 0, 0, 0])}])['bb'], pd.DataFrame([{'y': [0]}])['y'])
x, y_class, y_bb = test_ds[0]

xx = torch.FloatTensor(x[None,])
xx.shape
# prediction
out_class, out_bb = model(xx.cuda())
out_class, out_bb

# predicted class
class_dict = {'speedlimit': 0, 'stop': 1, 'crosswalk': 2, 'trafficlight': 3}
for lable, value in class_dict.items():
    if value == torch.max(out_class, 1)[1].item():
        print(lable)

bb_hat = out_bb.detach().cpu().numpy()
bb_hat = bb_hat.astype(int)
show_corner_bb(im, bb_hat[0])
