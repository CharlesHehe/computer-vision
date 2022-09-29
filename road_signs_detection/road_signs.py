# library imports
import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

import torch
from utils import resize_image_bb, create_bb_array, show_corner_bb, transformsXY, generate_train_df, update_optimizer, \
    train_epocs, val_metrics
from model_data import RoadDataset, BB_model, get_data_loader

images_path = Path(
    '/home/charleshe/PycharmProjects/torchVisionObjectDetectionFinetuning/road_signs_detection/data/archive/images')
anno_path = Path(
    '/home/charleshe/PycharmProjects/torchVisionObjectDetectionFinetuning/road_signs_detection/data/archive/annotations')

df_train = generate_train_df(anno_path, images_path)

# label encode target
class_dict = {'speedlimit': 0, 'stop': 1, 'crosswalk': 2, 'trafficlight': 3}
df_train['class'] = df_train['class'].apply(lambda x: class_dict[x])

print(df_train.shape)
df_train.head()

# Populating Training DF with new paths and bounding boxes
new_paths = []
new_bbs = []
train_path_resized = Path(
    '/home/charleshe/PycharmProjects/torchVisionObjectDetectionFinetuning/road_signs_detection/data/archive/images_resized')
for index, row in df_train.iterrows():
    new_path, new_bb = resize_image_bb(row['filename'], train_path_resized, create_bb_array(row.values), 300)
    new_paths.append(new_path)
    new_bbs.append(new_bb)
df_train['new_path'] = new_paths
df_train['new_bb'] = new_bbs

# original
im = cv2.imread(str(df_train.values[68][8]))
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
show_corner_bb(im, df_train.values[68][9])

# after transformation
im, bb = transformsXY(str(df_train.values[68][8]), df_train.values[68][9], True)
show_corner_bb(im, bb)

df_train = df_train.reset_index()
X = df_train[['new_path', 'new_bb']]
Y = df_train['class']

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

train_ds = RoadDataset(X_train['new_path'], X_train['new_bb'], y_train, transforms=True)
valid_ds = RoadDataset(X_val['new_path'], X_val['new_bb'], y_val)

batch_size = 10

train_dl = get_data_loader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
valid_dl = get_data_loader(valid_ds, batch_size=batch_size)

model = BB_model().cuda()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.01)

train_epocs(model, optimizer, train_dl, valid_dl, epochs=15)

torch.save(model.state_dict(),
           '/home/charleshe/PycharmProjects/torchVisionObjectDetectionFinetuning/road_signs_detection/road_signs_model.pth')

# update_optimizer(optimizer, 0.001)
# train_epocs(model, optimizer, train_dl, valid_dl, epochs=10)
