import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision

# Hyper-params
input_size = 1024
IN_SCALE = 1024//input_size 
MODEL_SCALE = 4

def plot(image, anns, conf):
    xs = anns['x'].to_list()
    ys = anns['y'].to_list()
    ws = anns['w'].to_list()
    hs = anns['h'].to_list()
    cs = anns['conf'].to_list()
    for x, y, w, h, c in zip(xs, ys, ws, hs, cs):
        if c >= conf:
            xmax = x+w
            ymax = y+h
            image = cv2.rectangle(image, (x,y), (xmax, ymax), (0, 220, 0), 2)
    return image

def pool(data):
    stride = 3
    for y in np.arange(1,data.shape[1]-1, stride):
        for x in np.arange(1, data.shape[0]-1, stride):
            a_2d = data[x-1:x+2, y-1:y+2]
            max = np.asarray(np.unravel_index(np.argmax(a_2d), a_2d.shape))            
            for c1 in range(3):
                for c2 in range(3):
                    if not (c1== max[0] and c2 == max[1]):
                        data[x+c1-1, y+c2-1] = -1
    return data


def pred2box(hm, regr, thresh=0.99):
    pred = hm > thresh
    pred_center = np.where(hm>thresh)
    pred_r = regr[:,pred].T

    boxes = []
    scores = hm[pred]
    for i, b in enumerate(pred_r):
        arr = np.array([pred_center[1][i]*MODEL_SCALE-b[0]*input_size//2, pred_center[0][i]*MODEL_SCALE-b[1]*input_size//2, 
                      int(b[0]*input_size), int(b[1]*input_size)])
        arr = np.clip(arr, 0, input_size)
        boxes.append(arr)
    return np.asarray(boxes), scores

def toANN(bbox, scores):
    ann = pd.DataFrame()
    ann['x'] = bbox[:,0].astype(int)
    ann['y'] = bbox[:,1].astype(int)
    ann['w'] = bbox[:,2].astype(int)
    ann['h'] = bbox[:,3].astype(int)
    ann['conf'] = scores
    return ann

def fullPLOT(image, hm, regr, conf, iou):
    hm = torch.sigmoid(hm).numpy()
    hm = pool(hm)
    bbox, scores = pred2box(hm[0,0], regr[0], thresh=conf)
    if bbox.shape[0]<=1:
        return image, len(bbox)
    bbox_torch = torch.FloatTensor([[x[0], x[1], x[0]+x[2], x[1]+x[3]] for x in bbox])
    bbox_pass = torchvision.ops.nms(bbox_torch, torch.FloatTensor(scores), iou)
    bbox = bbox[bbox_pass.tolist()]
    scores = scores[bbox_pass.tolist()]
    ann = toANN(bbox, scores)
    sample = plot(image, ann, 0.5)
    return sample, len(bbox)