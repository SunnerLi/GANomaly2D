import torchvision_sunner.transforms as sunnerTransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch
import cv2

def visualizeAnomalyMap(img, z, z_):
    # # Get the normalized anamoly score map
    # s_map = torch.abs(z - z_)
    # s_map = torch.mean(s_map, 1)
    # min_v = torch.min(s_map)
    # max_v = torch.max(s_map)
    
    # # TODO: it cannot substract [8, 512, 30, 40] - [8, 30, 40]

    # s_map = (s_map - min_v) / (max_v - min_v)
    # s_map = torch.unsqueeze(s_map, 0)                                   # CHW -> 1CHW (for F.interpolate)
    # s_map = F.interpolate(s_map, size = (img.size(-2), img.size(-1)))   # Expand the size as same as image
    # s_map = torch.cat([s_map, s_map, s_map], 1).permute(0, 2, 3, 1)     # 1HW -> HW3
    # s_map = s_map.cpu().numpy()[0]
    # return s_map

    # # Get the normalized anamoly score map
    # s_map = torch.abs(z - z_)
    # min_v, _ = torch.min(s_map, 0)
    # max_v, _ = torch.max(s_map, 0)
    # min_v = min_v.unsqueeze(0)
    # max_v = max_v.unsqueeze(0)

    # Get the min and max value through batch, height and width
    s_map = torch.abs(z - z_).permute(0, 2, 3, 1)
    b, h, w, c = s_map.size()
    min_v, _ = torch.min(s_map, 0)
    max_v, _ = torch.max(s_map, 0)
    min_v, _ = torch.min(min_v, 0)
    max_v, _ = torch.max(max_v, 0)
    min_v, _ = torch.min(min_v, 0)
    max_v, _ = torch.max(max_v, 0)
    min_v = min_v.expand(b, h, w, c).permute(0, 3, 1, 2)
    max_v = max_v.expand(b, h, w, c).permute(0, 3, 1, 2)
    s_map = s_map.permute(0, 3, 1, 2)
    
    # Get the normalized anamoly score map
    s_map = (s_map - min_v) / (max_v - min_v + 1e-8)
    s_map = torch.mean(s_map, 1)
    s_map = torch.unsqueeze(s_map, 1)                                   # CHW -> 1CHW (for F.interpolate)
    s_map = F.interpolate(s_map, size = (img.size(-2), img.size(-1)))   # Expand the size as same as image
    s_map = torch.cat([s_map, s_map, s_map], 1).permute(0, 2, 3, 1)     # 1HW -> HW3
    s_map = s_map.cpu().numpy()[0]
    return s_map


def visualizeEncoderDecoder(img, img_, z, z_):
    s_map = visualizeAnomalyMap(img, z, z_)
    img = sunnerTransforms.asImg(img)[0, :, :, ::-1]
    img_ = sunnerTransforms.asImg(img_)[0, :, :, ::-1]
    result = np.hstack((img, img_, s_map * 255.0))
    result = result.astype(np.uint8)    
    cv2.imshow('training visualization', result)
    cv2.imwrite("training_result.png", result)
    cv2.waitKey(10)

def visualizeAnomalyImage(img, img_, z, z_):
    s_map = visualizeAnomalyMap(img, z, z_)

    # Form the shown image
    img = sunnerTransforms.asImg(img)[0]
    img_ = sunnerTransforms.asImg(img_)[0]
    result = np.hstack((img, img_, s_map * 255.0, img * s_map))
    result = result.astype(np.uint8)
    
    # --- Plot with matplotlib
    plt.imshow(img, interpolation='nearest')
    plt.imshow(s_map[:, :, 0], cmap = plt.cm.viridis, alpha = 0.5, interpolation='nearest')
    plt.show()

    # --- Plot with OpenCV
    # cv2.imshow("Anomaly visualization", result)
    # cv2.waitKey(10000)