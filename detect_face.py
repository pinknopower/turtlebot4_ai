import os
import cv2
import numpy as np
import torch
from final.utils.utils import generate_bbox, py_nms, convert_to_square
from final.utils.utils import pad, calibrate_box, processed_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda")
model_path = os.path.join(BASE_DIR, "models")
pnet = torch.jit.load(os.path.join(model_path, "PNet.pth")).to(device).eval()
rnet = torch.jit.load(os.path.join(model_path, "RNet.pth")).to(device).eval()
onet = torch.jit.load(os.path.join(model_path, "ONet.pth")).to(device).eval()
softmax_p = torch.nn.Softmax(dim=0)
softmax_r = torch.nn.Softmax(dim=1)
softmax_o = torch.nn.Softmax(dim=1)

def predict_pnet(infer_data):
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)
    infer_data = torch.unsqueeze(infer_data, dim=0)

    cls_prob, bbox_pred, _ = pnet(infer_data)
    cls_prob = torch.squeeze(cls_prob)
    cls_prob = softmax_p(cls_prob)
    bbox_pred = torch.squeeze(bbox_pred)

    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()


def predict_rnet(infer_data):
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)

    cls_prob, bbox_pred, _ = rnet(infer_data)
    cls_prob = softmax_r(cls_prob)

    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()


def predict_onet(infer_data):
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)

    cls_prob, bbox_pred, landmark_pred = onet(infer_data)
    cls_prob = softmax_o(cls_prob)

    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy(), landmark_pred.detach().cpu().numpy()


def detect_pnet(im, min_face_size, scale_factor, thresh):
    """通过pnet筛选box和landmark
    参数：
      im:输入图像[h,2,3]
    """
    net_size = 12

    current_scale = float(net_size) / min_face_size
    im_resized = processed_image(im, current_scale)
    _, current_height, current_width = im_resized.shape
    all_boxes = list()

    while min(current_height, current_width) > net_size:

        cls_cls_map, reg = predict_pnet(im_resized)
        boxes = generate_bbox(cls_cls_map[1, :, :], reg, current_scale, thresh)
        current_scale *= scale_factor
        im_resized = processed_image(im, current_scale)
        _, current_height, current_width = im_resized.shape

        if boxes.size == 0:
            continue

        keep = py_nms(boxes[:, :5], 0.5, mode='Union')
        boxes = boxes[keep]
        all_boxes.append(boxes)
    if len(all_boxes) == 0:
        return None
    all_boxes = np.vstack(all_boxes)

    keep = py_nms(all_boxes[:, 0:5], 0.7, mode='Union')
    all_boxes = all_boxes[keep]

    bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
    bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1

    boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                         all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                         all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                         all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                         all_boxes[:, 4]])
    boxes_c = boxes_c.T

    return boxes_c



def detect_rnet(im, dets, thresh):
    """通过rent选择box
        参数：
          im：输入图像
          dets:pnet选择的box，是相对原图的绝对坐标
        返回值：
          box绝对坐标
    """
    h, w, c = im.shape

    dets = convert_to_square(dets)
    dets[:, 0:4] = np.round(dets[:, 0:4])

    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
    delete_size = np.ones_like(tmpw) * 20
    ones = np.ones_like(tmpw)
    zeros = np.zeros_like(tmpw)
    num_boxes = np.sum(np.where((np.minimum(tmpw, tmph) >= delete_size), ones, zeros))
    cropped_ims = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
    for i in range(int(num_boxes)):

        if tmph[i] < 20 or tmpw[i] < 20:
            continue
        tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
        try:
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            img = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_LINEAR)
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 128
            cropped_ims[i, :, :, :] = img
        except:
            continue
    cls_scores, reg = predict_rnet(cropped_ims)
    cls_scores = cls_scores[:, 1]
    keep_inds = np.where(cls_scores > thresh)[0]
    if len(keep_inds) > 0:
        boxes = dets[keep_inds]
        boxes[:, 4] = cls_scores[keep_inds]
        reg = reg[keep_inds]
    else:
        return None

    keep = py_nms(boxes, 0.6, mode='Union')
    boxes = boxes[keep]

    boxes_c = calibrate_box(boxes, reg[keep])
    return boxes_c



def detect_onet(im, dets, thresh):
    """将onet的选框继续筛选基本和rnet差不多但多返回了landmark"""
    h, w, c = im.shape
    dets = convert_to_square(dets)
    dets[:, 0:4] = np.round(dets[:, 0:4])
    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
    num_boxes = dets.shape[0]
    cropped_ims = np.zeros((num_boxes, 3, 48, 48), dtype=np.float32)
    for i in range(num_boxes):
        tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
        tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
        img = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_LINEAR)
        img = img.transpose((2, 0, 1))
        img = (img - 127.5) / 128
        cropped_ims[i, :, :, :] = img

    cls_scores, reg, landmark = predict_onet(cropped_ims)

    cls_scores = cls_scores[:, 1]
    keep_inds = np.where(cls_scores > thresh)[0]
    if len(keep_inds) > 0:
        boxes = dets[keep_inds]
        boxes[:, 4] = cls_scores[keep_inds]
        reg = reg[keep_inds]
        landmark = landmark[keep_inds]
    else:
        return None, None

    w = boxes[:, 2] - boxes[:, 0] + 1

    h = boxes[:, 3] - boxes[:, 1] + 1
    landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
    landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
    boxes_c = calibrate_box(boxes, reg)

    keep = py_nms(boxes_c, 0.6, mode='Minimum')
    boxes_c = boxes_c[keep]
    landmark = landmark[keep]
    return boxes_c, landmark



def infer_image(im):

    boxes_c = detect_pnet(im, 20, 0.79, 0.9)
    if boxes_c is None:
        return None, None

    boxes_c = detect_rnet(im, boxes_c, 0.6)
    if boxes_c is None:
        return None, None

    boxes_c, landmark = detect_onet(im, boxes_c, 0.7)
    if boxes_c is None:
        return None, None

    return boxes_c, landmark
