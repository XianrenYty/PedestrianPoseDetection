import cv2
import paddle

def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = paddle.zeros((nB, nA, nG, nG), dtype='byte')
    noobj_mask = paddle.ones((nB, nA, nG, nG), dtype='byte')
    class_mask = paddle.zeros((nB, nA, nG, nG), dtype='float32')
    iou_scores = paddle.zeros((nB, nA, nG, nG), dtype='float32')
    tx = paddle.zeros((nB, nA, nG, nG), dtype='float32')
    ty = paddle.zeros((nB, nA, nG, nG), dtype='float32')
    tw = paddle.zeros((nB, nA, nG, nG), dtype='float32')
    th = paddle.zeros((nB, nA, nG, nG), dtype='float32')
    tcls = paddle.zeros((nB, nA, nG, nG, nC), dtype='float32')

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = paddle.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = paddle.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = paddle.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = paddle.min(w1, w2) * paddle.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area

def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def to_cpu(tensor: paddle.Tensor):
    return tensor.detach().cpu()

def ResizePadding(height, width):
    desized_size = (height, width)

    def resizePadding(image, **kwargs):
        old_size = image.shape[:2]
        max_size_idx = old_size.index(max(old_size))
        ratio = float(desized_size[max_size_idx]) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        if new_size > desized_size:
            min_size_idx = old_size.index(min(old_size))
            ratio = float(desized_size[min_size_idx]) / min(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])

        image = cv2.resize(image, (new_size[1], new_size[0]))
        delta_w = desized_size[1] - new_size[1]
        delta_h = desized_size[0] - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
        return image
    return resizePadding

def xywh2xyxy(x):
    assert x.ndim == 3
    y = paddle.zeros(x.shape, x.dtype)
    y[:, :, 0] = x[:, :, 0] - x[:, :, 2] / 2
    y[:, :, 1] = x[:, :, 1] - x[:, :, 3] / 2
    y[:, :, 2] = x[:, :, 0] + x[:, :, 2] / 2
    y[:, :, 3] = x[:, :, 1] + x[:, :, 3] / 2
    return y

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    assert prediction.ndim == 3
    prediction[:, :, :4] = xywh2xyxy(prediction[:, :, :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred.numpy()
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        image_pred = paddle.to_tensor(image_pred)
        # If none are remaining => process next image
        if not image_pred.shape[0]:
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        score = score.numpy()
        # Sort by it
        image_pred = image_pred.numpy()
        image_pred = image_pred[(-score).argsort()]
        image_pred = paddle.to_tensor(image_pred)
        class_confs = image_pred[:, 5:].max(1, keepdim=True)
        class_preds = image_pred[:, 5:].argmax(1, keepdim=True)
        detections = paddle.concat((image_pred[:, :5], class_confs.cast("float32"), class_preds.cast("float32")), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.shape[0]:
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            detections = detections.numpy()
            large_overlap = large_overlap.numpy()
            label_match = label_match.numpy()
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [paddle.to_tensor(detections[0])]
            detections = detections[~invalid]
            detections = paddle.to_tensor(detections)
        if keep_boxes:
            output[image_i] = paddle.stack(keep_boxes)

    return output


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = paddle.maximum(b1_x1, b2_x1)
    inter_rect_y1 = paddle.maximum(b1_y1, b2_y1)
    inter_rect_x2 = paddle.minimum(b1_x2, b2_x2)
    inter_rect_y2 = paddle.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area = paddle.clip(inter_rect_x2 - inter_rect_x1 + 1, min=0) * paddle.clip(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou