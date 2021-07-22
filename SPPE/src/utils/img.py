import numpy as np
import cv2
import paddle


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = paddle.to_tensor(img).cast('float32')
    if img.max() > 1:
        img /= 255
    return img

def crop_dets(img, boxes, height, width):
    img = im_to_torch(img)
    img_h = img.shape[1]
    img_w = img.shape[2]
    img[0] = img[0] - 0.406
    img[1] = img[1] - 0.457
    img[2] = img[2] -0.480

    inps = paddle.zeros((len(boxes), 3, height, width))
    pt1 = paddle.zeros((len(boxes), 2))
    pt2 = paddle.zeros((len(boxes), 2))
    for i, box in enumerate(boxes):
        upLeft = paddle.to_tensor((float(box[0]), float(box[1])))
        bottomRight = paddle.to_tensor((float(box[2]), float(box[3])))

        h = bottomRight[1] - upLeft[1]
        w = bottomRight[0] - upLeft[0]
        if w > 100:
            scaleRate = 0.2
        else:
            scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - w * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - h * scaleRate / 2)
        bottomRight[0] = max(min(img_w - 1, bottomRight[0] + w * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(min(img_h - 1, bottomRight[1] + h * scaleRate / 2), upLeft[1] + 5)

        inps[i] = cropBox(img.clone(), upLeft, bottomRight, height, width)
        pt1[i] = upLeft
        pt2[i] = bottomRight

    return inps, pt1, pt2

def transformBoxInvert_batch(pt, ul, br, inpH, inpW, resH, resW):
    """
    pt:     [n, 17, 2]
    ul:     [n, 2]
    br:     [n, 2]
    """
    num_pt = pt.shape[1]
    center = (br - 1 - ul) / 2

    size = br - ul
    size[:, 0] *= (inpH / inpW)

    lenH, _ = paddle.max(size, axis=1)   # [n,]
    lenW = lenH * (inpW / inpH)

    _pt = (pt * lenH[:, np.newaxis, np.newaxis]) / resH
    _pt[:, :, 0] = _pt[:, :, 0] - ((lenW[:, np.newaxis].repeat(1, num_pt) - 1) /
                                   2 - center[:, 0].unsqueeze(-1).repeat(1, num_pt)).clamp(min=0)
    _pt[:, :, 1] = _pt[:, :, 1] - ((lenH[:, np.newaxis].repeat(1, num_pt) - 1) /
                                   2 - center[:, 1].unsqueeze(-1).repeat(1, num_pt)).clamp(min=0)

    new_point = paddle.zeros(pt.shape)
    new_point[:, :, 0] = _pt[:, :, 0] + ul[:, 0].unsqueeze(-1).repeat(1, num_pt)
    new_point[:, :, 1] = _pt[:, :, 1] + ul[:, 1].unsqueeze(-1).repeat(1, num_pt)
    return new_point

def transformBoxInvert_batch(pt, ul, br, inpH, inpW, resH, resW):
    """
    pt:     [n, 17, 2]
    ul:     [n, 2]
    br:     [n, 2]
    """
    num_pt = pt.shape[1]
    center = (br - 1 - ul) / 2

    size = br - ul
    size[:, 0] *= (inpH / inpW)

    lenH = paddle.max(size, axis=1)   # [n,]
    lenW = lenH * (inpW / inpH)

    _pt = (pt * lenH.reshape((-1, 1, 1))) / resH
    _pt[:, :, 0] = _pt[:, :, 0] - ((lenW.unsqueeze(-1).tile((1, num_pt)) - 1) /
                                   2 - center[:, 0].unsqueeze(-1).tile((1, num_pt))).clip(min=0)
    _pt[:, :, 1] = _pt[:, :, 1] - ((lenH.unsqueeze(-1).tile((1, num_pt)) - 1) /
                                   2 - center[:, 1].unsqueeze(-1).tile((1, num_pt))).clip(min=0)

    new_point = paddle.zeros(pt.shape)
    new_point[:, :, 0] = _pt[:, :, 0] + ul[:, 0].unsqueeze(-1).tile((1, num_pt))
    new_point[:, :, 1] = _pt[:, :, 1] + ul[:, 1].unsqueeze(-1).tile((1, num_pt))
    return new_point

def cropBox(img, ul, br, resH, resW):
    ul = ul.cast("int").numpy()
    br = (br - 1).cast("int").numpy()
    # br = br.int()
    lenH = max((br[1] - ul[1]), (br[0] - ul[0])* resH / resW)
    lenW = lenH * resW / resH
    if img.dim() == 2:
        img = img[np.newaxis, :]

    box_shape = [(br[1] - ul[1]), (br[0] - ul[0])]
    pad_size = [(lenH - box_shape[0]) // 2, (lenW - box_shape[1]) // 2]
    # Padding Zeros
    if ul[1] > 0:
        img[:, :ul[1], :] = 0
    if ul[0] > 0:
        img[:, :, :ul[0]] = 0
    if br[1] < img.shape[1] - 1:
        img[:, br[1] + 1:, :] = 0
    if br[0] < img.shape[2] - 1:
        img[:, :, br[0] + 1:] = 0

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = np.array(
        [ul[0] - pad_size[1], ul[1] - pad_size[0]], np.float32)
    src[1, :] = np.array(
        [br[0] + pad_size[1], br[1] + pad_size[0]], np.float32)
    dst[0, :] = 0
    dst[1, :] = np.array([resW - 1, resH - 1], np.float32)

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    dst_img = cv2.warpAffine(torch_to_im(img), trans,
                             (resW, resH), flags=cv2.INTER_LINEAR)
    return im_to_torch(dst_img)

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def torch_to_im(img):
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))  # C*H*W
    return img