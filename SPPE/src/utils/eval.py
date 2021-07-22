import paddle
from SPPE.src.utils.img import transformBoxInvert_batch
def getPrediction(hms, pt1, pt2, inpH, inpW, resH, resW):
    """
    Get keypoint location from heatmaps
    """
    assert hms.dim() == 4, 'Score maps should be 4-dim'
    maxval = paddle.max(hms.reshape((hms.shape[0], hms.shape[1], -1)), 2)
    idx = paddle.argmax(hms.reshape((hms.shape[0], hms.shape[1], -1)), 2)
    maxval = maxval.reshape((hms.shape[0], hms.shape[1], 1))
    idx = idx.reshape((hms.shape[0], hms.shape[1], 1)) + 1

    preds = idx.tile((1, 1, 2)).cast("float32")

    preds[:, :, 0] = (preds[:, :, 0] - 1) % hms.shape[3]
    preds[:, :, 1] = paddle.floor((preds[:, :, 1] - 1) / hms.shape[3])

    pred_mask = maxval.greater_than(paddle.zeros(maxval.shape)).tile((1, 1, 2)).cast("float32")
    preds *= pred_mask

    # Very simple post-processing step to improve performance at tight PCK thresholds
    """for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm = hms[i][j]
            pX, pY = int(round(float(preds[i][j][0]))), int(round(float(preds[i][j][1])))
            if 0 < pX < opt.outputResW - 1 and 0 < pY < opt.outputResH - 1:
                diff = torch.Tensor(
                    (hm[pY][pX + 1] - hm[pY][pX - 1], hm[pY + 1][pX] - hm[pY - 1][pX]))
                preds[i][j] += diff.sign() * 0.25
    preds += 0.2"""

    preds_tf = paddle.zeros(preds.shape)
    preds_tf = transformBoxInvert_batch(preds, pt1, pt2, inpH, inpW, resH, resW)
    return preds, preds_tf, maxval