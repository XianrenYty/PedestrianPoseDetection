# -*- coding: utf-8 -*-
import paddle
import numpy as np

''' Constant Configuration '''
delta1 = 1
mu = 1.7
delta2 = 2.65
gamma = 22.48
scoreThreds = 0.3
matchThreds = 5
areaThres = 0  # 40 * 40.5
alpha = 0.1
#pool = ThreadPool(4)


def pose_nms(bboxes, bbox_scores, pose_preds, pose_scores):
    """
    Parametric Pose NMS algorithm
    bboxes:         bbox locations list (n, 4)
    bbox_scores:    bbox scores list (n,)
    pose_preds:     pose locations list (n, 17, 2)
    pose_scores:    pose scores list    (n, 17, 1)
    """
    global ori_pose_preds, ori_pose_scores, ref_dists

    pose_scores = pose_scores.numpy()
    pose_scores[pose_scores == 0] = 1e-5
    pose_scores = paddle.to_tensor(pose_scores)

    final_result = []

    ori_bboxes = bboxes.clone()
    ori_bbox_scores = bbox_scores.clone()
    ori_pose_preds = pose_preds.clone()
    ori_pose_scores = pose_scores.clone()

    xmax = bboxes[:, 2]
    xmin = bboxes[:, 0]
    ymax = bboxes[:, 3]
    ymin = bboxes[:, 1]

    widths = xmax - xmin
    heights = ymax - ymin
    ref_dists = alpha * paddle.maximum(widths, heights)

    nsamples = bboxes.shape[0]
    human_scores = pose_scores.mean(axis=1)

    human_ids = np.arange(nsamples)
    # Do pPose-NMS
    pick = []
    merge_ids = []
    while human_scores.shape[0] != 0:
        # Pick the one with highest score
        pick_id = paddle.argmax(human_scores)
        pick.append(human_ids[pick_id])
        # num_visPart = torch.sum(pose_scores[pick_id] > 0.2)

        # Get numbers of match keypoints by calling PCK_match
        ref_dist = ref_dists.gather(paddle.to_tensor(human_ids[pick_id]))
        simi = get_parametric_distance(pick_id, pose_preds, pose_scores, ref_dist)
        num_match_keypoints = PCK_match(pose_preds.gather(pick_id).squeeze(0), pose_preds, ref_dist)

        # Delete humans who have more than matchThreds keypoints overlap and high similarity
        delete_ids = np.arange(human_scores.shape[0])
        delete_ids = delete_ids[(simi.numpy() > gamma) | (num_match_keypoints.numpy() >= matchThreds)]

        if delete_ids.shape[0] == 0:
            delete_ids = pick_id
        #else:
        #    delete_ids = torch.from_numpy(delete_ids)

        merge_ids.append(human_ids[delete_ids])
        pose_preds = paddle.to_tensor(np.delete(pose_preds.numpy(), delete_ids, axis=0))
        pose_scores = paddle.to_tensor(np.delete(pose_scores.numpy(), delete_ids, axis=0))
        human_ids = np.delete(human_ids, delete_ids)
        human_scores = paddle.to_tensor(np.delete(human_scores.numpy(), delete_ids, axis=0))
        bbox_scores = paddle.to_tensor(np.delete(bbox_scores.numpy(), delete_ids, axis=0))

    assert len(merge_ids) == len(pick)
    bboxs_pick = ori_bboxes.gather(paddle.to_tensor(pick))
    preds_pick = ori_pose_preds.gather(paddle.to_tensor(pick))
    scores_pick = ori_pose_scores.gather(paddle.to_tensor(pick))
    bbox_scores_pick = ori_bbox_scores.gather(paddle.to_tensor(pick))
    #final_result = pool.map(filter_result, zip(scores_pick, merge_ids, preds_pick, pick, bbox_scores_pick))
    #final_result = [item for item in final_result if item is not None]

    for j in range(len(pick)):
        ids = paddle.arange(pose_preds.shape[1])
        max_score = paddle.max(scores_pick[j, :, 0].gather(ids))

        if max_score < scoreThreds:
            continue

        # Merge poses
        merge_id = merge_ids[j]
        merge_pose, merge_score = p_merge_fast(
            preds_pick[j],
            ori_pose_preds.gather(paddle.to_tensor(merge_id)),
            ori_pose_scores.gather(paddle.to_tensor(merge_id)),
            ref_dists[int(pick[j])])

        max_score = paddle.max(merge_score.gather(ids))
        if max_score < scoreThreds:
            continue

        xmax = max(merge_pose[:, 0])
        xmin = min(merge_pose[:, 0])
        ymax = max(merge_pose[:, 1])
        ymin = min(merge_pose[:, 1])

        if 1.5 ** 2 * (xmax - xmin) * (ymax - ymin) < areaThres:
            continue

        final_result.append({
            'bbox': bboxs_pick[j],
            'bbox_score': bbox_scores_pick[j],
            'keypoints': merge_pose - 0.3,
            'kp_score': merge_score,
            'proposal_score': paddle.mean(merge_score) + bbox_scores_pick[j] + 1.25 * max(merge_score)
        })

    return final_result

def p_merge_fast(ref_pose, cluster_preds, cluster_scores, ref_dist):
    """
    Score-weighted pose merging
    INPUT:
        ref_pose:       reference pose          -- [17, 2]
        cluster_preds:  redundant poses         -- [n, 17, 2]
        cluster_scores: redundant poses score   -- [n, 17, 1]
        ref_dist:       reference scale         -- Constant
    OUTPUT:
        final_pose:     merged pose             -- [17, 2]
        final_score:    merged score            -- [17]
    """
    dist = paddle.sqrt(paddle.sum(
        paddle.pow(ref_pose.unsqueeze(0) - cluster_preds, 2),
        axis=2
    ))

    kp_num = 17
    ref_dist = min(ref_dist, 15)

    mask = (dist <= ref_dist)

    if cluster_preds.dim() == 2:
        cluster_preds.unsqueeze_(0)
        cluster_scores.unsqueeze_(0)
    if mask.dim() == 1:
        mask.unsqueeze_(0)

    # Weighted Merge
    masked_scores = cluster_scores.multiply(mask.cast("float32").unsqueeze(-1))
    normed_scores = masked_scores / paddle.sum(masked_scores, axis=0)

    final_pose = paddle.multiply(cluster_preds, normed_scores.tile((1, 1, 2))).sum(axis=0)
    final_score = paddle.multiply(masked_scores, normed_scores).sum(axis=0)
    return final_pose, final_score


def get_parametric_distance(i, all_preds, keypoint_scores, ref_dist):
    pick_preds = all_preds.gather(i).squeeze(0)
    pred_scores = keypoint_scores.gather(i).squeeze(0)
    dist = paddle.sqrt(paddle.sum(
        paddle.pow(pick_preds.unsqueeze(0) - all_preds, 2),
        axis=2
    ))
    mask = (dist <= 1)

    # Define a keypoints distance
    score_dists = paddle.zeros((all_preds.shape[0], all_preds.shape[1]))
    keypoint_scores.squeeze_()
    if keypoint_scores.dim() == 1:
        keypoint_scores.unsqueeze_(0)
    if pred_scores.dim() == 1:
        pred_scores.unsqueeze_(1)
    # The predicted scores are repeated up to do broadcast
    pred_scores = pred_scores.tile((1, all_preds.shape[0])).transpose((1, 0))

    pred_scores = pred_scores.numpy()
    keypoint_scores = keypoint_scores.numpy()
    score_dists = score_dists.numpy()
    mask = mask.numpy()
    score_dists[mask] = np.tanh(pred_scores[mask] / delta1) * \
                        np.tanh(keypoint_scores[mask] / delta1)
    score_dists = paddle.to_tensor(score_dists)
    # score_dists[mask] = paddle.tanh(pred_scores[mask] / delta1) * \
    #                     paddle.tanh(keypoint_scores[mask] / delta1)

    point_dist = paddle.exp((-1) * dist / delta2)
    final_dist = paddle.sum(score_dists, axis=1) + mu * paddle.sum(point_dist, axis=1)

    return final_dist


def PCK_match(pick_pred, all_preds, ref_dist):
    dist = paddle.sqrt(paddle.sum(
        paddle.pow(pick_pred.unsqueeze(0) - all_preds, 2),
        axis=2
    ))
    ref_dist = min(ref_dist, 7)
    dist = dist / ref_dist <= 1
    dist = dist.cast("int")
    num_match_keypoints = paddle.sum(
        dist,
        axis=1
    )

    return num_match_keypoints
