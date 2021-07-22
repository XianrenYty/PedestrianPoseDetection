import cv2
import numpy as np
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)

"""COCO_PAIR = [(0, 1), (0, 2), (1, 3), (2, 4),  # Head
             (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
             (17, 11), (17, 12),  # Body
             (11, 13), (12, 14), (13, 15), (14, 16)]"""
COCO_PAIR = [(0, 13), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (13, 7), (13, 8),  # Body
             (7, 9), (8, 10), (9, 11), (10, 12)]
POINT_COLORS = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
LINE_COLORS = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), (77, 255, 222),
               (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77), (77, 222, 255),
               (255, 156, 127), (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

MPII_PAIR = [(8, 9), (11, 12), (11, 10), (2, 1), (1, 0), (13, 14), (14, 15), (3, 4), (4, 5),
             (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)]

def draw_single(frame, pts, joint_format='coco'):
    if joint_format == 'coco':
        l_pair = COCO_PAIR
        p_color = POINT_COLORS
        line_color = LINE_COLORS
    elif joint_format == 'mpii':
        l_pair = MPII_PAIR
        p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED,BLUE,BLUE]
    else:
        NotImplementedError

    part_line = {}
    pts = np.concatenate((pts, np.expand_dims((pts[1, :] + pts[2, :]) / 2, 0)), axis=0)
    for n in range(pts.shape[0]):
        if pts[n, 2] <= 0.05:
            continue
        cor_x, cor_y = int(pts[n, 0]), int(pts[n, 1])
        part_line[n] = (cor_x, cor_y)
        cv2.circle(frame, (cor_x, cor_y), 3, p_color[n], -1)

    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            cv2.line(frame, start_xy, end_xy, line_color[i], int(1*(pts[start_p, 2] + pts[end_p, 2]) + 1))
    return frame