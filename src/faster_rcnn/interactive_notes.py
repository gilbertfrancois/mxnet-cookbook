import os
import matplotlib.pyplot as plt
import gluoncv
import mxnet.gluon.nn as nn
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd

# %%

net = gluoncv.model_zoo.get_model("VGG16", pretrained=True)

# %%
dummy_image = nd.zeros(shape=(1, 3, 800, 800))

# %%
# -- Take the part of the feature extractor where the output size is 800 // 16.
#    The image is 800x800, the feature map is 50x50. Every pixel in the feature map
#    corresponds to 16x16 pixels in the image.

faster_rcnn_fe_extractor = net.features[:30]
dummy_fe_map = faster_rcnn_fe_extractor(dummy_image)

subsample = dummy_image.shape[2] // dummy_fe_map.shape[2]

# %%
# -- Create anchor boxes

anchor_ratios = nd.array([0.5, 1.0, 2.0])
anchor_scales = nd.array([8, 16, 32])
anchor_base = nd.zeros(shape=(len(anchor_ratios) * len(anchor_scales), 4))
print(anchor_base)

# %%
# -- Create anchor boxes for the first feature pixel (0, 0)

_center_x = subsample // 2
_center_y = subsample // 2
for i in range(anchor_ratios.shape[0]):
    for j in range(anchor_scales.shape[0]):
        h = subsample * anchor_scales[j] * nd.sqrt(anchor_ratios[i])
        w = subsample * anchor_scales[j] * nd.sqrt(1.0 / anchor_ratios[i])
        print(f"Box size: {w.asscalar():0.1f} x {h.asscalar():0.1f}")
        index = i * len(anchor_scales) + j
        anchor_base[index, 0] = _center_y - h / 2.0
        anchor_base[index, 1] = _center_x - w / 2.0
        anchor_base[index, 2] = _center_y + h / 2.0
        anchor_base[index, 3] = _center_x + w / 2.0
print("\nAnchor boxes:")
print(anchor_base)

# %%

fe_size = dummy_fe_map.shape[2]
center_x = nd.arange(8, (fe_size) * 16, 16)
center_y = nd.arange(8, (fe_size) * 16, 16)
centers = nd.zeros(shape=(center_x.shape[0] * center_y.shape[0], 2))

# %%

index = 0
for x in range(len(center_x)):
    for y in range(len(center_y)):
        centers[index, 1] = center_x[x]
        centers[index, 0] = center_y[y]
        index += 1

anchor_boxes = nd.zeros(shape=(fe_size * fe_size * anchor_scales.shape[0] * anchor_ratios.shape[0], 4))

# %%
# -- plot centers

_img = np.zeros(shape=(800, 800, 3), dtype=np.int8)
plt.figure()
plt.imshow(_img)
plt.scatter(centers.asnumpy()[:,0], centers.asnumpy()[:,1], marker=".")
plt.title("Anchor centers")
plt.show()

# %%

index = 0
for center in centers:
    _center_y = center[0]
    _center_x = center[1]
    for i in range(anchor_ratios.shape[0]):
        for j in range(anchor_scales.shape[0]):
            h = subsample * anchor_scales[j] * nd.sqrt(anchor_ratios[i])
            w = subsample * anchor_scales[j] * nd.sqrt(1.0 / anchor_ratios[i])
            # index = i * len(anchor_scales) + j
            anchor_boxes[index, 0] = _center_y - h / 2.0
            anchor_boxes[index, 1] = _center_x - w / 2.0
            anchor_boxes[index, 2] = _center_y + h / 2.0
            anchor_boxes[index, 3] = _center_x + w / 2.0
            index += 1
print(anchor_boxes.shape)

# %%

bbox = nd.array([[20, 30, 400, 500], [300, 400, 500, 600]], dtype=np.float32)
labels = nd.array([6, 8])

# %%
index_inside = nd.array(np.where(
    (anchor_boxes.asnumpy()[:, 0] >= 0) & (anchor_boxes.asnumpy()[:, 1] >= 0) &
    (anchor_boxes.asnumpy()[:, 2] <= 800) & (anchor_boxes.asnumpy()[:, 3] <= 800)
)[0], dtype=np.int)
valid_anchor_boxes = anchor_boxes[index_inside]
print(f"Valid anchor boxes shape: {valid_anchor_boxes.shape}")


# %%

ious = nd.zeros(shape=(valid_anchor_boxes.shape[0], 2), dtype=np.float32)
print(bbox)
for num1, i in enumerate(valid_anchor_boxes):
    xa1, ya1, xa2, ya2 = i
    anchor_area = (ya2 - ya1) * (xa2 - xa1)
    for num2, j in enumerate(bbox):
        xb1, yb1, xb2, yb2 = j
        box_area = (yb2 - yb1) * (xb2 - xb1)

        inter_x1 = max([xb1, xa1])
        inter_y1 = max([yb1, ya1])
        inter_x2 = min([xb2, xa2])
        inter_y2 = min([yb2, ya2])
        
        if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
            
            inter_x2 = min([xb2, xa2])
            inter_y2 = min([yb2, ya2])
            
            iter_area = (inter_y2 - inter_y1) *  (inter_x2 - inter_x1)
            iou = iter_area / (anchor_area+ box_area - iter_area)
        else:
            iou = 0.
        ious[num1, num2] = iou
print(ious.shape)
ious = ious.asnumpy()
# %%

gt_argmax_ious = ious.argmax(axis=0)
gt_max_ious = ious.max(axis=0)
argmax_ious = ious.argmax(axis=1)
max_ious = ious.max(axis=1)
print("Ground truth: ", gt_argmax_ious.shape, gt_max_ious.shape)
print("Anchor boxes: ", argmax_ious.shape, max_ious.shape)

# %%
# -- Find the anchor boxes which have this max_ious (gt_max_ious)

# %%

# %%

gluoncv.utils.viz.plot_bbox(_img, bbox, scores=None, labels=labels)
plt.title("Ground truth bounding boxes")
plt.show()

# %%
print(type(valid_anchor_boxes))
all_boxes = nd.concat(valid_anchor_boxes[gt_argmax_ious], bbox)
all_labels = nd.concat(nd.ones(shape=(valid_anchor_boxes[gt_argmax_ious]).shape), labels)
gluoncv.utils.viz.plot_bbox(_img, all_boxes, scores=None, labels=all_labels)
plt.title("IOU anchor boxes")
plt.show()

# %%
# -- Let's put thresholds to some variables

pos_iou_threshold = 0.7
neg_iou_threshold = 0.3

# %%
label = np.zeros(shape=ious.shape)
label[max_ious < neg_iou_threshold] = 0
label[gt_argmax_ious] = 1
label[max_ious >= pos_iou_threshold] = 1


# %%
label_rpn = nd.zeros(shape=max_ious.shape) * -1
label_rpn = nd.where(max_ious < neg_iou_threshold, nd.zeros(shape=label_rpn.shape), label_rpn)
label_rpn[[gt_argmax_ious]] = 1
label_rpn = nd.where(max_ious > pos_iou_threshold, nd.ones(shape=label_rpn.shape), label_rpn)

