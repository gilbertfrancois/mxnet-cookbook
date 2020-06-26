import os
import matplotlib.pyplot as plt
import gluoncv
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd

# %%
# -- Load a pretrained model

net = gluoncv.model_zoo.get_model("faster_rcnn_resnet50_v1b_voc", pretrained=True)

# %%
# -- Load test image
root_folder = os.path.expanduser("~/Development/git/mxnet-cookbook/_resources")
image_filename = os.path.join(root_folder, "bike-and-bus.jpg")

x, image = gluoncv.data.transforms.presets.rcnn.load_test(image_filename)
print(f"    x: {x.shape}, xmin: {nd.min(x).asscalar()}, xmax: {nd.max(x).asscalar()}")
print(f"image: {image.shape}, xmin: {np.min(image)}, xmax: {np.max(image)}")

# %%
# -- Feed image tensor to the network and get the predictions.

box_ids, scores, bboxes = net(x)

# %%
# -- Let's dive deeper in the source code of faster r-cnn.
#    Set a breakpoint at the inference line above and step into the function. You should end up in the file:
#    https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/rcnn/faster_rcnn/faster_rcnn.py
#
#    IMPORTANT: FILE NAMES AND LINE NUMBERS ARE FOR Gluon-CV version 0.7.0

# import necessary libraries when we stop in the middle of the code and want to do interactive things in the console.

import matplotlib.pyplot as plt
import gluoncv
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd

# %%
# -- Breakpoint file: faster_rcnn.py, line: 365
# -- Plot features after inference through the feature extractor. This returns the features with shape
#    (N, 1024, w//stride, h//stride) and stride = 16 for Resnet 50.

# feat: (B, 1024, height // stride, width // stride)

plt.figure()
N = 32
fig, axs = plt.subplots(N, N, figsize=(32, 24))
fig.tight_layout()
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
for i in range(N):
    for j in range(N):
        k = i * N + j
        axs[i][j].imshow(feat[0, k, :, :].asnumpy(), cmap="magma")
        axs[i][j].set_axis_off()
plt.show()

# Save features to disk for later reference.
np.save("feat.npy", feat.asnumpy())

# %%
# -- Breakpoint file: faster_rcnn.py, line: 377 + 1
# -- Plot ROIs after feed forward the features into the RPN network

# Reload image since it might be out of scope at the point where this plot function is called.
root_folder = os.path.expanduser("~/Development/git/mxnet-cookbook/_resources")
image_filename = os.path.join(root_folder, "bike-and-bus.jpg")
x1, image1 = gluoncv.data.transforms.presets.rcnn.load_test(image_filename)

# The current implementation ignores the output of the rpn_scores. So we have to save them ourselves
# before they are gone.
rpn_scores = _

# rpn_box.shape: (B, N, C * 4)
# rpn_scores.shape:  (B, N, 1)

# Plot the region proposals produced by the RPN network. Note that we only plot the proposals
# with threshold > 0.9 to prevent cluttering the plot.
ax = gluoncv.utils.viz.plot_bbox(image1, rpn_box[0], rpn_scores[0], thresh=0.9, linewidth=2)
plt.show()

# %%
# -- Breakpoint file: faster_rcnn.py, line: 400
# -- Save pooled features to disk

# pooled_feat: (N, 1024, 14, 14)
np.save("pooled_feat.npy", pooled_feat.asnumpy())

# %%
# -- Breakpoint file: faster_rcnn.py, line: 408
# -- Save top features after feeding pooled_feat to TopFeatures network.

# top_feat: (N, 2048, 7, 7)
np.save("top_feat.npy", top_feat.asnumpy())
# box_feat: (N, 2048, 1, 1)
np.save("box_feat.npy", box_feat.asnumpy())




# %%
# -- Plot RPN after cls, scores, boxes predictions

# rpn_box (1x300x4)
# score   (20x300x1)
# cls_id  (20x300x1)
# bbox    (20x300x1)

_cls_id = nd.transpose(cls_id, axes=(2, 1, 0))
_cls_id = nd.max(_cls_id[0], axis=1)

_score = nd.transpose(score, axes=(2, 1, 0))
_score = nd.max(_score, axis=2)

classes = self.classes

# %%
# -- Broadcast rpn boxes from (1, 300, 4) to (20, 300, 4)

rpn_boxes = nd.repeat(rpn_box, 20, axis=0)
rpn_boxes = rpn_boxes.reshape((-3, 0))
rpn_boxes = nd.expand_dims(rpn_boxes, axis=0)
idx = nd.linspace(0, 6000, 6000)

# Reload image since it might be out of scope at the point where this plot function is called.
root_folder = os.path.expanduser("~/Development/git/mxnet-cookbook/_resources")
image_filename = os.path.join(root_folder, "bike-and-bus.jpg")
x1, image1 = gluoncv.data.transforms.presets.rcnn.load_test(image_filename)
# Plot the RPN and scores
ax = gluoncv.utils.viz.plot_bbox(image1, rpn_boxes[0], scores[0], labels=idx)
plt.show()

ax = gluoncv.utils.viz.plot_bbox(image1, bboxes[0], scores[0], labels=ids[0], class_names=self.classes)
plt.show()

# %%
# -- breakpoint line 447 + 1
import numpy as np

np.save("cls_ids.npy", cls_ids[0].asnumpy())
np.save("scores.npy", scores[0].asnumpy())
np.save("bbox.npy", bbox.asnumpy())
np.save("rpn_boxes.npy", rpn_boxes[0].asnumpy())
np.save("feat.npy", feat[0].asnumpy())

_cls_ids = cls_ids[0].asnumpy()
_rpn_boxes = rpn_boxes[0].asnumpy()
_scores = scores[0].asnumpy()
_bbox = bbox.asnumpy()

print(_cls_ids.shape)
print(_scores.shape)
print(_bbox.shape)
print(_rpn_boxes.shape)

import numpy as np

np.save("cls_ids.npy", cls_ids[0].asnumpy())
np.save("scores.npy", scores[0].asnumpy())
np.save("bbox.npy", bbox.asnumpy())
np.save("rpn_boxes.npy", rpn_boxes[0].asnumpy())

_cls_ids = cls_ids[0].asnumpy()
_rpn_boxes = rpn_boxes[0].asnumpy()
_scores = scores[0].asnumpy()
_bbox = bbox.asnumpy()

print(_cls_ids.shape)
print(_scores.shape)
print(_bbox.shape)
print(_rpn_boxes.shape)

# %%
# -- breakpoint line 468

_res = res.asnumpy()
np.save("res.npy", _res)

_res_th = _res[_res[:, 1] > 0.5]

# find indices of _res[2:] in _box_preds
indices = []
for _bb in _res_th[:, 2:]:
    print(_bb)
    for i in range(20):
        ans = np.where((_bbox[i] == _bb).all(axis=1))
        if len(ans[0]) > 0:
            indices.append(ans[0][0])

_rpn_used = _rpn_boxes[0, indices, :]
ax = gluoncv.utils.viz.plot_bbox(image1, _rpn_used, labels=np.zeros(len(_rpn_used)), class_names=("rpn",))
plt.show()
