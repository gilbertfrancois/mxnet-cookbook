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
root_folder = os.path.expanduser("~/Development/git/mxnet-cookbook/src/mxnet_cookbook/faster_rcnn")
image_filename = os.path.join(root_folder, "city.jpg")

x, image = gluoncv.data.transforms.presets.rcnn.load_test(image_filename)
print(f"    x: {x.shape}, xmin: {nd.min(x).asscalar()}, xmax: {nd.max(x).asscalar()}")
print(f"image: {image.shape}, xmin: {np.min(image)}, xmax: {np.max(image)}")

# %%
box_ids, scores, bboxes = net(x)

# %%
ax = gluoncv.utils.viz.plot_bbox(image, bboxes[0], scores[0], box_ids[0], class_names=net.classes)
plt.savefig(os.path.join(root_folder, "test_out.png"))