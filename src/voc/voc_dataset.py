import gluoncv
import matplotlib.pyplot as plt

# %%

train_dataset = gluoncv.data.VOCDetection(splits=[(2007, "trainval"), (2012, "trainval")])
val_dataset = gluoncv.data.VOCDetection(splits=[(2007, "test")])
print(f"Number of training images: {len(train_dataset)}")
print(f"Number of validation images: {len(val_dataset)}")

# %%
# -- Let's visualize one sample

X, y = train_dataset[5]
boxes = y[:, :4]
class_ids = y[:, 4:5]

print(f"Image size: {X.shape}")
print(f"Number of objects: {boxes.shape}")
print(f"Bounding boxes:")
print(boxes)
print(f"class IDs:")
print(class_ids)

# %%
# -- Visualize the image with bounding boxes

gluoncv.utils.viz.plot_bbox(X.asnumpy(), boxes, scores=None, labels=class_ids, class_names=train_dataset.classes)
plt.show()

