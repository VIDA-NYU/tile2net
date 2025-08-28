from __future__ import annotations

import itertools
from collections import namedtuple

Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!
    'color',  # The color of this label
])

labels = [
    #       name                     id    trainId     color
    Label('Sidewalk', 1, 0, (0, 0, 255)),
    Label('Road', 2, 1, (0, 128, 0)),
    Label('Crosswalk', 3, 2, (255, 0, 0)),
    Label('Background', 4, 3, (0, 0, 0)),
]

name2label = {label.name: label for label in labels}
# id to label object
id2label = {label.id: label for label in labels}
# trainId to label object
trainId2label = {label.trainId: label for label in reversed(labels)}
# label2trainid
label2trainid = {label.id: label.trainId for label in labels}
# trainId to label object
trainId2name = {label.trainId: label.name for label in labels}
trainId2color = {label.trainId: label.color for label in labels}
num_classes = len(labels)
ignore_label = -1
trainid_to_name = {}
color_mapping = []
