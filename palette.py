import json
import numpy as np
from PIL import Image

# 用于生成调色板
label = Image.open(r"D:\2PM_2023\basic\segmentation\VOCdevkit\VOC2012\SegmentationObject/2007_001284.png")

palette = label.getpalette()
# reshape成三列的列表，前一个维度-1表示自适应
palette = np.reshape(palette, (-1,3)).tolist()

pd = dict((i, color) for i, color in enumerate(palette))

json_str = json.dumps(pd)

with open("palette.json", "w") as f:
    f.write(json_str)

