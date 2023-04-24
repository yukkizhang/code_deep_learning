import os 
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np

from model import AlexNet
from PIL import Image

# to do --- deal with the gray (1 channel) image
# predict single image not appear in the train process but belongs to the 5 flower class
# to do -- using not  5 flower class image as input to see the predict result
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# # tranform for gray image
# data_transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5), (0.5))
# ])

# load test image
image_path = r"D:\2PM_2023\basic\classification\tulip3.png"

assert os.path.exists(image_path), "file: '{}' dose not exist.".format(image_path)

img = Image.open(image_path)

plt.imshow(img)
npimg = np.array(img)
print(npimg.shape)


img = data_transform(img)
img = torch.unsqueeze(img, dim=0)

# 这里class_indices.json文件的内容和意义，为什么采用json格式文件
json_path = r'D:\2PM_2023\basic\class_indices.json'
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

with  open(json_path, "r") as f:
    class_indict = json.load(f)

# load model
model = AlexNet(num_class=5).to(device)

# load model weights
# finish valadite process and then save the path
weight_path = r"D:\2PM_2023\basic\AlexNet.pth"
assert os.path.exists(weight_path), "file: '{}' dose not exist.".format(weight_path)
model.load_state_dict(torch.load(weight_path))

# test/validate mode instead of train mode
model.eval()

with torch.no_grad():
    output = torch.squeeze(model(img.to(device))).cpu()
    predict = torch.softmax(output, dim=0)
    # torch.argmax(input, dim=None, keepdim=False)返回指定维度最大值的序号
    predict_cla = torch.argmax(predict).numpy()

print_res = "class: {}  prob: {:.3}".format(class_indict[str(predict_cla)],
                                            predict[predict_cla].numpy())
# add title to plt.imshow(img),title conclude the class name and predicted probability
plt.title(print_res)
for i in range(len(predict)):
    print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                              predict[i].numpy()))
    
plt.show()