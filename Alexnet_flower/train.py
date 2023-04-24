import os 
import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.utils.data
from model import AlexNet
from tqdm import tqdm


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("using {} device.".format(device))

# to do: another try of normalization in the picture instead of tranforms.normalize
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))
                                 ]),

    "val":transforms.Compose([transforms.Resize((224,224)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))
                              ])
}

data_root = r"D:\deep-learning-for-image-processing\data_set"
# image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
# assert os.path.exists(image_path), "{} path does not exist.".format(image_path)




def train():
    # device = device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train_dataset = datasets.ImageFolder(root=os.path.join(data_root,"train"),
                                     transform=data_transform["train"])

    train_num = len(train_dataset)

    # to do---understand and use class_to_idx
    flower_list = train_dataset.class_to_idx
    print(flower_list)
    cla_dict = dict((val, key) for key, val in flower_list.items())

    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    print('Using {} dataloader workers every process'.format(num_workers))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size,
                                               num_workers)
    
    validatation_dataset = datasets.ImageFolder(root=os.path.join(data_root, "val"),
                                                transform=data_transform["val"])
    
    val_num = len(validatation_dataset)

    val_loader = torch.utils.data.DataLoader(validatation_dataset,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=num_workers)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))


    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = AlexNet(num_class=5, init_weights=True)

    net.to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 10

    save_path = "./AlexNet.pth"

    best_acc = 0

    train_steps = len(train_loader)

    for epoch in range(epochs):

        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        # train_bar = tqdm(train_loader, file=sys.stdout)

        # to do --- change this into enumerate dataloader
        for step, data in enumerate(train_bar):
            images, labels = data
            # print(images)
            # print(labels)
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()



            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
   
   
        # validate
        net.eval()
        acc = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                output = net(val_images.to(device))
                # 这里predict_y的意义？
                predict_y = torch.max(output, dim=1)[1]
                # print(predict_y)
                # acc 指标计算
                # torch.eq() -- 对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                    (epoch + 1, running_loss / train_steps, val_accurate))

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)

    print('Finished Training')
        

if __name__ == "__main__":
    train()
