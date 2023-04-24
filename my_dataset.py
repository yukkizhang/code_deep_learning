import os
import torch.utils.data as data
from PIL import Image

# from torchvision import transforms as T
import my_transform as T
# load the VOC dataset
# 继承自Dataset,包含__init__, __getitem__, __len__
class VOCSeg(data.Dataset):
    # train.txt是VOC数据集中各个文件的索引，用于寻找data and label
    def __init__(self, voc_root, year = "2012", transforms = None, txt_name: str = "train.txt"):
        super(VOCSeg, self).__init__()
        # using assert to identify the root exist or not
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        # get the semantic segmentation label and init image in the VOC
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')

        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), "path '{}' does not exist.".format(txt_path)

        # 通过索引的txt文件，读取获得文件名列表，之后加上后缀名即可读取相应图像及label
        with open(txt_path, "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms
    # 没有进__getitem__，__len__，collate_fn？？？
    def __getitem__(self, index):
        # using index to get the image and label
        img = Image.open(self.images[index]).convert('RGB')
        label = Image.open(self.masks[index])
        
        # 此处self.transforms直接传入两个参数图像！！
        if self.transforms is not None:
            img, label = self.transforms(img, label)


        return img, label
    
    def __len__(self):
        return len(self.images)
    
    @staticmethod
    def collate_fn(batch):
        # zip将对应位置元素打包，将img和label对应起来
        images, labels = list(zip(*batch))    
        batched_imgs = cat_list(images, fill_value=0)
        batched_labels = cat_list(labels, fill_value=255)

        return batched_imgs, batched_labels

# 进行图片的填充,并将多张图组合成一个batch
def cat_list(images, fill_value = 0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    # fill_将图片所有像素值都填充为fill_value
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy(img)

    return batched_imgs

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        # trans = [T.RandomResizedCrop((min_size, max_size))]
        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            # T.RandomResizedCrop(base_size),
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)

def get_transform(train):
    base_size = 520
    crop_size = 480

    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)

dataset = VOCSeg(voc_root="./", transforms=get_transform(train=True))
d1 = dataset[0]
print(d1)