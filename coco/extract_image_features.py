from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import models

import argparse
import os
import h5py
import progressbar
import torch.nn as nn
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description="Extract all the features for images in coco")
parser.add_argument("--images", type=str, default='/data/ranjaykrishna/coco')
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--output", type=str, default='/data/ranjaykrishna/coco/gen/features.hdf5')
parser.add_argument("--model", type=str, default='resnet50')
parser.add_argument("--image-size", type=int, default=224)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

class COCO(Dataset):

    def __init__(self, folder):
        images = []
        filenames = []
        self.transform = transforms.Compose([
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                 std = [ 0.229, 0.224, 0.225 ]),
        ])
        for image in os.listdir(folder):
            if not ('.jpg' in image):
                continue
            images.append(os.path.join(folder, image))
            filenames.append(image.replace('.jpg', ''))
        self.images = images
        self.filenames = filenames

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        image = self.transform(image)
        return self.filenames[index], image

    def __len__(self):
        return len(self.images)


class Features(nn.Module):
    def __init__(self, original_model):
        super(Features, self).__init__()
        self.model = original_model
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

# Let's get the model
original_model = getattr(models, args.model)(pretrained=True)
if args.cuda:
    original_model.cuda()
model = Features(original_model)
if args.cuda:
    model.cuda()
# Let's iterate over all the images
with h5py.File(args.output, 'w') as of:
    for split in ['val', 'train']:
        print "Extracting features in %s set" % split
        dataset = COCO(os.path.join(args.images, split + '2014'))
        dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size)
        bar = progressbar.ProgressBar(maxval=len(dataloader)).start()
        for progress, (filenames, images) in enumerate(dataloader):
            images = Variable(images)
            if args.cuda:
                images.cuda()
            features = model(images)
            bar.update(progress)
            for i, filename in enumerate(filenames):
                dset = of.create_dataset(filename, (1, features.size(1)), dtype='f')
                dset[0, :] = features.data[i, :].numpy()
        bar.finish()
