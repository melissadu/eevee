from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import models

import argparse
import h5py
import json
import os
import progressbar
import torch
import torch.nn as nn
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description="Extract all the features for images in coco")
parser.add_argument("--images", type=str, default='/data/ranjaykrishna/coco')
parser.add_argument("--data", type=str, default='/data/ranjaykrishna/coco/annotations')
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--output", type=str, default='/data/ranjaykrishna/coco/gen')
parser.add_argument("--model", type=str, default='resnet50')
parser.add_argument("--image-size", type=int, default=224)
parser.add_argument("--feature-size", type=int, default=2048)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()


class COCO(Dataset):

    def __init__(self, image_folder, annotations_file):
        self.image_folder = image_folder
        self.image_names = []
        self.categories = []
        self.image_ids = []
        self.boxes = []
        self.transform = transforms.Compose([
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                 std = [ 0.229, 0.224, 0.225 ]),
        ])

        # Now let's parse all the annotations
        annotations = json.load(open(annotations_file))
        id2name = {}
        for image in annotations['images']:
            id2name[image['id']] = image['file_name']
        for annotation in annotations['annotations']:
            self.image_ids.append(annotation['image_id'])
            self.image_names.append(id2name[annotation['image_id']])
            self.boxes.append(annotation['bbox'])
            self.categories.append(annotation['category_id'])

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.image_folder, self.image_names[index])).convert('RGB')
        box = [int(b) for b in self.boxes[index]]
        crop = image.crop((box[0], box[1], box[0]+box[2], box[1]+box[3]))
        crop = self.transform(crop)
        return torch.IntTensor(box).view(4), torch.IntTensor([self.image_ids[index]]).view(1), torch.IntTensor([self.categories[index]]).view(1), crop

    def __len__(self):
        return len(self.categories)


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
model = Features(original_model)
if args.cuda:
    model.cuda()

# Let's iterate over all the images
for split in ['val', 'train']:
    print "Extracting features in %s set" % split
    dataset = COCO(os.path.join(args.images, split + '2014'), os.path.join(args.data, 'instances_' + split + '2014.json'))
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size)
    total = len(dataset)
    with h5py.File(os.path.join(args.output, split + '_object_features.hdf5'), 'w') as of:
        # Create the 3 datasets
        dids = of.create_dataset('image_ids', (total, 1), dtype='int64')
        dboxes = of.create_dataset('boxes', (total, 4), dtype='int64')
        dcategories = of.create_dataset('categories', (total, 1), dtype='int64')
        dfeatures = of.create_dataset('features', (total, args.feature_size), dtype='f')
        bar = progressbar.ProgressBar(maxval=len(dataloader)).start()

        # iterate over the batches and extract features
        for progress, (boxes, image_ids, categories, images) in enumerate(dataloader):
            N = boxes.size(0)
            begin = progress*args.batch_size
            finish = begin + N

            # Forward the model and get features
            if args.cuda:
                images = images.cuda()
            images = Variable(images, volatile=True)
            features = model(images)

            # Store the values in hdf5
            dfeatures[begin:finish, :] = features.data.type(torch.FloatTensor).numpy()
            dids[begin:finish] = image_ids.numpy()
            dboxes[begin:finish, :] = boxes.numpy()
            dcategories[begin:finish] = categories.numpy()
            bar.update(progress)
        bar.finish()
