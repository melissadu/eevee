import argparse
import json
import os

parser = argparse.ArgumentParser(description="Parse the COCO annotations to the version of the annotations we use in actiongenome: {FILENAME: {image_id: 1, split: 'train/val', sentences: ['sadfsdf']}}")
parser.add_argument("-annotations_folder", type=str, default='/data/ranjaykrishna/coco/annotations')
parser.add_argument("-output_file", type=str, default='/data/ranjaykrishna/coco/gen/annotations.json')
args = parser.parse_args()

id2name = {}
annotations = {}

for split in ['train', 'val']:
    filename = os.path.join(args.annotations_folder, 'captions_' + split + '2014.json')
    with open(filename) as f:
        data = json.load(f)
    for image in data['images']:
        id2name[image['id']] = image['file_name'].replace('.jpg', '')
    for image in data['annotations']:
        annotations[id2name[image['image_id']]] = {
            'image_id': image['image_id'],
            'split': split,
            'sentences': [image['caption']]
        }

with open(args.output_file, 'w') as f:
    f.write(json.dumps(annotations))
