import argparse
import json
import h5py

parser = argparse.ArgumentParser(description="Extract all the features for images in coco")
parser.add_argument("--features", type=str, default='/data/ranjaykrishna/coco/gen/features.hdf5')
parser.add_argument("--annotations", type=str, default='/data/ranjaykrishna/coco/gen/annotations.json')
args = parser.parse_args()

annotations = json.load(open(args.annotations))
with h5py.File(args.features, 'r') as f:
    for name in annotations:
        features = f[name][()]
        assert(features.shape[1] == 2048)
