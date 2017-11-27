from sklearn.cluster import KMeans
from PIL import Image

import argparse
import h5py
import json
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os


def cluster(features, n_clusters, seed=1234):
    model = KMeans(n_clusters=n_clusters, random_state=args.seed).fit(features)
    return model

def load_features(features_path):
    f = h5py.File(os.path.join(features_path, 'train_object_features.hdf5'), 'r')
    categories = f['categories'][()]
    features = f['features'][()]
    image_ids = f['image_ids'][()]
    boxes = f['boxes'][()]
    return boxes, image_ids, categories, features

def find_category_features(boxes, image_ids, features, categories, category):
    indices = categories == category
    features = features[indices.reshape(indices.shape[0]), :]
    image_ids = image_ids[indices.reshape(indices.shape[0])]
    boxes = boxes[indices.reshape(indices.shape[0]), :]
    return boxes, image_ids, features

def get_images(image_ids, image_folder, annotations_filename):
    annotations = json.load(open(annotations_filename))
    images = []
    for i in range(image_ids.shape[0]):
        images.append(annotations[str(image_ids[i, 0])])
    return images

def visualize(boxes, images, labels, category_name, save_folder, ncols=10, size=128):
    # Organize the clusters
    clusters = {}
    for i in range(labels.shape[0]):
        num = labels[i]
        if num not in clusters:
            clusters[num] = []
        clusters[num].append((boxes[i], images[i]))

    # Generate the grid
    fig = plt.figure()
    nrows = np.max(labels)
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.0)
    ax = [plt.subplot(gs[i]) for i in range(nrows*ncols)]
    gs.update(hspace=0, wspace=0)

    fig.suptitle(category_name)
    for row in range(np.max(labels)):
        for col, (box, image) in enumerate(clusters[row][:ncols]):
            i = row*ncols + col
            im = Image.open(image).convert('RGB')
            im = im.crop((box[0], box[1], box[0]+box[2], box[1]+box[3]))
            im = im.resize((size, size))
            ax[i].imshow(im)
            ax[i].axis('off')
    plt.savefig(os.path.join(save_folder, category_name + '.png'))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Extract all the features for images in coco")
    parser.add_argument("--features_path", type=str, default='/data/ranjaykrishna/coco/gen')
    parser.add_argument("--annotations_filename", type=str, default='/data/ranjaykrishna/coco/gen/image_id2filename.json')
    parser.add_argument("--image_folder", type=str, default='/data/ranjaykrishna/coco/train2014')
    parser.add_argument("--category-id2name", type=str, default='/data/ranjaykrishna/coco/gen/category_id2name.json')
    parser.add_argument("--save", type=str, default='/data/ranjaykrishna/coco/gen/object_clusters')
    parser.add_argument("--category", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--recluster", action='store_true')
    args = parser.parse_args()

    # First let's load the features we want
    boxes, image_ids, categories, features = load_features(args.features_path)

    # Grab the features to categorize
    boxes, image_ids, features = find_category_features(boxes, image_ids, features, categories, args.category)

    # Check to see if the clusters have already been calculated
    if not args.recluster and os.path.isfile(os.path.join(args.save, str(args.category) + '.npy')):
        print "Loading existing clusters"
        labels = np.load(os.path.join(args.save, str(args.category) + '.npy'))
    else:
        # Now let's cluster these guys
        print "Clusting the features"
        model = cluster(features, args.n_clusters, seed=args.seed)
        labels = model.labels_
        np.save(os.path.join(args.save, str(args.category) + '.npy'), labels)

    # Now let's visualize them
    print "Grabbing images"
    images = get_images(image_ids, args.image_folder, args.annotations_filename)
    print "Visualizing clusters"
    category_name = json.load(open(args.category_id2name))[str(args.category)]
    visualize(boxes, images, labels, category_name, args.save)
