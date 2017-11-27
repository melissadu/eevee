from django.core.management.base import BaseCommand
from coco.models import Image, Category

import json

class Command(BaseCommand):
    help='Given a json file containing all the images along with their widths, heights, urls, categories and split, it imports them into the database'

    def add_arguments(self, parser):
        parser.add_argument('--image-cat', type=str, default='../data/coco/gen/image_cat.json')
        parser.add_argument('--category_id2name', type=str, default='../data/coco/gen/category_id2name.json')

    def handle(self, *args, **options):
        category_id2name = json.load(open(options['category_id2name']))
        splits = json.load(open(options['image_cat']))
        for split in splits:
            images = splits[split]
            for image in images:
                img = Image.objects.create(
                    id=image['id'],
                    url=image['url'],
                    width=image['width'],
                    height=image['height'],
                    split=split)
                for cat in image['categories']:
                    (c, _) = Category.objects.get_or_create(name=category_id2name[cat])
                    c.images.add(img)
