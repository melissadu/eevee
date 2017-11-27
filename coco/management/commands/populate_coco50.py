from django.core.management.base import BaseCommand
from coco.models import Image, Coco50

import json

class Command(BaseCommand):
    help='Imports all coco50 images into database'

    def handle(self, *args, **options):
        coco50_ids = [33554,290942,111842,100318,127751,104631,175479,374727,456972,261785,45750,250210,58149,269862,43543,212866,579312,198079,259005,438099,13150,127540,161515,503887,213181,283548,304445,314067,8981,299271,243491,425672,66166,342929,435260,36433,571048,388403,306950,158754,514180,14135,101720,210195,407930,30065,279027,293799,53958,10275]

        for id in coco50_ids:
            Coco50.objects.create(image=Image.objects.get(id=id))
