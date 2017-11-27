from __future__ import unicode_literals

from django.db import models

# Create your models here.
class Image(models.Model):
  img_file = models.FileField(upload_to = 'img_uploads/')
  img_name = models.CharField(max_length=20,default='none')
