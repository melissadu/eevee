from __future__ import unicode_literals

from django.db import models

class Image1(models.Model):
    id = models.IntegerField(primary_key=True)
    url = models.URLField(unique=True)
    width = models.PositiveIntegerField()
    height = models.PositiveIntegerField()
    split = models.CharField(max_length=10)

class Category(models.Model):
    name = models.CharField(max_length=20, primary_key=True)
    images = models.ManyToManyField(Image1)

class Coco50(models.Model):
  image = models.ForeignKey(Image1)
