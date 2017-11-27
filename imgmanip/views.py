from django.shortcuts import get_object_or_404, render, redirect
from django.template import RequestContext
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.conf import settings

from models_other import Category, Coco50
from imgmanip.models import Image
from imgmanip.forms import ImageUploadForm

import base64
import coco
import cStringIO
import utils
import os
from os import listdir
from os.path import isfile, join, splitext

def index(request):
  """
  Renders a page where you can either upload an image that will get saved in our database or choose from one of the existing files to play around with.
  """
  # Handle image upload
  if request.method == 'POST':
    form = ImageUploadForm(request.POST, request.FILES)
    if form.is_valid():
      new_img = Image(img_file = request.FILES['img_file'])
      new_img.save()

      # Redirect to the image edit page after POST
      url = '/imgmanip/edit?image_id=%s' % new_img.id
      return HttpResponseRedirect(url)

  else:
    form = ImageUploadForm()

  # Load all images for the image index page
  images = Image.objects.all()

  # Load all segmented images for image index page
  segmented = Coco50.objects.all()
  segmented_imgs = [coco50_obj.image.url for coco50_obj in segmented]

  # Render page with the form and all images
  context = {'images': images, 'form': form, 'segmented_imgs': segmented_imgs}
  return render(request, 'imgmanip/index.html', context)

def edit(request):
  """
  Renders a editable view given an image_id. Users can use this view to manipulate the image they are viewing.
  """
  if 'image_id' not in request.GET:
    return HttpResponseRedirect('/imgmanip')
  image_id = request.GET['image_id']
  image = get_object_or_404(Image, pk=image_id)
  return render(request, 'imgmanip/edit.html', {'image': image, 'image_id': image_id})

def manipulate(request):
  """
  This method manipulates the image passed in from the request.image_id and performs manipulations on it.

  Returns:
    an HttpResponse containing the image data of the manipulated image.
  """
  if 'image_id' not in request.GET or 'manipulation' not in request.GET:
    return HttpResponseRedirect('/imgmanip/edit')
  image_id = request.GET['image_id']
  image = get_object_or_404(Image, pk=image_id)
  image_original = utils.load_image(image)
  manipulation = {
      'foveate_naive': utils.foveate_naive,
      'foveate': utils.foveate,
      'segment': utils.segment,
      }[request.GET['manipulation']]
  image_editted = manipulation(image_original, float(request.GET['x']), float(request.GET['y']))
  url = utils.save_image(image_editted)
  return HttpResponse(url, 200)

def obj_interact(request, image_id):
  """
  Loads image with segmented objects; allows interaction with this image (TODO)
  """
  context = {
    'image_id': image_id
  }
  return render(request, 'imgmanip/obj_interact.html', context)


def clusters(request):
  categories = coco.categories()
  return render(request, 'imgmanip/clusters.html', {'categories': categories})

def attribute_charts(request):
  return render(request, 'imgmanip/attribute_charts.html', {})
