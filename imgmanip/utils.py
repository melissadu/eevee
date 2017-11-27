from PIL import Image, ImageFilter
from django.conf import settings
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries

import math
import numpy as np
import os
import json

def segment(image, x, y):
  """
  Arguments:
    image is a (W, H, C) numpy array

  Returns:
    out is a (W, H, 1) numpy array with different values for each object mask. Ex, obj 1 might have value 1 in the (W, H) locations where it exists. obj2 might have value 2, etc.
  """
  out = np.asarray(image)
  segments_fz = felzenszwalb(out, scale=100, sigma=0.5, min_size=50)
  out = mark_boundaries(out, segments_fz)

  # Converting back to PIL image
  out = np.uint8(out*255)
  out = Image.fromarray(out)
  return out

def foveate_naive(image, x, y, size=0.15):
  """
  Foveates the image such that the square region centered around (x, y) is left in focus. The size of the square is size*image_width, size*image_height.

  Arguments:
    image is a (W, H, C) PIL image array
    x is the col where the foveation should take place normalized by the width of the image
    y is the row where the foveation should take place normalized by the height of the image
    size is the percentage of the total image that should be left in focus

  Returns:
    out is a (W, H, C) numpy array after foveation
  """
  # Adding gaussian blur
  out = image.filter(ImageFilter.GaussianBlur(radius=3))
  out = np.asarray(out)
  out.setflags(write=1)

  # Converting image to numpy and adding foveation
  image = np.asarray(image)

  # un-normalizing x and y
  x = x*image.shape[1]
  y = y*image.shape[0]
  offsetX = math.floor(image.shape[1]*size/2)
  offsetY = math.floor(image.shape[0]*size/2)
  xmin, xmax = int(max(0, x-offsetX)), int(min(x+offsetX, image.shape[1]))
  ymin, ymax = int(max(0, y-offsetY)), int(min(y+offsetY, image.shape[0]))
  out[ymin:ymax, xmin:xmax, :] = image[ymin:ymax, xmin:xmax, :]

  # Converting back to PIL image
  out = Image.fromarray(out)
  return out


def foveate(image, x, y):
  """
  Foveates the image such that the object that the (x, y) location points to is in focus.

  Arguments:
    image is a (W, H, C) numpy image array
    x is the col where the foveation should take place
    y is the row where the foveation should take place

  Returns:
    out is a (W, H, C) numpy array after foveation
  """
  orig = np.asarray(image)
  segments = felzenszwalb(orig, scale=100, sigma=0.5, min_size=50)
  label = segments[y, x]
  mask = segments[segments == label]

  # Adding gaussian blur
  out = image.filter(ImageFilter.GaussianBlur(radius=3))
  out = np.asarray(out)
  out.setflags(write=1)
  for c in range(3):
    out[:, :, c][mask] = orig[:, :, c][mask]

  # Converting back to PIL image
  out = Image.fromarray(np.uint8(segments))
  return out

def load_image(image):
  """
  Arguments:
    image is the imgmanip.models Image object.

  Returns:
    out is a numpy array containing the image data
  """
  out = Image.open(image.img_file.url[1:])
  return out

def save_image(image, name = "temp.png"):
  """
  Arguments:
    image is the imgmanip.models Image object.

  Returns:
    out is the url of where the image is saved.
  """
  if type(image).__module__ == np.__name__:
    image = Image.fromarray(np.uint8(image))
  path = os.path.join(settings.MEDIA_ROOT, name)
  image.save(path)
  return settings.MEDIA_URL + name

def load_segmented_image(image_id):
  """
  Arguments:
  ID of the desired Coco image

  Returns:
  Coco image with object segments overlaid
  """
  # TODO

def select_obj(image, x, y):
  """
  Provides menu of options for a segmented object that the user has selected

  Arguments:
    image is a (W, H, C) numpy image array
    x is the col of selected object
    y is the row of selected object

  Returns:
    A
  """
  # TODO













