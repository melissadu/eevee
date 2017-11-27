from coco.models import Image, Category, Coco50
from django.shortcuts import render
from django.contrib.staticfiles.templatetags.staticfiles import static
from django.shortcuts import get_object_or_404, render, redirect
from django.http import HttpResponse, HttpResponseRedirect
import urllib2
import json
import os
from django.conf import settings
from imgmanip.models import Image
from imgmanip.forms import ImageUploadForm
import numpy as np


attributes_file = open(os.path.join(settings.MEDIA_ROOT, 'attributes/coco_attributes_with_images.json'))
attributes_string = attributes_file.read()
attributes = json.loads(attributes_string)

flickr_urls_file = open(os.path.join(settings.MEDIA_ROOT, 'attributes/coco_flickr_urls.json'))
flickr_urls_string = flickr_urls_file.read()
flickr_urls = json.loads(flickr_urls_string)

anns_file = open(os.path.join(settings.MEDIA_ROOT, 'attributes/coco_anns_grouped.json'))
anns_string = anns_file.read()
anns = json.loads(anns_string)

def index(request):
  """
  Renders a page where you can choose to interact with a Coco segmented image
  """
  # Load all segmented images for image index page
  segmented = Coco50.objects.all()
  segmented_imgs = [coco50_obj.image.id for coco50_obj in segmented]

  # Render page with the form and all images
  context = {'segmented_imgs': segmented_imgs}
  return render(request, 'coco/index.html', context)

def category_index(request):
    """
    Renders all the categories that are available.
    """
    categories = Category.objects.values_list('name')
    categories = [c[0] for c in categories]
    return render(request, 'coco/category_index.html', {'categories': categories})

def image_index(request):
    """
    Renders a few images for a given category
    """
    print "REQUEST", request
    MAX_IMAGES = 24
    images = []
    if 'category' in request.GET:
        name = request.GET['category']
        category = Category.objects.get(name=request.GET['category'])
        images = category.images.all()[:MAX_IMAGES]
        print category
    else:
        print "here"
        # category = Category.objects.all()[0]
        category = Category.objects.all()
        images = category.images.all()[:MAX_IMAGES]
    # images = category.images.all()[:MAX_IMAGES]
    images = [im.url for im in images]
    return render(request, 'coco/image_index.html', {'urls': images})

def obj_interact(request, image_id):
    """
    Loads image with segmented objects; allows selection of an object of interest
    """
    # image = get_object_or_404(Image, pk=image_id)
    # print Image.objects.all() # images are all empty
    images = Image.objects.all()
    # image = images[0]
    print "SANITY"
    print images[0].id
    context = {
        'image_id': image_id,
    }
    return render(request, 'coco/obj_interact.html', context)

def obj_interact2(request, image_id, src_theme, dst_theme):
    """
    Loads image with segmented objects; allows selection of an object of interest
    """
    # image = get_object_or_404(Image, pk=image_id)
    # print Image.objects.all() # images are all empty
    images = Image.objects.all()
    print "IMAGES", images

    # Hardcoded for the elephant pic (id = 30065)
    # Assume that data is in form: <object_id>:<suggested_edit>
    edits = {
        "580012": "Replace Object 580012",
        "580416": "Replace Object 580416",
        "b": "Add an object to the scene"
    }

    # Hardcoded for the elephant pic (id = 30065)
    # Assume that data is in form: <object_id>:[catId1, catId2, catId3]
    # Reference category_id2name.json
    replacementObjs = {
        "580012": [19, 20, 25],
        "580416": [20, 21, 23]
    }

    test = [19, 20];

    # image = images[0]
    # print images[0].id
    context = {
        'dst_theme': dst_theme,
        'edits': edits,
        'image_id': image_id,
        'replacement_objs': replacementObjs,
        'src_theme': src_theme,
        'test': test
    }
    return render(request, 'coco/obj_interact2.html', context)

def theme_id(request, image_name):
    """
    Loads image with segmented objects; allows selection of an object of interest
    """
    # image = get_object_or_404(Image, pk=image_id)
    print Image.objects.all() # images are all empty
    # image = Image.objects.get(id=1)

    print("image name", image_name)

    images = Image.objects.all()
  
    # Render page with the form and all images
    context = {'image_name': image_name}
    return render(request, 'coco/theme_id.html', context)

    # context = {
    #     'image_id': image_id,
    #     # 'image_id': image_id,
    # }
    # return render(request, 'coco/theme_id.html', context)

def first_screen(request):
  """
  Renders a page where you can either upload an image that will get saved in our database or choose from one of the existing files to play around with.
  """
  # Handle image upload
  # Image.objects.all().delete() # bad method, but just put this line here to clear the images

  if request.method == 'POST':
    form = ImageUploadForm(request.POST, request.FILES)
    if form.is_valid():
      temp1 = str(request.FILES['img_file'])
      temp = temp1[:temp1.index(".")]
      new_img = Image(img_file = request.FILES['img_file'], img_name = temp)
      new_img.save()
      url = '/coco/first_screen'
      return HttpResponseRedirect(url)
  else:
    form = ImageUploadForm()

  # Load all images for the image index page
  images = Image.objects.all()

  # for image in images:
  #   image.refresh_from_db()
  # images.refresh_from_db()
  # Render page with the form and all images
  context = {'images': images, 'form': form}
  return render(request, 'coco/first_screen.html', context)

def cat_id2name(cat_id):
    cat_id2name_file = open(os.path.join(settings.MEDIA_ROOT, 'attributes/category_id2name.json'))
    cat_id2name_string = cat_id2name_file.read()
    cat_id2name = json.loads(cat_id2name_string)
    cat_name = cat_id2name[str(cat_id)]
    return cat_name

def obj_attributes(request, image_id, obj_id, cat_id):
    """
    Loads all attributes for the selected object
    """
    cat_name = cat_id2name(cat_id)
    context = {
        'image_id': image_id,
        'obj_id': obj_id,
        'cat_id': cat_id,
        'cat_name': cat_name,
    }
    return render(request, 'coco/obj_attributes.html', context)

def obj_replacements(request, image_id, obj_id, cat_id, attr_id):
    """
    Loads all possible replacement images for the selected object + attribute(s)
    """
    # TODO: given obj_id and attr_id, load list of relevant image ids
    # get the urls of these image ids from Image database
    # pass these into an argument
    # load these images in template html file
    # crop the images by their polygon coords
    cat_name = cat_id2name(cat_id)

    # Get replacement images for selected object + attribute(s)
    attr_name = attributes[cat_name][int(attr_id)]["attribute"]
    repl_images = attributes[cat_name][int(attr_id)]["images"] # Store image ids of valid replacements
    print "len(repl_images) 1", len(repl_images)
    repl_images = list(set(repl_images)) # Remove duplicates
    print "len(repl_images) 2", len(repl_images)
    repl_images = [repl_image for repl_image in repl_images if str(repl_image) in anns]
    print "len(repl_images) 3", len(repl_images)

    # Get replacement image annotation data for selected object + attribute(s)
    repl_anns = [anns[str(repl_images[i])] for i in range(len(repl_images))]
    repl_urls = {}
    repl_polys = {}
    repl_bboxes = {} # replacement bounding boxes
    for ann in repl_anns:
        for obj in ann:
            if obj["category_id"] == int(cat_id):
                seg = obj["segmentation"][0] # TEMP: just take first segmentation
                poly = np.array(seg).reshape((int(len(seg)/2), 2))
                poly = [list(poly_row) for poly_row in poly]
                # Get the flickr url for the replacement object image
                obj_image_id = obj["image_id"]
                if str(obj_image_id) in flickr_urls:
                    repl_urls[obj_image_id] = flickr_urls[str(obj_image_id)]
                # Get the polygons and bounding boxes for the replacement object image
                repl_polys[obj_image_id] = list(poly)
                repl_bboxes[obj_image_id] = obj["bbox"]
                break
    print "len(repl_urls)", len(repl_urls)
    repl_urls = json.dumps(repl_urls)
    print "len(repl_polys)", len(repl_polys)
    repl_polys = json.dumps(repl_polys)

    for obj in anns[str(image_id)]:
        if obj["id"] == int(obj_id):
            orig_bbox = obj["bbox"]

    context = {
        'image_id': image_id,
        'obj_id': obj_id,
        'cat_id': cat_id,
        'cat_name': cat_name,
        'attr_id': attr_id,
        'attr_name': attr_name,
        'repl_ids': repl_images,
        'repl_urls': repl_urls,
        'repl_polys': repl_polys,
        'repl_bboxes': repl_bboxes,
        'orig_bbox': orig_bbox,
    }
    return render(request, 'coco/obj_replacements.html', context)
