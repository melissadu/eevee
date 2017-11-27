from django.conf.urls import url
from coco import views

app_name = 'coco'
urlpatterns = [
  # ex: /coco/categories
  url(r'^categories$', views.category_index, name='category_index'),

  # ex: /coco/images?category=airplane
  url(r'^images/$', views.image_index, name='image_index'),

  # ex: /coco/
  url(r'^$', views.index, name='index'),

  # ex: /coco/first_screen
  url(r'^first_screen$', views.first_screen, name='first_screen'),

  # ex: /coco/obj_interact/5 (image_id=5)
  url(r'^obj_interact/(?P<image_id>[0-9]+)$', views.obj_interact, name='obj_interact'),

  # ex: /coco/obj_interact/5 (image_id=5)
  url(r'^obj_interact2/(?P<image_id>[0-9]+)/(?P<src_theme>[a-zA-Z0-9_.-]+)/(?P<dst_theme>[a-zA-Z0-9_.-]+)$', views.obj_interact2, name='obj_interact2'),

  # ex: 
  # what i need is a direct url to the filename, and i want that to be in the view.
  url(r'^theme_id/(?P<image_name>[a-zA-Z0-9_.-]+)$', views.theme_id, name='theme_id'),

  # ex: /coco/obj_attributes/5/1/0 (image_id=5, obj_id=1, cat_id=0)
  url(r'^obj_attributes/(?P<image_id>[0-9]+)/(?P<obj_id>[0-9]+)/(?P<cat_id>[0-9]+)$', views.obj_attributes, name='obj_attributes'),

  # ex: /coco/obj_replacements/5/1/0/2 (image_id=5, obj_id=1, cat_id=0, attr_id=2)
  url(r'^obj_replacements/(?P<image_id>[0-9]+)/(?P<obj_id>[0-9]+)/(?P<cat_id>[0-9]+)/(?P<attr_id>[0-9]+)$', views.obj_replacements, name='obj_replacements'),

]
