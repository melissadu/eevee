from django.conf.urls import url
from . import views

app_name = 'imgmanip'
urlpatterns = [
  # ex: /imgmanip/
  url(r'^$', views.index, name='index'),

  # ex: /imgmanip/edit?image_id=5
  url(r'^edit/$', views.edit, name='edit'),

  # ex: /imgmanip/manipulate?image_id=5$manipuate=foveate&x=0&y=0
  url(r'^manipulate/$', views.manipulate, name='edit'),

  # ex: /imgmanip/obj_interact?image_id=5
  url(r'^obj_interact/(?P<image_id>[0-9]+)$', views.obj_interact, name='obj_interact'),

  url(r'^clusters$', views.clusters, name='clusters'),

  url(r'^attribute_charts$', views.attribute_charts, name='clusters'),
]
