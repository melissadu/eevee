from django import forms

class ImageUploadForm(forms.Form):
  img_file = forms.FileField(
    label='Select a file',
  )