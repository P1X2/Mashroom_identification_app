from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm

class ImageUploadForm(forms.Form):
    image = forms.ImageField()