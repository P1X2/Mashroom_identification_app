from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from .models import Profile
from django.contrib.auth.models import User

class ImageUploadForm(forms.Form):
    image = forms.ImageField()
