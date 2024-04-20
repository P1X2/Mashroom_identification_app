from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Mushroom(models.Model):
    name = models.CharField(max_length=200)
    specname = models.CharField(max_length=200)
    description = models.TextField(null=True, blank=True)
    edible = models.BooleanField(default=False)
    img1 = models.ImageField(null=True, default="grzybki.jpeg")
    img2 = models.ImageField(null=True, default="grzybki.jpeg")
    img3 = models.ImageField(null=True, default="grzybki.jpeg")
