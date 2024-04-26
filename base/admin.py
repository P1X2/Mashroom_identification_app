from django.contrib import admin

# Register your models here.
from .models import Mushroom, Recipe

admin.site.register(Mushroom)
admin.site.register(Recipe)