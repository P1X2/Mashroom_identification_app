from django.contrib import admin

# Register your models here.
from .models import *

admin.site.register(Profile)
admin.site.register(Mushroom)
admin.site.register(Recipe)
admin.site.register(Course)
admin.site.register(CourseScore)
admin.site.register(Theory)
admin.site.register(TheoryItem)
admin.site.register(Test)
admin.site.register(Question)
admin.site.register(Answer)