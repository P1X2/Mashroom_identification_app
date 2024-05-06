from django.db import models
from django.contrib.auth.models import User
from django.dispatch import receiver
from django.db.models.signals import post_save


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    level = models.IntegerField(default=1)
    points = models.IntegerField(default=0)


# Create your models here.
class Mushroom(models.Model):
    name = models.CharField(max_length=200)
    specname = models.CharField(max_length=200)
    characteristics = models.JSONField(null=True)
    description = models.TextField(null=True, blank=True)
    edible = models.BooleanField(default=False)
    img1 = models.ImageField(null=True, default="grzybki.jpeg")
    img2 = models.ImageField(null=True, default="grzybki.jpeg")
    img3 = models.ImageField(null=True, default="grzybki.jpeg")


class Recipe(models.Model):
    name = models.CharField(max_length=200)
    img1 = models.ImageField(null=True, default="przepis.jpg")
    img2 = models.ImageField(null=True, default="przepis.jpg")
    category = models.TextField(null=True, blank=False)
    preptime = models.TextField(null=True, blank=False)
    ingredients = models.JSONField()
    instructions = models.JSONField()


class Course(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    points = models.IntegerField(default=20)
    threshold = models.IntegerField(default=80)

class CourseScore(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    score = models.FloatField(default=0)
    passed = models.BooleanField(default=False)

class Theory(models.Model):
    course = models.OneToOneField(Course, on_delete=models.CASCADE)

class TheoryItem(models.Model):
    theory = models.ForeignKey(Theory, on_delete=models.CASCADE)
    image = models.ImageField(null=True, blank=True)
    content = models.TextField(null=True)

class Test(models.Model):
    course = models.OneToOneField(Course, on_delete=models.CASCADE)

class Question(models.Model):
    test = models.ForeignKey(Test, on_delete=models.CASCADE)
    image = models.ImageField(null=True, blank=True)
    text = models.TextField(null=True)

class Answer(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    text = models.TextField(null=True)
    correct = models.BooleanField(default=False)




