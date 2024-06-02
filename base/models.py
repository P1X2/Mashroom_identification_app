from django.db import models
from django.contrib.auth.models import User
from django.dispatch import receiver
from django.db.models.signals import post_save


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    level = models.IntegerField(default=1)
    points = models.IntegerField(default=0)

    def __str__(self):
        return f'{self.user}'


# Create your models here.
class Mushroom(models.Model):
    name = models.CharField(max_length=200)
    specname = models.CharField(max_length=200, default='nazwa lacinska')
    characteristics = models.JSONField(null=True, default=dict(cecha='niebieski'))
    description = models.TextField(null=True, blank=True, default='opis grzyba')
    edible = models.BooleanField(default=False)
    nn_id = models.IntegerField(default=-1)
    img1 = models.ImageField(null=True, default="grzybki.jpeg")
    img2 = models.ImageField(null=True, default="grzybki.jpeg")
    img3 = models.ImageField(null=True, default="grzybki.jpeg")

    def __str__(self):
        return f'{self.name} ---- {self.nn_id}'


class RecipeCategory(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return f'{self.name}'


class Recipe(models.Model):
    category = models.ForeignKey(RecipeCategory, on_delete=models.CASCADE, blank=True, null=True)
    name = models.CharField(max_length=200)
    img1 = models.ImageField(null=True, default="przepis.jpg")
    img2 = models.ImageField(null=True, default="przepis.jpg")
    preptime = models.TextField(null=True, blank=False)
    ingredients = models.JSONField()
    instructions = models.JSONField()

    def __str__(self):
        return f'{self.name}'


class Course(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    points = models.IntegerField(default=20)
    threshold = models.IntegerField(default=80)

    def __str__(self):
        return f'{self.name}'

class CourseScore(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    score = models.FloatField(default=0)
    passed = models.BooleanField(default=False)

class Theory(models.Model):
    course = models.OneToOneField(Course, on_delete=models.CASCADE)

    def __str__(self):
        return f'{self.course}'

class TheoryItem(models.Model):
    theory = models.ForeignKey(Theory, on_delete=models.CASCADE)
    image = models.ImageField(null=True, blank=True)
    content = models.TextField(null=True)

    def __str__(self):
        return f'{self.theory}'

class Test(models.Model):
    course = models.OneToOneField(Course, on_delete=models.CASCADE)

    def __str__(self):
        return f'{self.course}'

class Question(models.Model):
    test = models.ForeignKey(Test, on_delete=models.CASCADE)
    image = models.ImageField(null=True, blank=True)
    text = models.TextField(null=True)

    def __str__(self):
        return f'{self.test} --- {self.text}'

class Answer(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    text = models.TextField(null=True)
    correct = models.BooleanField(default=False)

    def __str__(self):
        return f'{self.question} --- {self.text} -- {self.correct}'




