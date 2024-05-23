from django.shortcuts import render
from .forms import ImageUploadForm
from PIL import Image
from .utils import *
from django.db.models import Q
from .models import *
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth import logout
import json
import math
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from .forms import ChangeUsernameForm, CustomPasswordChangeForm

def edit_username(request):
    if request.method == 'POST':
        form = ChangeUsernameForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            return redirect('profile')  # Przekierowanie na stronę profilu
    else:
        form = ChangeUsernameForm(instance=request.user)
    return render(request, 'base/edit_username.html', {'form': form})


def edit_password(request):
    if request.method == 'POST':
        form = CustomPasswordChangeForm(user=request.user, data=request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)  # Uaktualnienie sesji z nowym hasłem
            return redirect('profile')  # Przekierowanie na stronę profilu po pomyślnej zmianie hasła
    else:
        form = CustomPasswordChangeForm(user=request.user)
    return render(request, 'base/edit_password.html', {'form': form})


def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')  # gdzie 'home' to nazwa adresu URL do przekierowania po zalogowaniu
    else:
        form = AuthenticationForm()
    return render(request, 'base/login.html', {'form': form})

def user_register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            Profile.objects.create(user=user, level=1, points=0)
            login(request, user)
            return redirect('home')  # gdzie 'home' to nazwa adresu URL do przekierowania po zalogowaniu
    else:
        form = UserCreationForm()
    return render(request, 'base/register.html', {'form': form})


def user_logout(request):
    logout(request)
    return redirect('login')


# Create your views here.
def home(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = request.FILES['image']

            img = Image.open(uploaded_image)
            is_mushrooom = is_mushroom_classify_img_2(img)

            data_uri = pil_to_data_uri(img)

            if is_mushrooom:
                if request.user.is_authenticated:
                    # dodanie punktó użytkownikowi za znalezienie grzyba
                    profile = Profile.objects.get(user=request.user)
                    profile.points += math.ceil(5 / profile.level)
                    if profile.points >= 100:
                        profile.points = profile.points - 100
                        profile.level += 1
                    profile.save()

                request.session['uploaded_image'] = data_uri
                # classification with NN here
                result_id, pred_prob = 0, 88
                mushroom = Mushroom.objects.get(nn_id=result_id)
                predict_species(img)
                return render(request, 'base/classification.html', {'image': data_uri, 'mushroom': mushroom, 'probability': pred_prob})
            else:
                return render(request, 'base/no_mushroom.html', {})
    else:
        form = ImageUploadForm()

    context = {'form': form}
    return render(request, 'base/home.html', context)



def mushrooms(request):
    q = request.GET.get('q') if request.GET.get('q')!= None else ''
    mushrooms = Mushroom.objects.filter(
        Q(name__icontains=q) |
        Q(specname__icontains=q) |
        Q(description__icontains=q)
        )
    mush_count = mushrooms.count()

    context = {'mushrooms': mushrooms,
               'mush_count': mush_count
               }

    return render(request, 'base/mushrooms.html', context)


def abc(request):
    context = {}
    return render(request, 'base/abc.html', context)

def profile(request):
    profile = Profile.objects.get(user=request.user)
    context = {'profile': profile}
    return render(request, 'base/profile.html', context)


def recipes_categories(request):
    context = {}
    return render(request, 'base/recipes_categories.html', context)

def recipes(request):
    category = request.GET.get('text')
    recipes = Recipe.objects.filter(category=category)
    context = {"recipes": recipes}
    return render(request, 'base/recipes.html', context)


def recipe(request, pk):
    recipe = Recipe.objects.get(id=pk)
    context = {"recipe": recipe}
    return render(request, 'base/recipe.html', context)

def abc1(request):
    context = {}
    return render(request, 'base/abc/abc_1.html', context)

def abc2(request):
    context = {}
    return render(request, 'base/abc/abc_2.html', context)

def abc3(request):
    context = {}
    return render(request, 'base/abc/abc_3.html', context)

def mushroom(request, pk):
    mushroom = Mushroom.objects.get(id=pk)
    context = {"mushroom": mushroom}
    return render(request, 'base/mushroom.html', context)


def compare(request, pk):
    mushroom = Mushroom.objects.get(id=pk)
    data_uri = request.session.get('uploaded_image')
    return render(request, 'base/compare.html', {'user_image': data_uri, "mushroom": mushroom})

def courses(request):
    courses = Course.objects.all()

    if request.user.is_authenticated:
        scores = CourseScore.objects.filter(user=request.user)
        return render(request, 'base/courses.html', {'courses': courses, 'scores': scores})
    else:
        return render(request, 'base/courses.html', {'courses': courses})


def course_theory(request, pk):
    course = Course.objects.get(id=pk)
    theory = Theory.objects.get(course=course)
    theory_items = TheoryItem.objects.filter(theory=theory)
    return render(request, 'base/course_theory.html', {'theory_items': theory_items, 'course': course})

def course_test(request, pk):
    course = Course.objects.get(id=pk)
    test = Test.objects.get(course=course)
    questions = Question.objects.filter(test=test)

    if request.method == 'POST':
        selected_answers = request.POST.getlist('user_answers')

        questions = Question.objects.filter(test=test)
        correct_answers = []
        user_score = 0

        for question in questions:
            correct_answer = Answer.objects.get(question=question, correct=True)
            correct_answers.append(str(correct_answer.id))

        for selected_answer in selected_answers:
            if selected_answer in correct_answers:
                user_score += 1


        total_questions = len(questions)
        percentage_score = (user_score / total_questions) * 100

        try:
            course_score = CourseScore.objects.get(user=request.user, course=course)
            course_score.score = percentage_score
            course_score.save()
            if (course_score.passed == False) and (percentage_score >= course.threshold):
                profile = Profile.objects.get(user=request.user)
                profile.points += course.points
                if profile.points >= 100:
                    profile.points = profile.points - 100
                    profile.level += 1
                course_score.passed = True
                course_score.save()
                profile.save()

            return redirect('courses')

        except CourseScore.DoesNotExist:
            CourseScore.objects.create(user=request.user, course=course, score=percentage_score)
            return redirect('courses')

    return render(request, 'base/course_test.html', {'questions': questions})
