from django.shortcuts import render
from .forms import ImageUploadForm
from PIL import Image
from .utils import *
from django.db.models import Q
from .models import Mushroom, Recipe
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth import logout

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
            # Open the uploaded image with PIL
            img = Image.open(uploaded_image)
            is_mushrooom_str = classify_img(img)
            img = img.resize((100, 100))
            # processed_image = process_image(img)
            data_uri = pil_to_data_uri(img)
            # processed_image = process_image(img)
            # Now you can use processed_image in your neural network
            # Example: neural_network.predict(processed_image)
            return render(request, 'base/classification.html', {'image': data_uri, 'prediction': is_mushrooom_str})
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

def  profile(request):
    context = {}
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