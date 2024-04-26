from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('mushrooms', views.mushrooms, name="mushrooms"),
    path('abc', views.abc, name="abc"),
    path('login/', views.user_login, name='login'),
    path('register/', views.user_register, name='register'),
    path('logout/', views.user_logout, name='logout'),
    path('profile/', views.profile, name='profile'),
    path('recipes_categories/', views.recipes_categories, name='recipes_categories'),
    path('recipes/', views.recipes, name='recipes'),
    path('recipe/<str:pk>/', views.recipe, name='recipe'),
    # path('classification/', views.classify, name="classification")
    # path('mySets/', views.mySets, name="mySets"),room/<str:pk>/
]