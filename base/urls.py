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
    path('edit_username/', views.edit_username, name='edit_username'),
    path('edit_password/', views.edit_password, name='edit_password'),
    path('recipes_categories/', views.recipes_categories, name='recipes_categories'),
    path('recipes/<str:pk>/', views.recipes, name='recipes'),
    path('recipe/<str:pk>/', views.recipe, name='recipe'),
    path('abc1', views.abc1, name='abc1'),
    path('abc2', views.abc2, name='abc2'),
    path('abc3', views.abc3, name='abc3'),
    path('mushroom/<str:pk>/', views.mushroom, name='mushroom'),
    path('compare/<str:pk>/', views.compare, name='compare'),
    path('courses/', views.courses, name='courses'),
    path('course_theory/<str:pk>/', views.course_theory, name='course_theory'),
    path('course_test/<str:pk>/', views.course_test, name='course_test'),
    # path('classification/', views.classify, name="classification")
    # path('mySets/', views.mySets, name="mySets"),room/<str:pk>/
]