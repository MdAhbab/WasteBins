from django.urls import path
from . import views

urlpatterns = [
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('profile/', views.profile_view, name='profile'),
    path('settings/', views.settings_view, name='settings'),

    path('api/readings/', views.api_latest_readings, name='api_latest_readings'),
    path('api/readings/submit/', views.api_submit_reading, name='api_submit_reading'),
    path('api/predict-cost/', views.api_predict_cost, name='api_predict_cost'),
    path('api/compute-route/', views.api_compute_route, name='api_compute_route'),
    path('api/notifications/', views.api_notifications, name='api_notifications'),
    path('api/train-model/', views.api_train_model, name='api_train_model'),
    path('api/model-info/', views.api_model_info, name='api_model_info'),
]