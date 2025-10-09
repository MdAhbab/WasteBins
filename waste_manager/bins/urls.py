from django.urls import path
from django.views.generic import RedirectView
from . import views

urlpatterns = [
    # Root redirect to login
    path('', RedirectView.as_view(url='/login/', permanent=False), name='home'),

    # Authentication
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # Main pages
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('profile/', views.profile_view, name='profile'),
    path('settings/', views.settings_view, name='settings'),

    # API endpoints
    path('api/csrf/', views.api_csrf, name='api_csrf'),
    path('api/readings/', views.api_latest_readings, name='api_latest_readings'),
    path('api/readings/submit/', views.api_submit_reading, name='api_submit_reading'),
    path('api/predict-cost/', views.api_predict_cost, name='api_predict_cost'),
    path('api/compute-route/', views.api_compute_route, name='api_compute_route'),
    path('api/notifications/', views.api_notifications, name='api_notifications'),
    path('api/train-model/', views.api_train_model, name='api_train_model'),
    path('api/model-info/', views.api_model_info, name='api_model_info'),
    path('api/update-location/', views.api_update_location, name='api_update_location'),
    path('api/user-location/', views.api_get_user_location, name='api_get_user_location'),
]