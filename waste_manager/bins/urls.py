from django.urls import path
from django.views.generic import RedirectView
from . import views
from . import api_views

# NOTE: All template-based UI routes (login/, signup/, dashboard/, profile/, settings/)
# have been REMOVED. The React SPA (Vite) is the only frontend now.
# Django only serves /api/* endpoints.

urlpatterns = [
    # Root → redirect to React (Vite dev) or let Nginx handle in prod
    path('', RedirectView.as_view(url='http://localhost:5173/', permanent=False), name='home'),

    # logout still needs Django session handling (POST from React)
    path('logout/', views.logout_view, name='logout'),

    # -----------------------------------------------------------------------
    # Legacy JSON API endpoints (kept — consumed by dummy data sender + React)
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # DRF v1 API – consumed by the React frontend
    # -----------------------------------------------------------------------
    path('api/v1/me/', api_views.MeAPIView.as_view(), name='api_v1_me'),
    path('api/v1/auth/login/', api_views.LoginAPIView.as_view(), name='api_v1_login'),
    path('api/v1/auth/signup/', api_views.SignupAPIView.as_view(), name='api_v1_signup'),
    path('api/v1/auth/logout/', api_views.LogoutAPIView.as_view(), name='api_v1_logout'),
    # Nodes
    path('api/v1/nodes/', api_views.NodeListAPIView.as_view(), name='api_v1_nodes'),
    path('api/v1/nodes/ensure/', api_views.EnsureNodeAPIView.as_view(), name='api_v1_nodes_ensure'),
    # Readings
    path('api/v1/readings/submit/', api_views.ReadingSubmitAPIView.as_view(), name='api_v1_reading_submit'),
    # App data
    path('api/v1/dashboard/', api_views.DashboardAPIView.as_view(), name='api_v1_dashboard'),
    path('api/v1/profile/', api_views.ProfileAPIView.as_view(), name='api_v1_profile'),
    path('api/v1/settings/', api_views.SettingsAPIView.as_view(), name='api_v1_settings'),
    path('api/v1/notifications/', api_views.NotificationListAPIView.as_view(), name='api_v1_notifications'),
    path('api/v1/notifications/mark-all-read/', api_views.NotificationMarkAllReadAPIView.as_view(), name='api_v1_notifications_mark_all'),
    path('api/v1/notifications/<int:pk>/read/', api_views.NotificationMarkReadAPIView.as_view(), name='api_v1_notification_read'),
]