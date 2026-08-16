"""
URL routing.

Django serves the API only; the React SPA (Vite) is the sole frontend.

  /api/...        legacy JSON endpoints, retained for the telemetry sender
  /api/v1/...     the current API consumed by the SPA
"""
from django.urls import path
from django.views.generic import RedirectView

from . import api_platform, api_views, views

urlpatterns = [
    path("", RedirectView.as_view(url="http://localhost:5173/", permanent=False), name="home"),
    path("logout/", views.logout_view, name="logout"),

    # -----------------------------------------------------------------------
    # Legacy JSON API (telemetry sender + older clients)
    # -----------------------------------------------------------------------
    path("api/csrf/", views.api_csrf, name="api_csrf"),
    path("api/readings/", views.api_latest_readings, name="api_latest_readings"),
    path("api/readings/submit/", views.api_submit_reading, name="api_submit_reading"),
    path("api/predict-cost/", views.api_predict_cost, name="api_predict_cost"),
    path("api/compute-route/", views.api_compute_route, name="api_compute_route"),
    path("api/notifications/", views.api_notifications, name="api_notifications"),
    path("api/train-model/", views.api_train_model, name="api_train_model"),
    path("api/model-info/", views.api_model_info, name="api_model_info"),
    path("api/update-location/", views.api_update_location, name="api_update_location"),
    path("api/user-location/", views.api_get_user_location, name="api_get_user_location"),

    # -----------------------------------------------------------------------
    # v1 – identity and account
    # -----------------------------------------------------------------------
    path("api/v1/me/", api_views.MeAPIView.as_view(), name="api_v1_me"),
    path("api/v1/auth/login/", api_views.LoginAPIView.as_view(), name="api_v1_login"),
    path("api/v1/auth/signup/", api_views.SignupAPIView.as_view(), name="api_v1_signup"),
    path("api/v1/auth/logout/", api_views.LogoutAPIView.as_view(), name="api_v1_logout"),
    path("api/v1/profile/", api_views.ProfileAPIView.as_view(), name="api_v1_profile"),
    path("api/v1/settings/", api_views.SettingsAPIView.as_view(), name="api_v1_settings"),

    # -----------------------------------------------------------------------
    # v1 – bins and telemetry
    # -----------------------------------------------------------------------
    path("api/v1/nodes/", api_views.NodeListAPIView.as_view(), name="api_v1_nodes"),
    path("api/v1/nodes/ensure/", api_views.EnsureNodeAPIView.as_view(),
         name="api_v1_nodes_ensure"),
    path("api/v1/nodes/<int:pk>/history/", api_views.NodeHistoryAPIView.as_view(),
         name="api_v1_node_history"),
    path("api/v1/readings/submit/", api_views.ReadingSubmitAPIView.as_view(),
         name="api_v1_reading_submit"),
    path("api/v1/dashboard/", api_views.DashboardAPIView.as_view(), name="api_v1_dashboard"),

    # -----------------------------------------------------------------------
    # v1 – notifications
    # -----------------------------------------------------------------------
    path("api/v1/notifications/", api_views.NotificationListAPIView.as_view(),
         name="api_v1_notifications"),
    path("api/v1/notifications/mark-all-read/",
         api_views.NotificationMarkAllReadAPIView.as_view(),
         name="api_v1_notifications_mark_all"),
    path("api/v1/notifications/<int:pk>/read/",
         api_views.NotificationMarkReadAPIView.as_view(), name="api_v1_notification_read"),

    # -----------------------------------------------------------------------
    # v1 – fleet dispatch
    # -----------------------------------------------------------------------
    path("api/v1/fleet/config/", api_platform.FleetConfigAPIView.as_view(),
         name="api_v1_fleet_config"),
    path("api/v1/fleet/vehicles/<int:pk>/", api_platform.VehicleDetailAPIView.as_view(),
         name="api_v1_fleet_vehicle"),
    path("api/v1/fleet/plan/", api_platform.FleetPlanAPIView.as_view(),
         name="api_v1_fleet_plan"),
    path("api/v1/fleet/plans/", api_platform.PlanListAPIView.as_view(),
         name="api_v1_fleet_plans"),
    path("api/v1/fleet/compare/", api_platform.FleetCompareAPIView.as_view(),
         name="api_v1_fleet_compare"),
    path("api/v1/fleet/service/", api_platform.ServiceEventAPIView.as_view(),
         name="api_v1_fleet_service"),
    path("api/v1/fleet/equity/", api_platform.EquityAPIView.as_view(),
         name="api_v1_fleet_equity"),

    # -----------------------------------------------------------------------
    # v1 – sensing integrity
    # -----------------------------------------------------------------------
    path("api/v1/sensors/health/", api_platform.SensorHealthAPIView.as_view(),
         name="api_v1_sensor_health"),
    path("api/v1/sensors/faults/", api_platform.FaultTaxonomyAPIView.as_view(),
         name="api_v1_fault_taxonomy"),
    path("api/v1/sensors/inject/", api_platform.FaultInjectionAPIView.as_view(),
         name="api_v1_fault_inject"),

    # -----------------------------------------------------------------------
    # v1 – models and explainability
    # -----------------------------------------------------------------------
    path("api/v1/models/status/", api_platform.ModelStatusAPIView.as_view(),
         name="api_v1_model_status"),
    path("api/v1/models/continual/reset/", api_platform.ContinualResetAPIView.as_view(),
         name="api_v1_continual_reset"),
    path("api/v1/explain/<int:pk>/", api_platform.ExplainAPIView.as_view(),
         name="api_v1_explain"),

    # -----------------------------------------------------------------------
    # v1 – sustainability and context
    # -----------------------------------------------------------------------
    path("api/v1/emissions/", api_platform.EmissionsAPIView.as_view(),
         name="api_v1_emissions"),
    path("api/v1/traffic/", api_platform.TrafficAPIView.as_view(), name="api_v1_traffic"),

    # -----------------------------------------------------------------------
    # v1 – audit
    # -----------------------------------------------------------------------
    path("api/v1/audit/", api_platform.AuditLedgerAPIView.as_view(), name="api_v1_audit"),
    path("api/v1/audit/verify/", api_platform.AuditVerifyAPIView.as_view(),
         name="api_v1_audit_verify"),
]
