"""
DRF API views for the WasteBins React frontend (v1 API).
All views use SessionAuthentication so the existing Django login/logout
forms grant access — no extra token setup needed during development.
"""
import json
from datetime import datetime

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError as DjangoValidationError
from django.db import transaction
from rest_framework import generics, status
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .drf_serializers import (
    UserSerializer,
    UserSettingSerializer,
    SensorReadingSerializer,
    NotificationSerializer,
    CollectionRouteSerializer,
    NodeSerializer,
)
from .models import (
    Node, SensorReading, Notification, UserSetting,
    CollectionRoute,
)
from .utils.ai.model_store import get_model_version
from .utils.priority_calculator import priority_calculator


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
_AUTH = [SessionAuthentication, BasicAuthentication]
_PERM = [IsAuthenticated]


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
class DashboardAPIView(APIView):
    authentication_classes = _AUTH
    permission_classes = _PERM

    def get(self, request):
        # Latest reading per node
        latest_per_node = {}
        for r in SensorReading.objects.select_related('node').order_by('node_id', '-timestamp'):
            if r.node_id not in latest_per_node:
                latest_per_node[r.node_id] = r

        readings = list(latest_per_node.values())
        readings_data = SensorReadingSerializer(readings, many=True).data

        # Stats
        total_bins = Node.objects.count()
        critical_bins = sum(1 for r in readings if (r.waste_level or 0) >= 0.85)
        warning_bins = sum(1 for r in readings if 0.65 <= (r.waste_level or 0) < 0.85)
        avg_fill = (
            round(sum(float(r.waste_level or 0) for r in readings) / len(readings) * 100, 1)
            if readings else 0.0
        )

        # Latest route
        latest_route_obj = CollectionRoute.objects.order_by('-timestamp').first()
        latest_route = CollectionRouteSerializer(latest_route_obj).data if latest_route_obj else None

        # Unread notification count
        notif_count = Notification.objects.filter(
            user__in=[None, request.user], is_read=False
        ).count()

        # Model version
        model_version = get_model_version()

        # Priority info
        user_settings, _ = UserSetting.objects.get_or_create(user=request.user)
        priority_info = None
        user_lat = request.GET.get('lat')
        user_lng = request.GET.get('lng')
        if not user_lat or not user_lng:
            if user_settings.has_location():
                user_lat = user_settings.latitude
                user_lng = user_settings.longitude

        if user_lat and user_lng:
            try:
                user_lat = float(user_lat)
                user_lng = float(user_lng)
                nodes = list(Node.objects.all())
                top_nodes = priority_calculator.select_top_priority_nodes(
                    nodes=nodes, user_lat=user_lat, user_lng=user_lng, max_nodes=5
                )
                priorities, traffic_scores = priority_calculator.calculate_node_priorities(
                    nodes=nodes, user_lat=user_lat, user_lng=user_lng, use_ai_model=True
                )
                priority_info = {
                    'user_location': {'lat': user_lat, 'lng': user_lng},
                    'top_nodes': [
                        {'id': n.id, 'name': n.name, 'score': round(s, 3)}
                        for n, s in top_nodes
                    ],
                    'all_priorities': {str(k): round(v, 3) for k, v in priorities.items()},
                }
            except (ValueError, TypeError):
                pass

        return Response({
            'readings': readings_data,
            'stats': {
                'total_bins': total_bins,
                'critical_bins': critical_bins,
                'warning_bins': warning_bins,
                'normal_bins': total_bins - critical_bins - warning_bins,
                'avg_fill_pct': avg_fill,
            },
            'latest_route': latest_route,
            'notif_count': notif_count,
            'model_version': model_version,
            'priority_info': priority_info,
        })


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
class ProfileAPIView(APIView):
    authentication_classes = _AUTH
    permission_classes = _PERM

    def get(self, request):
        ser = UserSerializer(request.user)
        return Response(ser.data)

    def put(self, request):
        ser = UserSerializer(request.user, data=request.data, partial=True)
        ser.is_valid(raise_exception=True)
        ser.save()
        return Response(ser.data)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class SettingsAPIView(APIView):
    authentication_classes = _AUTH
    permission_classes = _PERM

    def get(self, request):
        obj, _ = UserSetting.objects.get_or_create(user=request.user)
        ser = UserSettingSerializer(obj)
        return Response(ser.data)

    def put(self, request):
        obj, _ = UserSetting.objects.get_or_create(user=request.user)
        ser = UserSettingSerializer(obj, data=request.data, partial=True)
        ser.is_valid(raise_exception=True)
        ser.save()
        return Response(ser.data)


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------
class NotificationListAPIView(generics.ListAPIView):
    authentication_classes = _AUTH
    permission_classes = _PERM
    serializer_class = NotificationSerializer

    def get_queryset(self):
        qs = Notification.objects.filter(
            user__in=[None, self.request.user]
        ).order_by('-created_at')
        level = self.request.GET.get('level')
        if level:
            qs = qs.filter(level=level.upper())
        unread_only = self.request.GET.get('unread')
        if unread_only == 'true':
            qs = qs.filter(is_read=False)
        return qs[:50]


class NotificationMarkReadAPIView(APIView):
    authentication_classes = _AUTH
    permission_classes = _PERM

    def post(self, request, pk):
        try:
            notif = Notification.objects.get(pk=pk)
        except Notification.DoesNotExist:
            return Response({'error': 'Not found'}, status=status.HTTP_404_NOT_FOUND)
        notif.is_read = True
        notif.save(update_fields=['is_read'])
        return Response({'ok': True})


class NotificationMarkAllReadAPIView(APIView):
    authentication_classes = _AUTH
    permission_classes = _PERM

    def post(self, request):
        Notification.objects.filter(
            user__in=[None, request.user], is_read=False
        ).update(is_read=True)
        return Response({'ok': True})


# ---------------------------------------------------------------------------
# Current user (me) – used by React to check if logged in
# ---------------------------------------------------------------------------
class MeAPIView(APIView):
    authentication_classes = _AUTH
    permission_classes = _PERM

    def get(self, request):
        ser = UserSerializer(request.user)
        return Response(ser.data)


# ---------------------------------------------------------------------------
# Nodes – list + get-or-create (for dummy data sender bootstrap)
# ---------------------------------------------------------------------------
class NodeListAPIView(generics.ListAPIView):
    """GET /api/v1/nodes/ – all nodes with IDs"""
    authentication_classes = _AUTH
    permission_classes = _PERM
    serializer_class = NodeSerializer

    def get_queryset(self):
        return Node.objects.all().order_by('id')


class EnsureNodeAPIView(APIView):
    """
    POST /api/v1/nodes/ensure/
    { name, latitude, longitude }  → get_or_create node by name.
    """
    authentication_classes = _AUTH
    permission_classes = _PERM

    def post(self, request):
        name = request.data.get('name', '').strip()
        if not name:
            return Response({'error': 'name required'}, status=status.HTTP_400_BAD_REQUEST)

        defaults = {}
        for key in ('latitude', 'longitude'):
            v = request.data.get(key)
            if v is not None:
                try:
                    defaults[key] = float(v)
                except (TypeError, ValueError):
                    pass

        node, created = Node.objects.get_or_create(name=name, defaults=defaults)

        # Update coords if absent on existing node
        if not created and defaults:
            changed = False
            for key, val in defaults.items():
                if getattr(node, key) is None:
                    setattr(node, key, val)
                    changed = True
            if changed:
                node.save(update_fields=list(defaults.keys()))

        return Response({
            'id': node.id, 'name': node.name,
            'latitude': node.latitude, 'longitude': node.longitude,
            'created': created,
        }, status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)


class ReadingSubmitAPIView(APIView):
    """
    POST /api/v1/readings/submit/
    Accepts node_id OR node_name + sensor fields.
    The dummy sender uses node_id (discovered via EnsureNodeAPIView).
    """
    authentication_classes = _AUTH
    permission_classes = _PERM

    def post(self, request):
        data = request.data
        node = None

        node_id = data.get('node_id')
        node_name = data.get('node_name', '').strip()

        if node_id:
            try:
                node = Node.objects.get(id=int(node_id))
            except (Node.DoesNotExist, ValueError, TypeError):
                return Response({'error': f'Node id={node_id} not found'}, status=status.HTTP_404_NOT_FOUND)
        elif node_name:
            try:
                node = Node.objects.get(name=node_name)
            except Node.DoesNotExist:
                return Response(
                    {'error': f'Node {node_name!r} not found. Call /api/v1/nodes/ensure/ first.'},
                    status=status.HTTP_404_NOT_FOUND,
                )
        else:
            return Response({'error': 'node_id or node_name required'}, status=status.HTTP_400_BAD_REQUEST)

        def _f(k, default=0.0):
            try:
                return float(data.get(k, default))
            except (TypeError, ValueError):
                return default

        reading = SensorReading.objects.create(
            node=node,
            temperature=_f('temperature', 25.0),
            humidity=_f('humidity', 60.0),
            gas_level=max(0.0, min(1.0, _f('gas_level'))),
            waste_level=max(0.0, min(1.0, _f('waste_level'))),
            traffic_density=max(0.0, min(1.0, _f('traffic_density'))),
            distance_to_next_bin=data.get('distance_to_next_bin'),
        )
        return Response(SensorReadingSerializer(reading).data, status=status.HTTP_201_CREATED)


# ---------------------------------------------------------------------------
# Auth – Login / Signup / Logout  (no Django template pages needed)
# ---------------------------------------------------------------------------
class LoginAPIView(APIView):
    """Accepts { username, password } and logs in via session."""
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        username = request.data.get('username', '').strip()
        password = request.data.get('password', '')

        if not username or not password:
            return Response(
                {'error': 'Username and password are required.'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Support login with email
        user = authenticate(request, username=username, password=password)
        if user is None and '@' in username:
            try:
                matched = User.objects.get(email__iexact=username)
                user = authenticate(request, username=matched.username, password=password)
            except User.DoesNotExist:
                pass

        if user is None:
            return Response(
                {'error': 'Invalid username or password.'},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        login(request, user)
        return Response(UserSerializer(user).data)


class SignupAPIView(APIView):
    """Accepts { username, email, password, password2 } and creates an account."""
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        username = request.data.get('username', '').strip()
        email = request.data.get('email', '').strip()
        password = request.data.get('password', '')
        password2 = request.data.get('password2', '')

        errors = {}
        if not username:
            errors['username'] = 'Username is required.'
        elif User.objects.filter(username__iexact=username).exists():
            errors['username'] = 'A user with that username already exists.'

        if not email:
            errors['email'] = 'Email is required.'
        elif User.objects.filter(email__iexact=email).exists():
            errors['email'] = 'A user with that email already exists.'

        if not password:
            errors['password'] = 'Password is required.'
        elif password != password2:
            errors['password2'] = 'Passwords do not match.'
        else:
            try:
                validate_password(password)
            except DjangoValidationError as e:
                errors['password'] = ' '.join(e.messages)

        if errors:
            return Response(errors, status=status.HTTP_400_BAD_REQUEST)

        user = User.objects.create_user(
            username=username, email=email, password=password
        )
        from .models import UserSetting
        UserSetting.objects.get_or_create(user=user)
        login(request, user)
        return Response(UserSerializer(user).data, status=status.HTTP_201_CREATED)


class LogoutAPIView(APIView):
    authentication_classes = _AUTH
    permission_classes = [IsAuthenticated]

    def post(self, request):
        logout(request)
        return Response({'ok': True})
