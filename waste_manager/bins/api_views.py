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
    CollectionRoute, RoutePlan,
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
    """
    Everything the operator's home screen needs, in one round trip.

    Deliberately read-only and cheap: it scores bins (milliseconds) but never
    solves a routing problem, so it is safe to poll.  Planning is a separate,
    explicit POST.
    """

    authentication_classes = _AUTH
    permission_classes = _PERM

    def get(self, request):
        from .services import dispatch, models_registry, telemetry

        nodes = list(Node.objects.filter(is_active=True).select_related('group'))
        node_ids = [n.id for n in nodes]

        latest_per_node = telemetry.latest_reading_map(node_ids)
        readings = [latest_per_node[nid] for nid in node_ids if nid in latest_per_node]
        # Serializing the nested node per reading would re-query; attach the
        # already-loaded instances instead.
        by_id = {n.id: n for n in nodes}
        for reading in readings:
            reading.node = by_id.get(reading.node_id, reading.node)
        readings_data = SensorReadingSerializer(readings, many=True).data

        total_bins = len(nodes)
        fills = [float(r.waste_level) for r in readings if r.waste_level is not None]
        critical_bins = sum(1 for f in fills if f >= 0.85)
        warning_bins = sum(1 for f in fills if 0.65 <= f < 0.85)
        offline_bins = total_bins - len(fills)
        avg_fill = round(sum(fills) / len(fills) * 100, 1) if fills else 0.0

        latest_plan = RoutePlan.objects.order_by('-timestamp').first()
        latest_route_obj = CollectionRoute.objects.order_by('-timestamp').first()

        notif_count = Notification.objects.filter(
            user__in=[None, request.user], is_read=False
        ).count()

        user_settings, _ = UserSetting.objects.get_or_create(user=request.user)
        user_lat = request.GET.get('lat') or user_settings.latitude
        user_lng = request.GET.get('lng') or user_settings.longitude
        try:
            user_lat = float(user_lat) if user_lat not in (None, '') else None
            user_lng = float(user_lng) if user_lng not in (None, '') else None
        except (TypeError, ValueError):
            user_lat = user_lng = None

        priority_info = None
        health_summary = None
        if nodes:
            scores = dispatch.score_bins(nodes, user_lat=user_lat, user_lng=user_lng)
            tiered = dispatch.apply_equity(scores, nodes)
            ranked = sorted(
                scores.items(),
                key=lambda kv: (tiered[kv[0]].tier if kv[0] in tiered else 1,
                                -(tiered[kv[0]].score if kv[0] in tiered else kv[1]['priority'])),
            )
            priority_info = {
                'user_location': ({'lat': user_lat, 'lng': user_lng}
                                  if user_lat is not None else None),
                'top_nodes': [
                    {
                        'id': nid,
                        'name': entry['name'],
                        'score': round(float(entry['priority']), 3),
                        'effective_score': round(float(tiered[nid].score), 3)
                        if nid in tiered else None,
                        'tier': tiered[nid].tier if nid in tiered else 1,
                        'hazard_prob': round(float(entry.get('hazard_prob', 0.0)), 3),
                        'confidence': entry['rule']['confidence'],
                        'actionable': entry['actionable'],
                        'hours_since_collection': entry['hours_since_collection'],
                        'predicted_overflow_h': (entry['model'] or {}).get('tto_p10_h'),
                    }
                    for nid, entry in ranked[:8]
                ],
                'all_priorities': {
                    str(nid): round(float(entry['priority']), 3)
                    for nid, entry in scores.items()
                },
                'low_confidence_bins': [
                    nid for nid, entry in scores.items() if not entry['actionable']
                ],
            }
            health_rows = {nid: entry['health'] for nid, entry in scores.items()}
            degraded = sum(
                1 for channels in health_rows.values()
                for a in channels.values() if a['status'] != 'ok'
            )
            trusts = [a['trust'] for channels in health_rows.values()
                      for a in channels.values()] or [1.0]
            health_summary = {
                'channels_degraded': degraded,
                'mean_trust': round(sum(trusts) / len(trusts), 3),
                'min_trust': round(min(trusts), 3),
            }

        return Response({
            'readings': readings_data,
            'stats': {
                'total_bins': total_bins,
                'critical_bins': critical_bins,
                'warning_bins': warning_bins,
                'normal_bins': max(0, len(fills) - critical_bins - warning_bins),
                'offline_bins': offline_bins,
                'avg_fill_pct': avg_fill,
            },
            'latest_route': (CollectionRouteSerializer(latest_route_obj).data
                             if latest_route_obj else None),
            'latest_plan': ({
                'id': latest_plan.id,
                'algorithm': latest_plan.algorithm,
                'metrics': latest_plan.metrics,
                'timestamp': latest_plan.timestamp.isoformat(),
            } if latest_plan else None),
            'notif_count': notif_count,
            'model_version': get_model_version(),
            'model_available': models_registry.load_forward_bundle() is not None,
            'priority_info': priority_info,
            'health_summary': health_summary,
        })


class NodeHistoryAPIView(APIView):
    """Recent telemetry for one bin, for the detail chart."""

    authentication_classes = _AUTH
    permission_classes = _PERM

    def get(self, request, pk):
        try:
            node = Node.objects.get(pk=pk)
        except Node.DoesNotExist:
            return Response({'error': f'node {pk} not found'},
                            status=status.HTTP_404_NOT_FOUND)
        try:
            limit = max(1, min(1000, int(request.GET.get('limit', 200))))
        except (TypeError, ValueError):
            limit = 200

        rows = list(SensorReading.objects.filter(node=node)
                    .order_by('-timestamp')[:limit])[::-1]
        return Response({
            'node': {'id': node.id, 'name': node.name,
                     'latitude': node.latitude, 'longitude': node.longitude,
                     'waste_stream': node.waste_stream,
                     'last_collected_at': (node.last_collected_at.isoformat()
                                           if node.last_collected_at else None)},
            'points': [
                {
                    'timestamp': r.timestamp.isoformat(),
                    'waste_level': r.waste_level,
                    'gas_level': r.gas_level,
                    'temperature': r.temperature,
                    'humidity': r.humidity,
                    'is_synthetic': r.is_synthetic,
                    'fault_label': r.fault_label,
                }
                for r in rows
            ],
            'service_events': [
                {'collected_at': e.collected_at.isoformat(),
                 'fill_at_collection': e.fill_at_collection,
                 'wait_hours': e.wait_hours}
                for e in node.service_events.order_by('-collected_at')[:20]
            ],
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
        return Node.objects.select_related('group').order_by('id')


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

    Accepts ``node_id`` or ``node_name`` plus sensor fields, singly or as a
    ``readings`` array for batch ingestion.

    A field that is absent or explicitly ``null`` is stored as NULL, *not* as
    zero.  That distinction is the whole point of the renormalisation work: a
    missing measurement must not be indistinguishable from an empty bin.
    """

    authentication_classes = _AUTH
    permission_classes = _PERM

    @staticmethod
    def _resolve_node(item):
        node_id = item.get('node_id')
        node_name = (item.get('node_name') or '').strip()
        if node_id not in (None, ''):
            try:
                return Node.objects.get(id=int(node_id)), None
            except (Node.DoesNotExist, ValueError, TypeError):
                return None, f'Node id={node_id} not found'
        if node_name:
            try:
                return Node.objects.get(name=node_name), None
            except Node.DoesNotExist:
                return None, (f'Node {node_name!r} not found. '
                              f'Call /api/v1/nodes/ensure/ first.')
        return None, 'node_id or node_name required'

    @staticmethod
    def _optional(item, key, lo=None, hi=None):
        """``None`` for an absent/unparseable value, so missingness is preserved."""
        if key not in item or item[key] is None or item[key] == '':
            return None
        try:
            value = float(item[key])
        except (TypeError, ValueError):
            return None
        if value != value:                      # NaN
            return None
        if lo is not None:
            value = max(lo, value)
        if hi is not None:
            value = min(hi, value)
        return value

    def post(self, request):
        payload = request.data
        items = payload.get('readings') if isinstance(payload, dict) else None
        if items is None:
            items = payload if isinstance(payload, list) else [payload]
        if not isinstance(items, list) or not items:
            return Response({'error': 'no readings supplied'},
                            status=status.HTTP_400_BAD_REQUEST)
        if len(items) > 1000:
            return Response({'error': 'batch limited to 1000 readings'},
                            status=status.HTTP_400_BAD_REQUEST)

        created, errors, audit_records = [], [], []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append({'index': index, 'error': 'each reading must be an object'})
                continue
            node, error = self._resolve_node(item)
            if node is None:
                errors.append({'index': index, 'error': error})
                continue

            reading = SensorReading.objects.create(
                node=node,
                temperature=self._optional(item, 'temperature'),
                humidity=self._optional(item, 'humidity', 0.0, 100.0),
                gas_level=self._optional(item, 'gas_level', 0.0, 1.0),
                waste_level=self._optional(item, 'waste_level', 0.0, 1.3),
                traffic_density=self._optional(item, 'traffic_density', 0.0, 1.0) or 0.0,
                distance_to_next_bin=self._optional(item, 'distance_to_next_bin'),
                battery_v=self._optional(item, 'battery_v'),
                rssi_dbm=self._optional(item, 'rssi_dbm'),
                is_synthetic=bool(item.get('is_synthetic', False)),
            )
            created.append(reading)
            audit_records.append({
                'node_id': node.id, 'ts': reading.timestamp.isoformat(),
                'waste': reading.waste_level, 'gas': reading.gas_level,
                'temp': reading.temperature, 'humidity': reading.humidity,
            })

        if not created:
            return Response({'error': 'no readings accepted', 'errors': errors},
                            status=status.HTTP_400_BAD_REQUEST)

        # Ingestion-time side effects, each individually bounded and non-fatal.
        from .services import audit, telemetry
        touched = sorted({r.node_id for r in created})
        health = {}
        try:
            health = telemetry.assess_nodes(touched, persist=True)
        except Exception:
            pass
        try:
            audit.log_readings(audit_records, actor=request.user.username)
        except Exception:
            pass

        if len(created) == 1:
            body = SensorReadingSerializer(created[0]).data
            if health.get(created[0].node_id):
                body['health'] = {c: a.as_dict()
                                  for c, a in health[created[0].node_id].items()}
            return Response(body, status=status.HTTP_201_CREATED)

        return Response({
            'created': len(created),
            'nodes': touched,
            'errors': errors,
        }, status=status.HTTP_201_CREATED)


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
