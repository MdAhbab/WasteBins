import json
import math
from datetime import datetime
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db import transaction
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponseForbidden
from django.shortcuts import render, redirect
from django.views.decorators.http import require_GET, require_POST
from .forms import SignupForm, ProfileForm, SettingsForm
from .models import Node, SensorReading, AICost, CollectionRoute, Notification, UserSetting, BinGroup
from .serializers import serialize_reading, serialize_node
from .utils.dijkstra import compute_route
from .utils.ai.model_store import load_model, get_model_version
from .utils.ai import train_model as trainer

def _haversine_m(lat1, lon1, lat2, lon2):
    # Haversine distance in meters
    R = 6371000.0
    phi1, phi2 = math.radians(lat1 or 0.0), math.radians(lat2 or 0.0)
    dphi = math.radians((lat2 or 0.0) - (lat1 or 0.0))
    dlambda = math.radians((lon2 or 0.0) - (lon1 or 0.0))
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def _build_graph(nodes, costs_by_node, alpha):
    # Fully connect nodes with haversine distance weighted by normalized predicted cost
    ids = [n.id for n in nodes]
    graph = {nid: [] for nid in ids}
    # Normalize costs
    costs = [costs_by_node.get(nid, 0.0) for nid in ids]
    cmin, cmax = min(costs) if costs else 0.0, max(costs) if costs else 1.0
    def norm(c):
        if cmax - cmin < 1e-9:
            return 0.0
        return (c - cmin) / (cmax - cmin)
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i == j:
                continue
            d = _haversine_m(u.latitude or 0.0, u.longitude or 0.0, v.latitude or 0.0, v.longitude or 0.0)
            w = d * (1.0 + alpha * norm(costs_by_node.get(v.id, 0.0)))
            graph[u.id].append((v.id, w))
    return graph

def signup_view(request):
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            UserSetting.objects.get_or_create(user=user)
            return redirect('dashboard')
    else:
        form = SignupForm()
    return render(request, 'bins/auth/signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('dashboard')
        messages.error(request, 'Invalid credentials.')
    return render(request, 'bins/auth/login.html')

def logout_view(request):
    logout(request)
    return redirect('login')

@login_required
def dashboard_view(request):
    latest_readings = (SensorReading.objects
                       .select_related('node')
                       .order_by('-timestamp')[:10])
    latest_route = CollectionRoute.objects.order_by('-timestamp').first()
    notif_count = Notification.objects.filter(user__in=[None, request.user], is_read=False).count()
    return render(request, 'bins/bins/dashboard.html', {
        'latest_readings': [serialize_reading(r) for r in latest_readings],
        'latest_route': latest_route.route_data if latest_route else None,
        'notif_count': notif_count,
        'now': datetime.now(),
        'model_version': get_model_version(),
    })

@login_required
def profile_view(request):
    if request.method == 'POST':
        form = ProfileForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, 'Profile updated.')
            return redirect('profile')
    else:
        form = ProfileForm(instance=request.user)
    return render(request, 'bins/bins/profile.html', {'form': form})

@login_required
def settings_view(request):
    settings_obj, _ = UserSetting.objects.get_or_create(user=request.user)
    if request.method == 'POST':
        form = SettingsForm(request.POST, instance=settings_obj)
        if form.is_valid():
            form.save()
            messages.success(request, 'Settings saved.')
            return redirect('settings')
    else:
        form = SettingsForm(instance=settings_obj)
    return render(request, 'bins/bins/settings.html', {'form': form})

@require_GET
@login_required
def api_latest_readings(request):
    N = int(request.GET.get('limit', 10))
    readings = (SensorReading.objects
                .select_related('node')
                .order_by('-timestamp')[:N])
    data = [serialize_reading(r) for r in readings]
    return JsonResponse({'readings': data})

@require_POST
@login_required
def api_submit_reading(request):
    try:
        payload = json.loads(request.body.decode('utf-8'))
        node_id = payload['node_id']
        node = Node.objects.get(id=node_id)
        r = SensorReading.objects.create(
            node=node,
            temperature=float(payload.get('temperature', 0.0)),
            humidity=float(payload.get('humidity', 0.0)),
            gas_level=float(payload.get('gas_level', 0.0)),
            distance_to_next_bin=payload.get('distance_to_next_bin'),
        )
        return JsonResponse({'status': 'ok', 'reading': serialize_reading(r)}, status=201)
    except Exception as e:
        return HttpResponseBadRequest(json.dumps({'error': str(e)}), content_type='application/json')

@require_POST
@login_required
def api_train_model(request):
    # Basic protection: staff-only
    if not request.user.is_staff:
        return HttpResponseForbidden(json.dumps({'error': 'Forbidden'}), content_type='application/json')
    result = trainer.train_from_db()
    return JsonResponse({'status': 'trained', **result})

@require_POST
@login_required
def api_predict_cost(request):
    # Accept either explicit features or compute from latest readings
    model = load_model()
    if model is None:
        return HttpResponseBadRequest(json.dumps({'error': 'Model not found. Train first.'}), content_type='application/json')
    try:
        payload = json.loads(request.body.decode('utf-8')) if request.body else {}
        per_node = payload.get('per_node')  # [{node_id, features: {...}}]
        meta_version = get_model_version()
        outputs = []
        if per_node:
            # Use provided features (columns must match trained model)
            # For simplicity, assume server knows feature order from meta
            from .utils.ai.model_store import meta_path
        else:
            # Build features from latest readings like training code
            df = trainer._extract_features_df()
            feature_cols = [c for c in df.columns if c not in ('node_id',)]
            X = df[[c for c in feature_cols if c != 'node_id']].values
            preds = model.predict(X)
            for idx, row in df.iterrows():
                node_id = int(row['node_id'])
                pred = float(preds[idx])
                AICost.objects.create(
                    node=Node.objects.get(id=node_id),
                    features={k: float(row[k]) for k in feature_cols if k != 'node_id'},
                    predicted_cost=pred,
                    model_version=meta_version,
                )
                outputs.append({
                    'node_id': node_id,
                    'predicted_cost': pred,
                    'model_version': meta_version,
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                })
        return JsonResponse({'predictions': outputs})
    except Exception as e:
        return HttpResponseBadRequest(json.dumps({'error': str(e)}), content_type='application/json')

@require_POST
@login_required
def api_compute_route(request):
    try:
        payload = json.loads(request.body.decode('utf-8')) if request.body else {}
        group_name = payload.get('group')  # optional: route only within a group
        alpha = float(payload.get('alpha', getattr(settings, 'ROUTING_ALPHA', 0.5)))
        source_node_id = payload.get('source_node_id')
        target_node_ids = payload.get('target_node_ids')  # optional

        nodes_qs = Node.objects.all().order_by('id')
        if group_name:
            nodes_qs = nodes_qs.filter(group__name=group_name)
        nodes = list(nodes_qs)
        if not nodes:
            return HttpResponseBadRequest(json.dumps({'error': 'No nodes available'}), content_type='application/json')

        # use latest AICost per node
        costs_by_node = {}
        for n in nodes:
            cost = (AICost.objects.filter(node=n).order_by('-timestamp').first())
            costs_by_node[n.id] = float(cost.predicted_cost) if cost else 0.0

        graph = _build_graph(nodes, costs_by_node, alpha)
        if source_node_id is None:
            source_node_id = nodes[0].id
        targets = target_node_ids or [n.id for n in nodes if n.id != source_node_id]

        result = compute_route(graph, source=source_node_id, targets=targets)
        edges = []
        path = result.get('path', [])
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            # find weight
            w = next((w for (nb, w) in graph[u] if nb == v), 0.0)
            edges.append({'u': u, 'v': v, 'w': w})
        route_json = {'path': path, 'edges': edges, 'alpha': alpha}

        cr = CollectionRoute.objects.create(
            route_data=route_json,
            total_cost=float(result.get('total_cost', 0.0)),
            generated_by=request.user
        )
        return JsonResponse({'route': route_json, 'total_cost': cr.total_cost, 'id': cr.id})
    except Exception as e:
        return HttpResponseBadRequest(json.dumps({'error': str(e)}), content_type='application/json')

@require_GET
@login_required
def api_notifications(request):
    notifs = Notification.objects.filter(user__in=[None, request.user]).order_by('-created_at')[:20]
    data = [{
        'id': n.id, 'message': n.message, 'level': n.level,
        'is_read': n.is_read, 'created_at': n.created_at.isoformat()
    } for n in notifs]
    return JsonResponse({'notifications': data})

@require_GET
@login_required
def api_model_info(request):
    return JsonResponse({'model_version': get_model_version()})