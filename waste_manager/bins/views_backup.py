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
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.middleware.csrf import get_token
from .forms import SignupForm, ProfileForm, SettingsForm, LocationForm
from .models import Node, SensorReading, AICost, CollectionRoute, Notification, UserSetting, BinGroup
from .serializers import serialize_reading, serialize_node
from .utils.dijkstra import compute_route, compute_optimal_route
from .utils.ai.model_store import load_model, get_model_version
from .utils.ai import train_model as trainer
from .utils.priority_calculator import priority_calculator

def _haversine_m(lat1, lon1, lat2, lon2):
    # Haversine distance in meters
    R = 6371000.0
    phi1, phi2 = math.radians(lat1 or 0.0), math.radians(lat2 or 0.0)
    dphi = math.radians((lat2 or 0.0) - (lat1 or 0.0))
    dlambda = math.radians((lon2 or 0.0) - (lon1 or 0.0))
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def _compute_priority(distance_m, waste_level, gas_level, temperature, humidity,
                      dist_weight=0.2, waste_weight=0.4, gas_weight=0.3, temp_hum_weight=0.1,
                      dist_cap_m=2000.0):
    """Compute a priority score in [0,1] where higher => more urgent.
    - distance_m: farther => lower priority (we invert after normalization)
    - waste_level: 0..1 higher => higher priority
    - gas_level: 0..1 higher => higher priority
    - temperature/humidity: slightly increase
    """
    # Normalize distance: 0 (near) to 1 (far); cap to avoid extreme dilution
    dn = min(1.0, max(0.0, distance_m / max(1.0, dist_cap_m)))
    dist_component = 1.0 - dn  # nearer => higher priority
    wl = 0.0 if waste_level is None else max(0.0, min(1.0, float(waste_level)))
    gl = max(0.0, min(1.0, float(gas_level or 0.0)))
    # Temperature: center at 25C, scale +/-10C to [0,1]
    t = float(temperature or 0.0)
    t_score = min(1.0, max(0.0, abs(t - 25.0) / 10.0))
    # Humidity: >70% slightly higher, map 50..90 to 0..1
    h = float(humidity or 0.0)
    h_score = min(1.0, max(0.0, (h - 50.0) / 40.0))
    th = 0.5 * t_score + 0.5 * h_score
    score = (
        dist_weight * dist_component +
        waste_weight * wl +
        gas_weight * gl +
        temp_hum_weight * th
    )
    return max(0.0, min(1.0, score))

def _build_graph(nodes, priorities_by_node, alpha):
    """Fully connect nodes with base distance, then decrease the cost towards high-priority destinations.
    Weight rule (normalized by x10):
        w(u->v) = base_distance_m / (1 + alpha * (priority[v] * 10))
    This inversely scales cost by destination priority as requested.
    """
    ids = [n.id for n in nodes]
    graph = {nid: [] for nid in ids}
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i == j:
                continue
            base = _haversine_m(u.latitude or 0.0, u.longitude or 0.0, v.latitude or 0.0, v.longitude or 0.0)
            pv = float(priorities_by_node.get(v.id, 0.0))
            w = base / (1.0 + alpha * (pv * 10.0))
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

@ensure_csrf_cookie
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
@csrf_exempt
#@login_required
@login_required
def dashboard_view(request):
    """
    Enhanced dashboard view with location integration and priority information
    """
    # Ensure user has settings - only for authenticated users
    user_settings, _ = UserSetting.objects.get_or_create(user=request.user)
    
    # Get latest reading per node with last update timestamps
    latest_per_node = {}
    for r in SensorReading.objects.select_related('node').order_by('node_id', '-timestamp'):
        if r.node_id not in latest_per_node:
            latest_per_node[r.node_id] = r
    
    latest_readings = list(latest_per_node.values())
    
    # Enhance readings with node last update timestamps
    for reading in latest_readings:
        reading.node_last_update = reading.node.last_update
    
    # Get latest route
    latest_route = CollectionRoute.objects.order_by('-timestamp').first()
    
    # Get notification count - only for authenticated users
    notif_count = Notification.objects.filter(
        user__in=[None, request.user], 
        is_read=False
    ).count()
    
    # Get model information
    model_version = get_model_version()
    
    # Calculate priority scores for nodes if user location is available
    priority_info = None
    user_lat = request.GET.get('lat')
    user_lng = request.GET.get('lng')
    
    # Use saved location if no GET parameters provided
    if not user_lat or not user_lng:
        if user_settings.has_location():
            user_lat = user_settings.latitude
            user_lng = user_settings.longitude
    
    if user_lat and user_lng:
        try:
            user_lat = float(user_lat)
            user_lng = float(user_lng)
            
            # Get all nodes
            nodes = Node.objects.all()
            
            # Calculate priorities
            priorities = priority_calculator.calculate_node_priorities(
                nodes=list(nodes),
                user_lat=user_lat,
                user_lng=user_lng,
                use_ai_model=True
            )
            
            # Get top priority nodes (1-5)
            top_nodes = priority_calculator.select_top_priority_nodes(
                nodes=list(nodes),
                user_lat=user_lat,
                user_lng=user_lng,
                max_nodes=5
            )
            
            priority_info = {
                'user_location': {'lat': user_lat, 'lng': user_lng},
                'priorities': priorities,
                'top_nodes': [(node.id, node.name, score) for node, score in top_nodes]
            }
            
        except (ValueError, TypeError):
            priority_info = None
    
    return render(request, 'bins/bins/dashboard.html', {
        'latest_readings': [serialize_reading(r) for r in latest_readings],
        'latest_route': latest_route.route_data if latest_route else None,
        'notif_count': notif_count,
        'now': datetime.now(),
        'model_version': model_version,
        'priority_info': priority_info,
        'user_settings': user_settings,
    })
@csrf_exempt
#@login_required
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


@csrf_exempt
@require_POST
#@login_required
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
            waste_level=(
                float(payload.get('waste_level'))
                if payload.get('waste_level') is not None else None
            ),
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
    """
    Enhanced route computation API using new priority-based Dijkstra implementation
    """
    try:
        payload = json.loads(request.body.decode('utf-8')) if request.body else {}
        
        # Extract parameters
        group_name = payload.get('group')  # optional: route only within a group
        alpha = float(payload.get('alpha', getattr(settings, 'ROUTING_ALPHA', 0.5)))
        source_node_id = payload.get('source_node_id')
        target_node_ids = payload.get('target_node_ids')  # optional
        
        # User location parameters
        user_lat = payload.get('user_lat')
        user_lng = payload.get('user_lng')
        top_n = int(payload.get('top_n', 5))  # Default to 5 nodes as specified
        
        # Get nodes
        nodes_qs = Node.objects.all().order_by('id')
        if group_name:
            nodes_qs = nodes_qs.filter(group__name=group_name)
        nodes = list(nodes_qs)
        
        if not nodes:
            return HttpResponseBadRequest(
                json.dumps({'error': 'No nodes available'}), 
                content_type='application/json'
            )
        
        # Validate user location
        if user_lat is None or user_lng is None:
            return HttpResponseBadRequest(
                json.dumps({'error': 'User location (user_lat, user_lng) required'}),
                content_type='application/json'
            )
        
        try:
            user_lat = float(user_lat)
            user_lng = float(user_lng)
        except (ValueError, TypeError):
            return HttpResponseBadRequest(
                json.dumps({'error': 'Invalid user location coordinates'}),
                content_type='application/json'
            )
        
        # Calculate priorities using the new system
        priorities = priority_calculator.calculate_node_priorities(
            nodes=nodes,
            user_lat=user_lat,
            user_lng=user_lng,
            use_ai_model=True
        )
        
        # Select top N priority nodes if specified
        if top_n > 0:
            top_priority_nodes = priority_calculator.select_top_priority_nodes(
                nodes=nodes,
                user_lat=user_lat,
                user_lng=user_lng,
                max_nodes=top_n
            )
            # Use only the top priority nodes for routing
            nodes = [node for node, _ in top_priority_nodes]
            # Update priorities dict to only include selected nodes
            priorities = {node.id: priorities[node.id] for node in nodes}
        
        # Compute optimal route using enhanced Dijkstra
        user_location = {'lat': user_lat, 'lng': user_lng}
        
        result = compute_optimal_route(
            nodes=nodes,
            priority_scores=priorities,
            source_node_id=source_node_id,
            user_location=user_location,
            alpha=alpha
        )
        
        # Build edges information for visualization
        edges = []
        path = result.get('path', [])
        graph_edges = result.get('graph_edges', {})
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            # Find weight from graph
            weight = 0.0
            for neighbor_id, w in graph_edges.get(u, []):
                if neighbor_id == v:
                    weight = w
                    break
            edges.append({'u': u, 'v': v, 'w': weight})
        
        # Prepare route data
        route_json = {
            'path': path,
            'edges': edges,
            'alpha': alpha,
            'priority_scores': priorities,
            'source': result.get('source_type', 'user_location'),
            'user_location': user_location,
            'top_n_selected': top_n,
            'algorithm_version': 'priority_based_v2'
        }
        
        # Save route to database
        cr = CollectionRoute.objects.create(
            route_data=route_json,
            total_cost=float(result.get('total_cost', 0.0)),
            generated_by=request.user
        )
        
        return JsonResponse({
            'route': route_json, 
            'total_cost': cr.total_cost, 
            'id': cr.id,
            'priorities': priorities,
            'selected_nodes': [n.id for n in nodes]
        })
        
    except Exception as e:
        return HttpResponseBadRequest(
            json.dumps({'error': f'Route computation failed: {str(e)}'}), 
            content_type='application/json'
        )

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

@require_POST
@login_required
def api_update_location(request):
    """
    API endpoint to update user's location
    """
    try:
        payload = json.loads(request.body.decode('utf-8'))
        latitude = payload.get('latitude')
        longitude = payload.get('longitude')
        location_name = payload.get('location_name', '')
        
        # Validate coordinates
        if latitude is None or longitude is None:
            return HttpResponseBadRequest(
                json.dumps({'error': 'Both latitude and longitude are required'}),
                content_type='application/json'
            )
        
        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except (ValueError, TypeError):
            return HttpResponseBadRequest(
                json.dumps({'error': 'Invalid latitude or longitude format'}),
                content_type='application/json'
            )
        
        # Validate coordinate ranges
        if not (-90 <= latitude <= 90):
            return HttpResponseBadRequest(
                json.dumps({'error': 'Latitude must be between -90 and 90 degrees'}),
                content_type='application/json'
            )
        
        if not (-180 <= longitude <= 180):
            return HttpResponseBadRequest(
                json.dumps({'error': 'Longitude must be between -180 and 180 degrees'}),
                content_type='application/json'
            )
        
        # Update user settings
        user_settings, created = UserSetting.objects.get_or_create(user=request.user)
        user_settings.latitude = latitude
        user_settings.longitude = longitude
        user_settings.location_name = location_name
        user_settings.save()
        
        return JsonResponse({
            'success': True,
            'message': 'Location updated successfully',
            'location': {
                'latitude': latitude,
                'longitude': longitude,
                'name': location_name
            }
        })
        
    except Exception as e:
        return HttpResponseBadRequest(
            json.dumps({'error': f'Failed to update location: {str(e)}'}),
            content_type='application/json'
        )

@require_GET
@login_required
def api_get_user_location(request):
    """
    API endpoint to get user's saved location
    """
    try:
        user_settings = UserSetting.objects.get(user=request.user)
        if user_settings.has_location():
            return JsonResponse({
                'success': True,
                'location': user_settings.get_location_dict()
            })
        else:
            return JsonResponse({
                'success': False,
                'message': 'No location set'
            })
    except UserSetting.DoesNotExist:
        return JsonResponse({
            'success': False,
            'message': 'User settings not found'
        })

@require_GET
@ensure_csrf_cookie
def api_csrf(request):
    # Returns a CSRF token and sets csrftoken cookie
    return JsonResponse({'csrfToken': get_token(request)})