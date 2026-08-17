"""
Populate the database with a realistic, reproducible demonstration network.

Uses the same physically-grounded simulator as the experiment suite, so the
application is exercised on data with the correct dynamics -- diurnal and weekly
demand, temperature-driven decomposition, event bursts, scheduled collections --
rather than on uniform noise that would make every algorithm look identical.
"""
from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from bins.models import (BinGroup, Depot, Node, Notification, SensorHealth,
                         SensorReading, ServiceEvent, UserSetting, Vehicle)
from bins.services import audit, dispatch

BASE_BINS = [
    ("Mirpur 10 Circle", 23.8069, 90.3687, "general"),
    ("Sony Square", 23.7910, 90.3550, "general"),
    ("Mirpur Stadium", 23.8050, 90.3630, "recyclable"),
    ("Mirpur 11 Bazar", 23.8203, 90.3650, "organic"),
    ("Mirpur 14 Gate", 23.8100, 90.3780, "general"),
    ("Pallabi Station", 23.8250, 90.3650, "general"),
    ("Kazipara Market", 23.7980, 90.3720, "organic"),
    ("Shewrapara", 23.7890, 90.3740, "general"),
    ("Agargaon Link", 23.7780, 90.3800, "recyclable"),
    ("Kafrul Depot Road", 23.7900, 90.3850, "general"),
    ("Mirpur 12 Block C", 23.8290, 90.3690, "general"),
    ("Rupnagar", 23.8230, 90.3560, "organic"),
    ("Ibrahimpur", 23.7850, 90.3820, "general"),
    ("Kalshi Crossing", 23.8180, 90.3760, "hazardous"),
    ("Mirpur DOHS", 23.8320, 90.3620, "recyclable"),
    ("Baishteki", 23.7960, 90.3480, "general"),
    ("Monipur School", 23.8020, 90.3665, "general"),
    ("Mirpur 6 Market", 23.8110, 90.3600, "organic"),
    ("Chidiakhana Road", 23.8140, 90.3480, "general"),
    ("Technical More", 23.7930, 90.3585, "hazardous"),
]


class Command(BaseCommand):
    help = "Seed a reproducible demonstration bin network with realistic telemetry."

    def add_arguments(self, parser):
        parser.add_argument("--bins", type=int, default=20, help="Number of bins (max 20).")
        parser.add_argument("--days", type=int, default=21,
                            help="Days of history to generate.")
        parser.add_argument("--interval-min", type=int, default=30,
                            help="Sampling interval in minutes.")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--reset", action="store_true",
                            help="Delete existing bins, readings and history first.")
        parser.add_argument("--user", type=str, default="demo",
                            help="Create/refresh this operator account.")
        parser.add_argument("--password", type=str, default="wastebins2026")

    # ------------------------------------------------------------------
    def handle(self, *args, **options):
        rng = np.random.default_rng(int(options["seed"]))
        n_bins = max(1, min(len(BASE_BINS), int(options["bins"])))
        days = max(1, int(options["days"]))
        interval_min = max(5, int(options["interval_min"]))

        if options["reset"]:
            self.stdout.write("Clearing existing network...")
            ServiceEvent.objects.all().delete()
            SensorReading.objects.all().delete()
            Notification.objects.all().delete()
            Node.objects.all().delete()

        group, _ = BinGroup.objects.get_or_create(name="Mirpur Zone")
        depot, vehicles = dispatch.ensure_default_fleet()
        self.stdout.write(f"Fleet: {depot.name} with {len(vehicles)} vehicles")

        nodes = self._create_nodes(group, n_bins, rng)
        self.stdout.write(f"Bins: {len(nodes)}")

        total = self._generate_telemetry(nodes, days, interval_min, rng)
        self.stdout.write(f"Readings: {total:,} over {days} days at {interval_min}-minute cadence")

        self._create_user(options["user"], options["password"])
        self._create_notifications(nodes)

        audit.append("config", {
            "action": "seed_demo", "bins": len(nodes), "days": days,
            "interval_min": interval_min, "seed": int(options["seed"]),
            "readings": total,
        }, actor="seed")

        self.stdout.write(self.style.SUCCESS(
            f"\nSeeded {len(nodes)} bins, {total:,} readings.\n"
            f"Sign in as '{options['user']}' / '{options['password']}'.\n"
            f"Next: python manage.py train_forward --quick"
        ))

    # ------------------------------------------------------------------
    def _create_nodes(self, group, n_bins, rng):
        nodes = []
        now = timezone.now()
        for i in range(n_bins):
            name, lat, lng, stream = BASE_BINS[i]
            # A commercial bin is emptied on a tighter window than a residential one.
            commercial = stream in ("organic", "hazardous")
            node, _ = Node.objects.update_or_create(
                name=name,
                defaults=dict(
                    group=group, latitude=lat, longitude=lng, waste_stream=stream,
                    capacity_liters=float(rng.choice([660, 1100, 1100, 1700])),
                    waste_density_kg_per_m3=float(rng.uniform(180, 260)),
                    service_minutes=float(rng.uniform(2.5, 6.0)),
                    window_start_min=int(rng.choice([0, 0, 0, 120, 180])),
                    window_end_min=int(rng.choice([480, 480, 420, 360])),
                    is_active=True,
                    last_collected_at=now - timedelta(
                        hours=float(rng.uniform(2, 60 if commercial else 96))),
                ),
            )
            nodes.append(node)
        return nodes

    # ------------------------------------------------------------------
    def _generate_telemetry(self, nodes, days, interval_min, rng):
        """
        Latent fill/decomposition dynamics observed through noisy sensors.

        The generative process is documented in the manuscript and mirrors
        ``experiments/sim.py``: arrivals follow a diurnal/weekly demand profile
        with Poisson burst events, gas accumulates by temperature-driven
        (Arrhenius q10) decomposition of the organic fraction, and internal
        temperature rises with that decomposition.
        """
        now = timezone.now()
        steps = int(days * 24 * 60 / interval_min)
        dt_h = interval_min / 60.0
        start = now - timedelta(minutes=steps * interval_min)

        SensorReading.objects.filter(node__in=nodes).delete()
        # Health rows summarise readings, so they are stale the moment the
        # readings are replaced.  Leaving them behind meant the dashboard reported
        # trust derived from data that no longer existed -- and because the health
        # endpoint serves stored rows rather than recomputing on every poll (so a
        # page refresh is not a fleet-wide reassessment), the stale figures were
        # indistinguishable from live ones and survived a full reseed unchanged.
        SensorHealth.objects.filter(node__in=nodes).delete()

        # ---- weather is shared across the network ---------------------------
        # Ambient temperature and rain are properties of the city, not of a bin.
        # Drawing them per bin (as this did) meant it rained on one bin and not on
        # the one 300 m away, and the ambient temperature differed between bins at
        # the same instant -- physically impossible, and it removed the shared
        # structure that makes fault detection hard: a lone +22-point humidity
        # jump is exactly what a fleet-relative detector should flag, so the
        # simulator was manufacturing anomalies and then being surprised by them.
        # Measured effect: 6 of 80 channels on the seeded network were flagged
        # degraded or suspect despite no fault ever being injected.
        #
        # Rain also persists: a shower that appears and vanishes independently
        # every 30 minutes is not weather.  A two-state chain gives showers a
        # realistic duration and, being shared, produces the genuinely correlated
        # fleet-wide humidity excursions that `faults.weather_correlated` models.
        ambient_series = np.empty(steps, dtype=float)
        rain_series = np.zeros(steps, dtype=float)
        raining = False
        for step in range(steps):
            stamp = start + timedelta(minutes=step * interval_min)
            hour = stamp.hour + stamp.minute / 60.0
            ambient_series[step] = (
                28.0
                + 4.0 * math.sin(2 * math.pi * (stamp.timetuple().tm_yday - 120) / 365.0)
                + 5.0 * math.sin(2 * math.pi * (hour - 9) / 24.0)
                + float(rng.normal(0, 1.2))
            )
            # ~4% of steps wet in the long run, in showers rather than flickers.
            raining = (rng.random() < 0.70) if raining else (rng.random() < 0.018)
            rain_series[step] = 1.0 if raining else 0.0

        batch, total = [], 0
        for node in nodes:
            # A persistent microclimate offset is legitimate heterogeneity -- sun
            # versus shade, an enclosed yard versus an open kerb -- and unlike
            # per-step independent noise it is a stable bias the detectors can
            # learn as normal for that bin rather than read as an excursion.
            microclimate = float(rng.normal(0, 0.6))
            organic = {"organic": 0.85, "general": 0.55,
                       "recyclable": 0.20, "hazardous": 0.40}[node.waste_stream]
            arrival = float(rng.uniform(0.008, 0.030)) * dt_h
            period_h = float(rng.choice([36, 48, 60, 72]))
            phase = float(rng.uniform(0, period_h))

            fill = float(rng.uniform(0.0, 0.3))
            decomp = 0.0
            bursts = set(np.flatnonzero(rng.random(steps) < 0.004).tolist())
            last_collect = start

            for step in range(steps):
                stamp = start + timedelta(minutes=step * interval_min)
                hour = stamp.hour + stamp.minute / 60.0
                dow = stamp.weekday()

                # demand: two daily peaks, heavier on the Friday/Saturday weekend
                diurnal = (0.6 + 0.7 * math.exp(-((hour - 9) ** 2) / 8.0)
                           + 0.9 * math.exp(-((hour - 20) ** 2) / 6.0))
                weekend = 1.35 if dow in (4, 5) else 1.0
                fill = min(1.3, fill + max(0.0, arrival * diurnal * weekend
                                           * (1 + float(rng.normal(0, 0.15)))))
                if step in bursts:
                    fill = min(1.3, fill + float(rng.uniform(0.15, 0.45)))

                ambient = float(ambient_series[step]) + microclimate
                rain = float(rain_series[step])
                humidity = float(np.clip(62.0 - 0.8 * (ambient - 28.0) + 22.0 * rain
                                         + rng.normal(0, 4.0), 25.0, 100.0))

                q10 = 2.0 ** ((ambient - 20.0) / 10.0)
                decomp = (decomp + organic * fill * 0.010 * q10 * dt_h) * (0.98 ** dt_h)
                gas = 1.0 - math.exp(-1.4 * decomp)
                temp = ambient + 6.0 * gas

                elapsed_h = (stamp - start).total_seconds() / 3600.0
                due = (elapsed_h - phase) % period_h < dt_h and elapsed_h > phase

                batch.append(SensorReading(
                    node=node, timestamp=stamp,
                    waste_level=round(float(np.clip(fill + rng.normal(0, 0.02), 0, 1.3)), 4),
                    gas_level=round(float(np.clip(gas + rng.normal(0, 0.03), 0, 1)), 4),
                    temperature=round(float(temp + rng.normal(0, 0.5)), 3),
                    humidity=round(float(np.clip(humidity + rng.normal(0, 2.0), 0, 100)), 2),
                    traffic_density=round(float(np.clip(
                        0.25 + 0.5 * math.exp(-((hour - 9) ** 2) / 6.0)
                        + 0.55 * math.exp(-((hour - 18) ** 2) / 6.0)
                        + rng.normal(0, 0.05), 0, 1)), 3),
                    battery_v=round(float(3.7 - 0.25 * (step / max(steps, 1))
                                          + rng.normal(0, 0.02)), 3),
                    rssi_dbm=round(float(rng.normal(-78, 6)), 1),
                    is_synthetic=True,
                ))
                total += 1

                if due or fill >= 1.25:
                    ServiceEvent.objects.create(
                        node=node, collected_at=stamp,
                        fill_at_collection=round(float(fill), 4),
                        load_kg=round(node.load_kg(fill), 2),
                        wait_hours=round((stamp - last_collect).total_seconds() / 3600.0, 3),
                        was_overflowing=fill >= 1.0,
                    )
                    last_collect = stamp
                    fill = float(rng.uniform(0.0, 0.05))
                    decomp = 0.0

                if len(batch) >= 5000:
                    SensorReading.objects.bulk_create(batch, batch_size=1000)
                    batch = []

            node.last_collected_at = last_collect
            node.save(update_fields=["last_collected_at"])

        if batch:
            SensorReading.objects.bulk_create(batch, batch_size=1000)
        return total

    # ------------------------------------------------------------------
    def _create_user(self, username, password):
        user, created = User.objects.get_or_create(
            username=username,
            defaults={"email": f"{username}@wastebins.local",
                      "first_name": "Demo", "last_name": "Operator",
                      "is_staff": True},
        )
        user.set_password(password)
        user.save()
        UserSetting.objects.update_or_create(
            user=user,
            defaults={"latitude": 23.8069, "longitude": 90.3687,
                      "location_name": "Mirpur Central Depot",
                      "polling_interval_sec": 15},
        )
        self.stdout.write(f"Operator account: {username} ({'created' if created else 'updated'})")

    # ------------------------------------------------------------------
    def _create_notifications(self, nodes):
        Notification.objects.all().delete()
        entries = []
        for node in nodes:
            latest = node.get_latest_reading()
            if latest is None or latest.waste_level is None:
                continue
            if latest.waste_level >= 0.9:
                entries.append(Notification(
                    node=node, level="CRITICAL", category="overflow",
                    message=f"{node.name} is at {latest.waste_level * 100:.0f}% and "
                            f"needs collection now."))
            elif (latest.gas_level or 0) >= 0.6:
                entries.append(Notification(
                    node=node, level="WARN", category="hazard",
                    message=f"{node.name} shows elevated gas ({latest.gas_level:.2f}); "
                            f"decomposition hazard likely."))
        entries.append(Notification(
            level="INFO", category="system",
            message="Demonstration network seeded. Run `train_forward` to enable "
                    "predictive prioritisation."))
        Notification.objects.bulk_create(entries)
        self.stdout.write(f"Notifications: {len(entries)}")
