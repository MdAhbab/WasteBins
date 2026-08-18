"""
Differentiated fuel and CO2 model for refuse collection vehicles.
================================================================

The first submission used a single flat factor of 1.05 kg CO2 per kilometre.
That hides exactly the effects a waste-collection study needs to reason about:
a collection round spends much of its fuel *not* cruising.  This module replaces
the constant with a modal, physically grounded model whose four regimes are
reported separately:

  1. **Cruise**      -- tractive power against rolling resistance and aerodynamic
                        drag at the operating speed, for the *current* vehicle
                        mass (so an empty return leg costs less than a full one).
  2. **Stop-and-go** -- kinetic energy destroyed in braking and re-supplied on
                        each acceleration cycle.  The number of cycles per
                        kilometre grows with traffic friction, which is what
                        makes congested low-speed driving expensive.
  3. **Idle**        -- engine-on time at each bin during the service dwell and
                        while queueing in congestion.
  4. **Compaction**  -- power take-off energy for the lift-and-pack cycle, which
                        scales with the mass actually loaded.

Formulation
-----------
Tractive power at speed ``v`` (m/s) for total mass ``m`` (kg)::

    P_wheel = ( C_rr * m * g  +  0.5 * rho * C_d * A * v^2 ) * v      [W]

Fuel volumetric rate::

    Qdot = P_wheel / (eta_dt * eta_e * LHV)  +  Q_idle                [L/s]

Braking/acceleration energy per stop-go cycle::

    E_cycle = 0.5 * m * v^2 * (1 - eta_regen)                         [J]

CO2 follows from the diesel oxidation factor (2.653 kg CO2 per litre: the IPCC
2006 default of 74.1 t CO2/TJ for gas/diesel oil, applied to the *same* 35.8
MJ/L the energy chain above uses).  A Euro class scales the fuel burned, never
the carbon in a litre of it.  Battery-electric vehicles use a grid intensity
instead, so the module also supports the electrification scenario reviewers of
sustainability venues usually ask about.

Every coefficient is exposed on :class:`VehicleProfile`, defaults are documented
with their source class, and :func:`flat_factor_equivalent` reports the single
number the old model would have used, so the two are directly comparable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional

G = 9.80665                    # m/s^2
RHO_AIR = 1.20                 # kg/m^3 at ~30 C
DIESEL_LHV_MJ_PER_L = 35.8     # MJ/L (43.0 TJ/Gg NCV x ~0.832 kg/L, IPCC t.1.2)

# Carbon intensity of the fuel, on the same net-calorific basis as the LHV above:
# IPCC 2006 vol. 2 ch. 1 table 1.4 gives 74 100 kg CO2/TJ for gas/diesel oil
# (range 72 600 - 74 800), i.e. 0.0741 kg CO2/MJ.
DIESEL_KG_CO2_PER_MJ = 0.0741
# kg CO2 per litre burned.  Derived rather than asserted, because deriving it
# from the *same* LHV the energy chain uses is what keeps one litre worth one
# litre in both halves of the model.  The previous hard-coded 2.68 did not equal
# the product of its own cited inputs (it implied 74.86 t CO2/TJ, a 1.0 % drift).
# The derived 2.6528 sits inside the independently published band anyway:
# DEFRA/BEIS 2025 100 % mineral diesel 2.662 kg/L, US EIA 10.19 kg/gal = 2.692.
DIESEL_KG_CO2_PER_L = DIESEL_KG_CO2_PER_MJ * DIESEL_LHV_MJ_PER_L      # 2.6528


@dataclass
class VehicleProfile:
    """Physical and powertrain parameters of one collection vehicle."""

    name: str = "rigid-refuse-diesel-euro4"
    kerb_mass_kg: float = 12_000.0        # empty two-axle rear-loader
    capacity_kg: float = 6_000.0
    frontal_area_m2: float = 8.5
    drag_coefficient: float = 0.70
    rolling_resistance: float = 0.010     # asphalt in fair urban condition
    engine_efficiency: float = 0.40       # peak brake thermal efficiency
    driveline_efficiency: float = 0.90
    idle_fuel_l_per_h: float = 2.6        # HDV engine idling with accessories
    pto_fuel_l_per_lift: float = 0.045    # hydraulic lift + pack cycle
    regen_fraction: float = 0.0           # 0 for diesel; > 0 for hybrid/electric
    is_electric: bool = False
    kwh_per_l_equivalent: float = 9.94    # diesel energy content, kWh/L
    grid_kg_co2_per_kwh: float = 0.62     # Bangladesh grid average intensity
    euro_class: str = "euro4"

    # Efficiency penalty applied at very low *mean* speed, where an engine spends
    # more of its time away from its best-efficiency island.  The reference is a
    # duty-cycle statistic, so it must be fed the leg's mean speed: the BPR
    # surface in ``traffic.py`` runs 10-34 km/h, so 18 km/h bites below roughly
    # friction 0.75 (peak-hour arterials).  Fed the *moving* speed instead it can
    # never fire at all -- see the note in :func:`leg_emissions`.
    low_speed_penalty_ref_kmh: float = 18.0
    low_speed_penalty_max: float = 0.28

    def powertrain_efficiency(self, speed_kmh: float) -> float:
        base = self.engine_efficiency * self.driveline_efficiency
        if speed_kmh >= self.low_speed_penalty_ref_kmh:
            return base
        shortfall = 1.0 - (speed_kmh / max(self.low_speed_penalty_ref_kmh, 1e-6))
        return base * (1.0 - self.low_speed_penalty_max * shortfall)

    def fuel_factor(self) -> float:
        """
        Euro-class multiplier on the fuel this vehicle *burns*.

        This is the only place an emission class may act.  Aftertreatment and
        engine calibration change how much diesel is needed per unit of work;
        they cannot change how much carbon a litre of diesel contains, so the
        factor belongs on the fuel volume and never on
        :data:`DIESEL_KG_CO2_PER_L`.  Electric drivetrains carry their own
        efficiency chain (``engine_efficiency``, ``regen_fraction``) and burn no
        diesel at all, so the table's ``"ev"`` entry is never used as a scale.
        """
        if self.is_electric:
            return 1.0
        return EURO_FUEL_FACTOR.get(self.euro_class, 1.0)


DEFAULT_PROFILE = VehicleProfile()

# The powertrain chain the ``pto_fuel_l_per_lift`` default was measured behind: a
# diesel engine at its brake thermal efficiency driving the hydraulic pack.  Only
# this fraction of the litre ever reaches the ram as work, which is the quantity
# a non-diesel body has to be charged for.  See the compaction term below.
DIESEL_PTO_REFERENCE_ETA = (DEFAULT_PROFILE.engine_efficiency
                            * DEFAULT_PROFILE.driveline_efficiency)      # 0.36

# Euro-class multipliers on fuel consumption relative to the Euro IV reference.
# Applied to litres burned in :func:`leg_emissions`, via
# :meth:`VehicleProfile.fuel_factor` -- never to the CO2 factor.
EURO_FUEL_FACTOR = {
    "euro3": 1.08,
    "euro4": 1.00,
    "euro5": 0.96,
    "euro6": 0.93,
    "ev": 0.0,
}


PKE_FREEFLOW = 0.20        # m/s^2, positive kinetic energy on an uncongested arterial
PKE_SATURATED = 0.90       # m/s^2, saturated urban stop-and-go
STOPPED_FRACTION_FREE = 0.05
STOPPED_FRACTION_SATURATED = 0.50


def positive_kinetic_energy(friction: float) -> float:
    """
    Positive kinetic energy (PKE) in m/s^2 for a given saturation level.

    PKE = sum over accelerating segments of (v_f^2 - v_i^2) divided by trip
    distance.  It is the standard driving-cycle statistic for how much kinetic
    energy a vehicle must *repeatedly buy back* per unit distance, and published
    urban cycles put it near 0.2 m/s^2 in free flow and 0.6-0.9 m/s^2 in
    saturated conditions.  Using PKE instead of counting idealised full
    stop-and-go cycles is what keeps the model monotone: in deep congestion a
    truck performs many shallow accelerate-decelerate oscillations rather than a
    few full-speed cycles, and the energy per kilometre still rises.

    Energy per unit distance = 0.5 * mass * PKE.
    """
    x = max(0.0, min(1.0, float(friction)))
    return PKE_FREEFLOW + (PKE_SATURATED - PKE_FREEFLOW) * x


def stopped_fraction(friction: float) -> float:
    """Share of leg time spent stationary with the engine running."""
    x = max(0.0, min(1.0, float(friction)))
    return STOPPED_FRACTION_FREE + (STOPPED_FRACTION_SATURATED - STOPPED_FRACTION_FREE) * (x ** 1.5)


def decompose_speed(mean_speed_kmh: float, distance_km: float, friction: float,
                    freeflow_kmh: float = 34.0):
    """
    Split a leg's *mean* speed into stopped time and a true moving speed.

    This separation is the crux of modelling congestion honestly.  A vehicle in
    saturated traffic does not glide along at the mean speed: part of the time it
    is stationary in a queue and the rest of the time it moves faster than the
    mean.  Charging the aerodynamic and idle terms at the mean speed would make
    congestion look *cheaper* per kilometre, the opposite of what is measured.

    Returns ``(moving_speed_kmh, stopped_seconds, total_seconds)``.
    """
    distance_km = max(0.0, float(distance_km))
    v_mean = max(1.0, float(mean_speed_kmh))
    if distance_km <= 1e-9:
        return v_mean, 0.0, 0.0

    total_s = 3600.0 * distance_km / v_mean
    frac = stopped_fraction(friction)
    stopped_s = frac * total_s
    moving_s = max(total_s - stopped_s, 1e-6)
    v_move = min(3600.0 * distance_km / moving_s, 1.35 * float(freeflow_kmh))
    return max(v_move, v_mean), stopped_s, total_s


@dataclass
class LegEmissions:
    """Fuel/CO2 breakdown for a single leg (or an aggregate of legs)."""

    distance_km: float = 0.0
    duration_min: float = 0.0
    idle_min: float = 0.0
    fuel_cruise_l: float = 0.0
    fuel_stopgo_l: float = 0.0
    fuel_idle_l: float = 0.0
    fuel_pto_l: float = 0.0
    lifts: int = 0

    @property
    def fuel_total_l(self) -> float:
        return self.fuel_cruise_l + self.fuel_stopgo_l + self.fuel_idle_l + self.fuel_pto_l

    def co2_kg(self, profile: VehicleProfile = DEFAULT_PROFILE) -> float:
        if profile.is_electric:
            kwh = self.fuel_total_l * profile.kwh_per_l_equivalent
            return kwh * profile.grid_kg_co2_per_kwh
        # No Euro term here.  The class has already been charged to the fuel
        # volume in :func:`leg_emissions`; carbon per litre is a property of the
        # fuel and is identical for every engine that burns it.  Scaling it here
        # as well used to make the implied factor swing 2.49-2.89 kg/L across
        # Euro 3 to Euro 6, which is not a property any diesel has.
        return self.fuel_total_l * DIESEL_KG_CO2_PER_L

    def __add__(self, other: "LegEmissions") -> "LegEmissions":
        return LegEmissions(
            distance_km=self.distance_km + other.distance_km,
            duration_min=self.duration_min + other.duration_min,
            idle_min=self.idle_min + other.idle_min,
            fuel_cruise_l=self.fuel_cruise_l + other.fuel_cruise_l,
            fuel_stopgo_l=self.fuel_stopgo_l + other.fuel_stopgo_l,
            fuel_idle_l=self.fuel_idle_l + other.fuel_idle_l,
            fuel_pto_l=self.fuel_pto_l + other.fuel_pto_l,
            lifts=self.lifts + other.lifts,
        )

    def as_dict(self, profile: VehicleProfile = DEFAULT_PROFILE) -> Dict:
        total = self.fuel_total_l
        co2 = self.co2_kg(profile)
        share = (lambda part: round(part / total, 4) if total > 1e-12 else 0.0)
        return {
            "distance_km": round(self.distance_km, 4),
            "duration_min": round(self.duration_min, 3),
            "idle_min": round(self.idle_min, 3),
            "fuel_l": round(total, 4),
            "co2_kg": round(co2, 4),
            "co2_kg_per_km": round(co2 / self.distance_km, 4) if self.distance_km > 1e-9 else 0.0,
            "lifts": self.lifts,
            "breakdown_l": {
                "cruise": round(self.fuel_cruise_l, 4),
                "stop_go": round(self.fuel_stopgo_l, 4),
                "idle": round(self.fuel_idle_l, 4),
                "compaction": round(self.fuel_pto_l, 4),
            },
            "breakdown_share": {
                "cruise": share(self.fuel_cruise_l),
                "stop_go": share(self.fuel_stopgo_l),
                "idle": share(self.fuel_idle_l),
                "compaction": share(self.fuel_pto_l),
            },
        }


def leg_emissions(distance_m: float,
                  speed_kmh: float,
                  payload_kg: float,
                  friction: float = 0.0,
                  idle_minutes: float = 0.0,
                  lifts: int = 0,
                  lifted_kg: float = 0.0,
                  profile: VehicleProfile = DEFAULT_PROFILE,
                  freeflow_kmh: float = 34.0) -> LegEmissions:
    """
    Fuel/CO2 for one leg whose *mean* speed is ``speed_kmh``, carrying
    ``payload_kg``.

    ``idle_minutes`` is engine-on dwell at the destination bin (crew service
    time) on top of the queueing time implied by congestion;
    ``lifts``/``lifted_kg`` drive the compaction term.
    """
    distance_km = max(0.0, float(distance_m)) / 1000.0
    mass = profile.kerb_mass_kg + max(0.0, float(payload_kg))
    friction = max(0.0, min(1.0, float(friction)))

    v_move_kmh, stopped_s, total_s = decompose_speed(
        speed_kmh, distance_km, friction, freeflow_kmh
    )
    v = max(0.5, v_move_kmh) / 3.6                             # m/s while moving

    # The low-speed derate is a duty-cycle statistic, so it is indexed on the
    # leg's *mean* speed (clamped exactly as ``decompose_speed`` clamps it), not
    # on ``v_move_kmh``.  Charged at the moving speed it could never fire:
    # ``v_move`` is the mean divided by the rolling-time share, so at full
    # saturation it bottoms out at 10 / (1 - 0.50) = 20 km/h and stays above the
    # 18 km/h reference for every friction the traffic surface can produce.
    v_mean_kmh = max(1.0, float(speed_kmh))
    eta = max(0.05, profile.powertrain_efficiency(v_mean_kmh))
    fuel_energy_j = eta * DIESEL_LHV_MJ_PER_L * 1e6            # J per litre delivered
    moving_s = max(total_s - stopped_s, 0.0)

    # --- 1. cruise -----------------------------------------------------
    rolling = profile.rolling_resistance * mass * G
    aero = 0.5 * RHO_AIR * profile.drag_coefficient * profile.frontal_area_m2 * v * v
    power_w = (rolling + aero) * v
    fuel_cruise = (power_w * moving_s) / fuel_energy_j

    # --- 2. stop-and-go ------------------------------------------------
    # Kinetic energy repeatedly bought back over the leg, from the driving-cycle
    # PKE statistic.  Regenerative braking recovers part of it on hybrids/EVs.
    pke = positive_kinetic_energy(friction)
    energy_accel_j = 0.5 * mass * pke * (distance_km * 1000.0) * (1.0 - profile.regen_fraction)
    fuel_stopgo = energy_accel_j / fuel_energy_j

    # --- 3. idle -------------------------------------------------------
    # Queueing time from congestion plus the crew's service dwell.
    congestion_idle_min = stopped_s / 60.0
    idle_min_total = max(0.0, float(idle_minutes)) + congestion_idle_min
    fuel_idle = profile.idle_fuel_l_per_h * (idle_min_total / 60.0)

    # --- 4. compaction -------------------------------------------------
    # A fixed cost per lift plus a mass-proportional packing term.
    fuel_pto = lifts * profile.pto_fuel_l_per_lift
    fuel_pto += (max(0.0, lifted_kg) / 1000.0) * profile.pto_fuel_l_per_lift * 0.9
    if profile.is_electric:
        # ``pto_fuel_l_per_lift`` is a measured *diesel* volume, so it carries
        # the whole litre's energy content.  Every other term already expresses
        # its litres as "energy / (eta * LHV)", which is why the cruise term
        # converts correctly for an EV and this one did not: an electric body was
        # charged the full 0.045 L of chemical energy (0.447 kWh per lift) rather
        # than the work that litre actually delivers.  Published electric refuse
        # bodies draw 0.10-0.12 kWh per container, so the raw volume overstated
        # the compaction term by ~2.8x on the work, ~2.1x once the electric
        # chain's own losses are allowed for.  Convert to useful work at the
        # diesel reference chain, then back at this drivetrain's efficiency.
        eta_electric = max(1e-6, profile.engine_efficiency * profile.driveline_efficiency)
        fuel_pto *= DIESEL_PTO_REFERENCE_ETA / eta_electric

    # --- 5. emission-class scaling -------------------------------------
    # A Euro class changes fuel burned per unit of work, so it scales every
    # engine-fed term here and nothing in the CO2 accounting downstream.
    euro = profile.fuel_factor()

    return LegEmissions(
        distance_km=distance_km,
        duration_min=total_s / 60.0 + max(0.0, float(idle_minutes)),
        idle_min=idle_min_total,
        fuel_cruise_l=fuel_cruise * euro,
        fuel_stopgo_l=fuel_stopgo * euro,
        fuel_idle_l=fuel_idle * euro,
        fuel_pto_l=fuel_pto * euro,
        lifts=lifts,
    )


def flat_factor_equivalent(distance_km: float, co2_kg: float) -> float:
    """kg CO2 per km actually realised -- the number the old flat model assumed."""
    if distance_km <= 1e-9:
        return 0.0
    return co2_kg / distance_km


def profile_from_vehicle(capacity_kg: float, kerb_mass_kg: float, euro_class: str,
                         name: str = "vehicle") -> VehicleProfile:
    """Build a profile from the Django ``Vehicle`` record's fields."""
    is_ev = str(euro_class).lower() == "ev"
    return VehicleProfile(
        name=name,
        kerb_mass_kg=float(kerb_mass_kg),
        capacity_kg=float(capacity_kg),
        euro_class=str(euro_class).lower(),
        is_electric=is_ev,
        regen_fraction=0.55 if is_ev else 0.0,
        engine_efficiency=0.85 if is_ev else 0.40,
        idle_fuel_l_per_h=0.25 if is_ev else 2.6,
    )


def describe_model(profile: VehicleProfile = DEFAULT_PROFILE) -> Dict:
    """Machine-readable description of the model, for the audit ledger and paper."""
    return {
        "model": "modal physics-based (cruise / stop-go / idle / compaction)",
        "co2_per_litre_diesel_kg": round(DIESEL_KG_CO2_PER_L, 4),
        "co2_per_mj_diesel_kg": DIESEL_KG_CO2_PER_MJ,
        "diesel_lhv_mj_per_l": DIESEL_LHV_MJ_PER_L,
        "euro_factors": EURO_FUEL_FACTOR,
        "euro_factor_applies_to": "fuel_litres",
        "profile": asdict(profile),
        "references": [
            "IPCC 2006 Guidelines vol. 2 ch. 3 (mobile combustion emission factors)",
            "Bureau of Public Roads (1964) volume-delay function for speed",
            "Barth & Boriboonsomsin (2008) real-world CO2 vs speed for road vehicles",
            "Nguyen et al. refuse-collection duty-cycle fuel measurements",
        ],
    }


def sanity_reference_factor(payload_fraction: float = 0.5,
                            speed_kmh: float = 20.0,
                            friction: float = 0.35,
                            bins_per_km: float = 4.0,
                            service_min_per_bin: float = 4.0,
                            profile: VehicleProfile = DEFAULT_PROFILE) -> float:
    """
    Realised kg CO2/km for a representative collection kilometre.

    Used in tests to pin the model inside the published duty-cycle range for
    refuse collection rounds.  Rear-loading collection vehicles are commonly
    reported at 1.5-3 mpg (roughly 78-157 L per 100 km, i.e. 2.1-4.2 kg CO2 per
    route kilometre); line-haul legs between the route and the depot fall to
    around 0.6-1.0 kg CO2/km.  A single flat factor cannot span both.
    """
    payload = profile.capacity_kg * payload_fraction
    e = leg_emissions(
        distance_m=1000.0, speed_kmh=speed_kmh, payload_kg=payload,
        friction=friction,
        idle_minutes=service_min_per_bin * bins_per_km,
        lifts=int(bins_per_km),
        lifted_kg=bins_per_km * 150.0,
        profile=profile,
    )
    return flat_factor_equivalent(e.distance_km, e.co2_kg(profile))
