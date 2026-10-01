"""
Tests for the framework-independent core, one class per module.

These subsystems had no tests at all, and two of them underwrite claims the
manuscript makes: the ledger is the tamper-evidence argument, and the continual
learner is the argument that adapting online never costs the operator latency.
Both had defects found by hand that nothing would have caught again.

No database is used. Every case is either a known answer computed by hand or a
property that must hold for any correct implementation.
"""
from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timedelta, timezone
from unittest import mock

import numpy as np
from django.test import SimpleTestCase

from wastebins_core import aging as AG
from wastebins_core import continual as CL
from wastebins_core import emissions as EM
from wastebins_core import faults as F
from wastebins_core import features as FEAT
from wastebins_core import geo as GEO
from wastebins_core import ledger as LG
from wastebins_core import roadnet as RN
from wastebins_core import scenario as SC
from wastebins_core import stats as ST
from wastebins_core import traffic as TR
from wastebins_core import vrp as VRP
from wastebins_core import weather as WX


class LedgerTests(SimpleTestCase):
    """
    Hash chain, Merkle roots and inclusion proofs.

    The odd-leaf-count case is tested exhaustively because it was wrong once:
    the index tracking through the promotion step lost a bit, so proofs for
    trees with an odd number of leaves failed to verify.
    """

    def _leaves(self, n):
        return [LG.sha256_hex(f"leaf-{i}") for i in range(n)]

    def test_canonical_json_is_stable_under_key_order(self):
        a = {"b": 1, "a": {"d": 2, "c": [3, 4]}}
        b = {"a": {"c": [3, 4], "d": 2}, "b": 1}
        self.assertEqual(LG.canonical_json(a), LG.canonical_json(b))
        self.assertEqual(LG.payload_digest(a), LG.payload_digest(b))

    def test_payload_digest_changes_with_content(self):
        self.assertNotEqual(LG.payload_digest({"x": 1}), LG.payload_digest({"x": 2}))

    def test_merkle_proof_verifies_for_every_leaf_count(self):
        """Exhaustive over sizes 1 to 33, which covers every odd/even path."""
        for n in range(1, 34):
            leaves = self._leaves(n)
            root = LG.merkle_root(leaves)
            for index in range(n):
                proof = LG.merkle_proof(leaves, index)
                self.assertTrue(
                    LG.verify_merkle_proof(leaves[index], proof, root),
                    f"valid proof rejected for leaf {index} of {n}")

    def test_merkle_proof_rejects_a_forged_leaf(self):
        for n in (1, 2, 3, 7, 8, 16, 17):
            leaves = self._leaves(n)
            root = LG.merkle_root(leaves)
            proof = LG.merkle_proof(leaves, 0)
            forged = LG.sha256_hex("not-the-leaf")
            self.assertFalse(LG.verify_merkle_proof(forged, proof, root),
                             f"forged leaf accepted for a tree of {n}")

    def test_chain_detects_a_modified_payload(self):
        entries, prev = [], LG.GENESIS_HASH
        for i in range(6):
            entry = LG.build_entry(i, "READING", {"value": i}, prev)
            entries.append(entry)
            prev = entry.entry_hash
        self.assertTrue(LG.verify_chain(entries).valid)

        # Rewrite history: change one payload and keep every stored hash.
        import dataclasses
        tampered = list(entries)
        tampered[3] = dataclasses.replace(entries[3], payload={"value": 999})
        report = LG.verify_chain(tampered)
        self.assertFalse(report.valid)
        self.assertEqual(report.first_invalid_sequence, 3)

    def test_chain_detects_a_reordered_entry(self):
        entries, prev = [], LG.GENESIS_HASH
        for i in range(5):
            entry = LG.build_entry(i, "PLAN", {"i": i}, prev)
            entries.append(entry)
            prev = entry.entry_hash
        swapped = entries[:2] + [entries[3], entries[2]] + entries[4:]
        self.assertFalse(LG.verify_chain(swapped).valid)

    def test_entry_encoding_is_injective(self):
        """
        No field value may shift a field boundary.

        The encoding was a "|".join, and its docstring claimed no combination of
        values could be re-partitioned. It could: moving the separator between
        two adjacent fields produced the same digest for a different record, so
        an entry could be rewritten to carry a different payload hash and still
        verify. Length-prefixing removes the class of attack, not just this case.
        """
        collide_a = LG.entry_digest(1, "P", "A|B", "C", "D")
        collide_b = LG.entry_digest(1, "P", "A", "B|C", "D")
        self.assertNotEqual(collide_a, collide_b)

        # A separator anywhere must not merge or split fields.
        for evil in ("|", "a|b", "|||", "5:x", "3:abc"):
            first = LG.entry_digest(1, "prev", evil, "digest", "ts")
            second = LG.entry_digest(1, "prev", "", evil + "digest", "ts")
            self.assertNotEqual(first, second, f"collision via {evil!r}")

    def test_actor_is_authenticated(self):
        """
        An audit trail whose "who" is unsigned records nothing worth auditing.

        The actor was stored on the entry but left out of the digest, so two
        entries differing only in actor hashed identically and rewriting the
        actor on a stored entry passed every check.
        """
        import dataclasses
        alice = LG.build_entry(1, "PLAN", {"x": 1}, LG.GENESIS_HASH,
                               timestamp="2026-01-01T00:00:00", actor="alice")
        bob = LG.build_entry(1, "PLAN", {"x": 1}, LG.GENESIS_HASH,
                             timestamp="2026-01-01T00:00:00", actor="bob")
        self.assertNotEqual(alice.entry_hash, bob.entry_hash)

        forged = dataclasses.replace(alice, actor="mallory")
        self.assertFalse(LG.verify_chain([forged]).valid,
                         "a rewritten actor still verified")

    def test_signature_round_trip_and_rejection(self):
        digest = LG.sha256_hex("entry")
        signature = LG.sign(digest, "secret")
        self.assertTrue(LG.verify_signature(digest, signature, "secret"))
        self.assertFalse(LG.verify_signature(digest, signature, "wrong-key"))
        self.assertFalse(LG.verify_signature(LG.sha256_hex("other"),
                                             signature, "secret"))


class FaultTaxonomyTests(SimpleTestCase):
    """Each mode must produce the signature it claims, and label it correctly."""

    def _clean(self, n=120):
        rng = np.random.default_rng(0)
        return 0.4 + 0.002 * np.arange(n) + rng.normal(0, 0.01, n)

    def test_every_mode_is_documented(self):
        described = F.describe_taxonomy()
        for mode in F.FAULT_MODES:
            self.assertIn(mode, described)
            self.assertTrue(described[mode].strip())

    def test_dropout_produces_missing_not_zero(self):
        """A lost packet is absent. Recording it as 0.0 is a different fault."""
        series = self._clean()
        spec = F.FaultSpec(mode="dropout", rate=0.9, start_frac=0.5,
                           duration_frac=0.4, seed=1)
        out, mask = F.inject(series, spec)
        self.assertTrue(np.isnan(out[mask]).any())
        self.assertFalse(np.any(out[mask] == 0.0))

    def test_zero_stuck_produces_zero_not_missing(self):
        series = self._clean()
        spec = F.FaultSpec(mode="zero_stuck", rate=1.0, start_frac=0.5,
                           duration_frac=0.4, seed=1)
        out, mask = F.inject(series, spec)
        self.assertTrue(np.any(out[mask] == 0.0))
        self.assertFalse(np.isnan(out[mask]).all())

    def test_drift_is_monotone_in_magnitude(self):
        """A larger drift must move the series further, in the same direction."""
        series = self._clean()
        deltas = []
        for magnitude in (0.05, 0.2, 0.8):
            spec = F.FaultSpec(mode="drift", rate=1.0, magnitude=magnitude,
                               start_frac=0.4, duration_frac=0.6, seed=7)
            out, mask = F.inject(series, spec)
            deltas.append(abs(float(np.mean(out[mask] - series[mask]))))
        self.assertLess(deltas[0], deltas[1])
        self.assertLess(deltas[1], deltas[2])

    def test_stuck_at_holds_one_value(self):
        series = self._clean()
        spec = F.FaultSpec(mode="stuck_at", rate=1.0, start_frac=0.5,
                           duration_frac=0.4, seed=3)
        out, mask = F.inject(series, spec)
        affected = out[mask]
        self.assertLessEqual(float(np.nanstd(affected)), 1e-9)

    def test_mask_marks_only_changed_samples_for_deterministic_modes(self):
        """
        Ground truth has to be trustworthy: recall and precision are measured
        against this mask, so a mask that over-claims would inflate recall.
        """
        series = self._clean()
        for mode in ("zero_stuck", "stuck_at", "drift", "calibration"):
            spec = F.FaultSpec(mode=mode, rate=1.0, magnitude=0.3,
                               start_frac=0.5, duration_frac=0.3, seed=5)
            out, mask = F.inject(series, spec)
            unaffected = ~mask
            np.testing.assert_allclose(out[unaffected], series[unaffected],
                                       err_msg=f"{mode} changed samples outside its mask")

    def test_canonical_channel_accepts_both_vocabularies(self):
        """Django field names and core short names must resolve identically."""
        self.assertEqual(F.canonical_channel("gas_level"), F.canonical_channel("gas"))
        self.assertEqual(F.canonical_channel("temperature"), F.canonical_channel("temp"))
        self.assertEqual(F.canonical_channel("waste_level"), F.canonical_channel("waste"))


class EmissionsTests(SimpleTestCase):
    """
    Fuel and CO2 must rise with congestion, load and idling.

    Monotonicity in congestion is tested because it failed once: charging the
    aerodynamic and kinetic terms at the mean speed made a heavily congested leg
    look cleaner than a moderately congested one.
    """

    def test_intensity_is_monotone_in_congestion(self):
        previous = -1.0
        for friction in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            leg = EM.leg_emissions(distance_m=2000.0, speed_kmh=30.0,
                                   payload_kg=1500.0, friction=friction)
            intensity = leg.co2_kg() / 2.0          # per km
            self.assertGreater(intensity, previous,
                               f"intensity fell at friction {friction}")
            previous = intensity

    def test_heavier_payload_burns_more(self):
        light = EM.leg_emissions(distance_m=3000.0, speed_kmh=25.0, payload_kg=0.0)
        heavy = EM.leg_emissions(distance_m=3000.0, speed_kmh=25.0, payload_kg=5000.0)
        self.assertGreater(heavy.co2_kg(), light.co2_kg())

    def test_idling_emits_without_moving(self):
        """A stationary truck still burns fuel, which a per-km factor cannot express."""
        parked = EM.leg_emissions(distance_m=0.0, speed_kmh=0.0, payload_kg=1000.0,
                                  idle_minutes=30.0)
        self.assertGreater(parked.co2_kg(), 0.0)

    def test_compaction_lifts_cost_fuel(self):
        without = EM.leg_emissions(distance_m=500.0, speed_kmh=20.0, payload_kg=800.0)
        with_lifts = EM.leg_emissions(distance_m=500.0, speed_kmh=20.0,
                                      payload_kg=800.0, lifts=6, lifted_kg=600.0)
        self.assertGreater(with_lifts.co2_kg(), without.co2_kg())

    def test_reference_factor_sits_in_the_published_band(self):
        """
        `sanity_reference_factor` returns kilograms of CO2 per route kilometre,
        not miles per gallon. Published in-use measurements of rear-loading
        refuse vehicles give 1.5 to 3 miles per gallon on collection rounds,
        which is 2.1 to 4.2 kg CO2 per kilometre. A model outside that band is
        wrong however carefully it was derived, so this pins it against reported
        measurements rather than against itself.
        """
        collection = EM.sanity_reference_factor()
        self.assertGreater(collection, 2.1)
        self.assertLess(collection, 4.2)

    def test_duty_cycle_ordering_is_physical(self):
        """
        Line-haul must be the cheapest kilometre and congested collection the
        dearest. The spread between them is the argument against a single
        emission factor, so its direction has to be right.
        """
        line_haul = EM.sanity_reference_factor(
            bins_per_km=0.0, service_min_per_bin=0.0, friction=0.1, speed_kmh=45.0)
        collection = EM.sanity_reference_factor()
        congested = EM.sanity_reference_factor(
            bins_per_km=12.0, service_min_per_bin=5.0, friction=0.9,
            payload_fraction=0.8)
        self.assertLess(line_haul, collection)
        self.assertLess(collection, congested)
        self.assertGreater(congested / line_haul, 3.0,
                           "the duty-cycle spread is the whole point")


class ContinualLearningTests(SimpleTestCase):
    """
    The corrector must help under drift, do nothing without it, and stay bounded.

    "Does nothing without drift" is the property such systems most often fail,
    and a continual learner that degrades a healthy model is worse than none.
    """

    def _stream(self, n, shift_at=None, shift=0.0, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, 8))
        y = 2.0 * X[:, 0] - X[:, 3] + rng.normal(0, 0.2, n)
        base = 2.0 * X[:, 0] - X[:, 3]              # a good frozen model
        if shift_at is not None:
            y = y + np.where(np.arange(n) >= shift_at, shift, 0.0)
        return X, y, base

    def test_dormant_on_a_stationary_stream(self):
        X, y, base = self._stream(600, seed=1)
        learner = CL.ContinualResidualLearner(8, warmup_samples=50, seed=1)
        report = CL.prequential_evaluation(learner, X, y, base, report_every=0)
        self.assertLess(abs(report["improvement_pct"]), 1.0,
                        "corrector interfered with a healthy model")

    def test_recovers_after_an_abrupt_shift(self):
        X, y, base = self._stream(1200, shift_at=600, shift=5.0, seed=2)
        learner = CL.ContinualResidualLearner(8, warmup_samples=50, seed=2)
        report = CL.prequential_evaluation(learner, X, y, base, report_every=0)
        self.assertGreater(report["improvement_pct"], 20.0)

    def test_update_respects_its_budget(self):
        X, y, base = self._stream(400, shift_at=100, shift=4.0, seed=3)
        learner = CL.ContinualResidualLearner(8, warmup_samples=20, seed=3)
        batch = [(X[i], float(y[i]), float(base[i])) for i in range(len(X))]
        learner.last_update_ts = 0.0                 # bypass the rate limit
        report = learner.update(batch)
        self.assertLessEqual(report.n_samples, learner.budget.max_samples_per_update)

    def test_page_hinkley_ignores_scale(self):
        """
        The detector is configured in standard deviations, not absolute units, so
        the same relative shift must be detected whatever the signal's scale.
        """
        for scale in (0.01, 1.0, 100.0):
            detector = CL.PageHinkley(warmup=30)
            rng = np.random.default_rng(4)
            fired = False
            for i in range(400):
                value = rng.normal(0, scale) + (6 * scale if i > 200 else 0.0)
                if detector.update(value):
                    fired = True
            self.assertTrue(fired, f"no detection at scale {scale}")

    def test_reservoir_buffer_stays_within_capacity(self):
        buffer = CL.ReservoirBuffer(capacity=50, seed=0)
        for i in range(1000):
            buffer.add(np.zeros(3), float(i))
        self.assertLessEqual(len(buffer), 50)


class StatsTests(SimpleTestCase):
    """Known-answer checks on the statistics behind every reported comparison."""

    def test_bootstrap_interval_brackets_a_known_mean(self):
        rng = np.random.default_rng(0)
        sample = rng.normal(10.0, 1.0, 400)
        result = ST.bootstrap_ci(sample, n_resamples=2000, seed=0)
        self.assertLess(result["ci_low"], 10.0)
        self.assertGreater(result["ci_high"], 10.0)
        self.assertLess(result["ci_low"], result["ci_high"])

    def test_cliffs_delta_signs_and_bounds(self):
        a = list(range(20))
        b = [x + 100 for x in a]
        self.assertAlmostEqual(ST.cliffs_delta(b, a)["delta"], 1.0, places=6)
        self.assertAlmostEqual(ST.cliffs_delta(a, b)["delta"], -1.0, places=6)
        self.assertAlmostEqual(ST.cliffs_delta(a, a)["delta"], 0.0, places=6)

    def test_identical_samples_are_not_significant(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = ST.paired_test(values, values)
        self.assertGreater(result["p_value"], 0.05)

    def test_clearly_different_samples_are_significant(self):
        a = [10.0, 11.0, 9.5, 10.5, 10.2, 9.8, 10.1, 10.3]
        b = [x + 5.0 for x in a]
        result = ST.paired_test(a, b)
        self.assertLess(result["p_value"], 0.05)

    def test_holm_never_lowers_a_p_value(self):
        """Correction for multiplicity can only make a claim harder to make."""
        raw = {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5}
        corrected = ST.holm_bonferroni(raw)
        for name, entry in corrected.items():
            self.assertGreaterEqual(entry["p_adjusted"], raw[name] - 1e-12)
            self.assertLessEqual(entry["p_adjusted"], 1.0 + 1e-12)

    def test_normal_quantile_matches_known_values(self):
        self.assertAlmostEqual(ST._norm_ppf(0.5), 0.0, places=4)
        self.assertAlmostEqual(ST._norm_ppf(0.975), 1.959964, places=3)
        self.assertAlmostEqual(ST._norm_ppf(0.025), -1.959964, places=3)


class ForwardLabelTests(SimpleTestCase):
    """
    The forward labels must not manufacture the evidence they report.

    A row near the end of a record has less future to be judged against than the
    censoring cap, and the labels used to hand it the cap anyway, which reads as
    "observed not to overflow for a full day" on a bin nobody watched for a full
    day. These cases pin the distinction, and one of them pins the fact that the
    trimmed training set is unaffected, because the published figures were
    computed on it.
    """

    CAP = FEAT.TTO_CAP_H
    HORIZON = FEAT.HORIZON_H

    def _flat_record(self, span_h, fill=0.30, gas=0.10):
        """A record of `span_h` hours in which nothing ever happens."""
        hours = np.arange(0.0, float(span_h) + 1e-9, 1.0)
        return (np.full(hours.size, float(fill)),
                np.full(hours.size, float(gas)), hours)

    def test_a_truncated_row_is_censored_at_its_own_follow_up(self):
        waste, gas, hours = self._flat_record(29)
        _, tto, observed = FEAT.forward_labels(waste, gas, hours)

        for i, hour in enumerate(hours):
            followup = hours[-1] - hour
            self.assertEqual(observed[i], 0, f"row {i} claims an overflow it never saw")
            self.assertAlmostEqual(
                tto[i], min(self.CAP, followup), places=9,
                msg=f"row {i} was watched for {followup:g} h but is labelled {tto[i]:g} h")

    def test_a_short_record_never_claims_a_full_day_of_safety(self):
        """A record shorter than the cap can support no censoring time at the cap."""
        waste, gas, hours = self._flat_record(8)
        _, tto, observed = FEAT.forward_labels(waste, gas, hours)
        self.assertEqual(int(observed.sum()), 0)
        self.assertLess(float(tto.max()), self.CAP,
                        "an 8 h record produced a 24 h censoring time")
        self.assertAlmostEqual(float(tto.max()), 8.0, places=9)

    def test_an_observed_overflow_is_flagged_and_timed(self):
        """A real crossing keeps its exact delay and is marked as an event."""
        waste, gas, hours = self._flat_record(29)
        waste[28] = FEAT.OVERFLOW_LEVEL
        _, tto, observed = FEAT.forward_labels(waste, gas, hours)

        self.assertEqual(observed[26], 1)
        self.assertAlmostEqual(tto[26], 2.0, places=9)
        self.assertEqual(observed[28], 1)
        self.assertAlmostEqual(tto[28], 0.0, places=9)
        # A row before the crossing but more than the cap away from it is still
        # censored at the cap, not credited with the distant overflow.
        self.assertEqual(observed[0], 0)
        self.assertAlmostEqual(tto[0], self.CAP, places=9)

    def test_a_full_horizon_row_keeps_the_cap_encoded_convention(self):
        """
        The guard on the published numbers.

        Everything downstream identifies a censored row by its label sitting at
        the cap, and the trainer keeps only rows with a full label horizon. On
        exactly those rows the returned flag and the cap test must agree, and the
        labels must be what they always were, or the reported censored share,
        R^2, MAE and concordance index would move.
        """
        rng = np.random.default_rng(0)
        label_horizon = max(self.HORIZON, self.CAP)
        for trial in range(25):
            n = int(rng.integers(80, 200))
            # An irregular clock, because real telemetry is not on the hour.
            hours = np.concatenate([[0.0], np.cumsum(rng.uniform(0.4, 1.8, n))[:-1]])
            fill, waste = float(rng.uniform(0.0, 0.4)), []
            for _ in range(n):
                fill += float(rng.uniform(0.0, 0.09))
                waste.append(min(1.0, fill))
                if fill >= 1.0:
                    fill = float(rng.uniform(0.0, 0.1))
            waste = np.asarray(waste)
            gas = np.clip(0.7 * waste + rng.normal(0.0, 0.05, n), 0.0, 1.0)

            _, tto, observed = FEAT.forward_labels(waste, gas, hours)
            keep = hours <= hours[-1] - label_horizon
            if not keep.any():
                continue

            cap_says_censored = tto[keep] >= self.CAP - 1e-9
            self.assertTrue(
                np.array_equal(cap_says_censored, observed[keep] == 0),
                f"trial {trial}: the cap test and the censoring flag disagree")
            # Every kept censored row sits on the cap exactly, so no clock
            # arithmetic can leave one a hair short and have it read as an event.
            self.assertTrue(
                np.all(tto[keep][cap_says_censored] == self.CAP),
                f"trial {trial}: a kept censored row is not exactly at the cap")

    def test_a_truncated_hazard_label_is_identifiable(self):
        """
        The hazard label is truncated too, and the docstring promises a caller
        can spot it from the three returned arrays alone.
        """
        waste, gas, hours = self._flat_record(29)
        hazard, tto, observed = FEAT.forward_labels(waste, gas, hours)
        truncated = (hazard == 0) & (observed == 0) & (tto < self.HORIZON)
        expected = (hours[-1] - hours) < self.HORIZON
        self.assertTrue(np.array_equal(truncated, expected))

    def test_an_empty_series_returns_three_empty_arrays(self):
        hazard, tto, observed = FEAT.forward_labels([], [], [])
        self.assertEqual(hazard.size, 0)
        self.assertEqual(tto.size, 0)
        self.assertEqual(observed.size, 0)


class _FakeHTTPResponse:
    """Minimal stand-in for the object `urlopen` returns."""

    def __init__(self, payload: str):
        self._payload = payload.encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class LiveTrafficTimeTests(SimpleTestCase):
    """
    A route planned for a departure time must not be costed at present conditions.

    The synthetic surface is a function of the clock and always was. The live
    adapter discarded the timestamp entirely: it fetched the same URL and
    returned the same friction whether asked about 03:00 or the morning peak. It
    cannot forecast, because the feed it speaks to publishes only the present, so
    the fix is to say so rather than to invent one.
    """

    LAT, LNG = 23.8069, 90.3687
    NOW = datetime(2026, 3, 10, 3, 0, tzinfo=timezone.utc)
    NOWCAST_URL = "https://example.test/flow?p={lat},{lng}&key={key}"
    FORECAST_URL = "https://example.test/flow?p={lat},{lng}&at={time}&key={key}"

    def setUp(self):
        self.synthetic = TR.SyntheticTrafficProvider(seed=42)
        self.urls = []

    def _urlopen(self, request, timeout=None):
        self.urls.append(request.full_url)
        return _FakeHTTPResponse(json.dumps(
            {"flowSegmentData": {"currentSpeed": 12.0, "freeFlowSpeed": 34.0}}))

    def _provider(self, url, **kwargs):
        kwargs.setdefault("cache_seconds", 0.0)
        return TR.LiveTrafficProvider(url, api_key="SECRET", fallback=self.synthetic,
                                      clock=lambda: self.NOW, **kwargs)

    def test_the_synthetic_surface_depends_on_the_requested_time(self):
        """
        Guard the guard. If congestion did not vary with the clock, every case
        below would pass without testing anything.
        """
        night = self.synthetic.friction(self.LAT, self.LNG, self.NOW)
        peak = self.synthetic.friction(self.LAT, self.LNG,
                                       self.NOW + timedelta(hours=6))
        self.assertGreater(abs(peak - night), 0.2,
                           "the synthetic surface is flat over the day")

    def test_a_nowcast_still_answers_for_the_present(self):
        with mock.patch("urllib.request.urlopen", self._urlopen):
            provider = self._provider(self.NOWCAST_URL)
            value = provider.friction(self.LAT, self.LNG, self.NOW)
        self.assertAlmostEqual(value, TR.speed_to_friction(12.0, 34.0), places=9)
        self.assertEqual(len(self.urls), 1)
        self.assertEqual(provider.out_of_window_count, 0)

    def test_a_nowcast_is_not_reused_for_a_future_departure(self):
        later = self.NOW + timedelta(hours=6)
        with mock.patch("urllib.request.urlopen", self._urlopen):
            provider = self._provider(self.NOWCAST_URL)
            with self.assertWarns(RuntimeWarning):
                future = provider.friction(self.LAT, self.LNG, later)

        # The feed was never asked, and the answer is the surface for that hour.
        self.assertEqual(self.urls, [])
        self.assertAlmostEqual(
            future, self.synthetic.friction(self.LAT, self.LNG, later), places=9)
        self.assertEqual(provider.out_of_window_count, 1)

        described = provider.describe()
        self.assertFalse(described["honours_requested_time"])
        self.assertEqual(described["out_of_window_fallbacks"], 1)
        self.assertIn("nowcast", described["time_handling"])

    def test_a_nowcast_route_cost_now_varies_with_departure_time(self):
        """The defect as the planner saw it: identical minutes at every hour."""
        coords = [(23.7790, 90.3660), (23.8203, 90.3650)]
        distance = np.array([[0.0, 4800.0], [4800.0, 0.0]])
        minutes = []
        with mock.patch("urllib.request.urlopen", self._urlopen), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            provider = self._provider(self.NOWCAST_URL)
            for offset in (0, 6, 12):
                context = TR.build_travel_context(
                    coords, distance, provider, self.NOW + timedelta(hours=offset))
                minutes.append(context.travel_minutes(0, 1))
        self.assertEqual(len(set(round(m, 6) for m in minutes)), 3,
                         f"departure time made no difference: {minutes}")

    def test_a_departure_time_placeholder_reaches_the_api(self):
        with mock.patch("urllib.request.urlopen", self._urlopen):
            provider = self._provider(self.FORECAST_URL, cache_seconds=600.0)
            self.assertTrue(provider.supports_departure_time)
            for offset in (0, 6, 12):
                provider.friction(self.LAT, self.LNG,
                                  self.NOW + timedelta(hours=offset))

        self.assertEqual(len(self.urls), 3, "requests were collapsed by the cache")
        for offset, url in zip((3, 9, 15), self.urls):
            self.assertIn(f"at=2026-03-10T{offset:02d}:00:00Z", url)
        self.assertEqual(provider.out_of_window_count, 0)
        self.assertTrue(provider.describe()["honours_requested_time"])

    def test_a_forecast_is_cached_per_requested_instant(self):
        """Two instants must not share a cache entry; the same instant must."""
        with mock.patch("urllib.request.urlopen", self._urlopen):
            provider = self._provider(self.FORECAST_URL, cache_seconds=600.0)
            provider.friction(self.LAT, self.LNG, self.NOW)
            provider.friction(self.LAT, self.LNG, self.NOW)
            provider.friction(self.LAT, self.LNG, self.NOW + timedelta(hours=6))
        self.assertEqual(len(self.urls), 2)
        self.assertEqual(provider.hit_count, 1)

    def test_an_unreachable_feed_still_falls_back(self):
        """The pre-existing contract: a dead upstream must not cost the planner."""
        def broken(request, timeout=None):
            raise OSError("connection refused")

        with mock.patch("urllib.request.urlopen", broken):
            provider = self._provider(self.NOWCAST_URL)
            value = provider.friction(self.LAT, self.LNG, self.NOW)
        self.assertAlmostEqual(
            value, self.synthetic.friction(self.LAT, self.LNG, self.NOW), places=9)
        self.assertEqual(provider.fallback_count, 1)
        self.assertIn("OSError", provider.last_error or "")


class EquityGuaranteeTests(SimpleTestCase):
    """
    The overdue tier has to survive the pricing channel, not just the ranking one.

    An earlier version ordered overdue bins correctly and then priced them off
    that same bounded ordering score, which capped the cost of skipping one at
    ``lambda_prize * overdue_multiplier``.  A bin whose detour cost more than the
    cap was abandoned at every wait, so the wait bound was an ordering claim with
    no consequence for what was actually collected.  These cases pin both halves.
    """

    WEIGHTS = VRP.ObjectiveWeights()

    def _task(self, wait_h, node_id=1):
        """Build one bin through the production path, not by hand."""
        tiered = AG.effective_priorities({node_id: 0.30}, {node_id: wait_h},
                                         {node_id: 0.0})
        return SC.make_tasks(
            [node_id], {node_id: 0.5}, {node_id: tiered[node_id].score},
            index_of={node_id: 1}, hazards={node_id: False},
            tiers={node_id: tiered[node_id].tier},
            overdue_pressures={node_id: tiered[node_id].pressure},
            densities={node_id: 220.0}, capacities_l={node_id: 1100.0},
        )[0]

    def _travel(self, km):
        return VRP.TravelModel(np.array([[0.0, km * 1000.0], [km * 1000.0, 0.0]]),
                               default_speed_kmh=20.0)

    def test_skip_penalty_grows_without_bound(self):
        """The defect in one line: the penalty must not have a supremum."""
        cap = self.WEIGHTS.lambda_prize * self.WEIGHTS.overdue_multiplier
        far = VRP.skip_cost(self._task(10 ** 6), self.WEIGHTS)
        self.assertGreater(far, 100 * cap)
        # And it is monotone in the wait, so waiting is never rewarded.
        costs = [VRP.skip_cost(self._task(w), self.WEIGHTS)
                 for w in (48, 96, 240, 960)]
        self.assertEqual(costs, sorted(costs))

    def test_penalty_matches_the_old_one_at_promotion(self):
        """
        Continuity at tau, so the existing weight tuning still means what it did.

        At w = tau the pressure w/(2 tau) and the ordering score w/(tau + w) are
        both exactly 0.5, so the escalation changes nothing at the threshold and
        only ever raises the penalty past it.
        """
        at_tau = VRP.skip_cost(self._task(AG.DEFAULT_TAU_H), self.WEIGHTS)
        self.assertAlmostEqual(
            at_tau,
            0.5 * self.WEIGHTS.lambda_prize * self.WEIGHTS.overdue_multiplier,
            places=9)

    def test_a_remote_overdue_bin_is_eventually_served(self):
        """
        The property the guarantee actually claims, measured end to end.

        A bin 40 km out costs about 218 to insert, comfortably past the old 180
        cap, so before the fix it was skipped at a wait of a million hours.  It
        must now be collected, and by the hour the bound predicts.
        """
        travel, vehicle = self._travel(40.0), VRP.VehicleSpec(
            vehicle_id=0, depot_index=0, shift_minutes=1200.0)
        cost = VRP.route_cost(
            VRP.evaluate_route([self._task(AG.DEFAULT_TAU_H)], vehicle, travel),
            self.WEIGHTS)
        predicted = AG.worst_case_wait_bound(
            tau_h=AG.DEFAULT_TAU_H, cycle_h=12.0, max_overdue=1,
            served_overdue_per_cycle=1, max_insertion_cost=cost,
            lambda_prize=self.WEIGHTS.lambda_prize,
            overdue_multiplier=self.WEIGHTS.overdue_multiplier)

        served_at = None
        for wait in range(48, 1201, 12):
            plan = VRP.solve([self._task(wait)], [vehicle], travel, self.WEIGHTS,
                             time_budget_s=0.4)
            if any(r.stops for r in plan.routes):
                served_at = wait
                break
        self.assertIsNotNone(served_at, "remote overdue bin was never served")
        self.assertLessEqual(served_at, predicted + 1e-9)

    def test_bound_is_tighter_than_the_retired_form(self):
        """
        The retired bound was true but loose by tau + Delta - Delta*ceil(tau/Delta).

        At the defaults that is 12 h, which is exactly the gap between the 60.0 h
        the old formula stated and the 48.0 h the rollout actually attained.
        """
        tight = AG.worst_case_wait_bound(tau_h=48.0, cycle_h=12.0,
                                         max_overdue=1, served_overdue_per_cycle=1)
        retired = 48.0 + 12.0 * 1
        self.assertAlmostEqual(tight, 48.0, places=9)
        self.assertAlmostEqual(retired - tight, 12.0, places=9)

    def test_bound_reports_no_guarantee_when_the_fleet_cannot_keep_up(self):
        self.assertEqual(
            AG.worst_case_wait_bound(served_overdue_per_cycle=0), math.inf)

    def test_pressure_is_zero_until_a_bin_is_overdue(self):
        """A bin inside its deadline is priced on urgency alone, as before."""
        tiered = AG.effective_priorities({1: 0.3}, {1: AG.DEFAULT_TAU_H - 1.0},
                                         {1: 0.0})
        self.assertEqual(tiered[1].tier, AG.TIER_NORMAL)
        self.assertEqual(tiered[1].pressure, 0.0)

    def test_overdue_tier_is_served_longest_wait_first(self):
        """
        The queueing term of the bound assumes a bin cannot be overtaken.

        The case has to be one where waiting time and cost genuinely disagree,
        or it proves nothing.  The longest-waiting bin is put 20 km out and the
        younger one 5 km out, with waits close enough (60 h against 49 h) that the
        difference in skip penalty, 20.6 cost units, is smaller than the 82 units
        of extra detour the older bin costs.  A planner choosing on cost alone
        takes the near bin.

        Before the ordering was enforced the constructor selected inside the tier
        by regret and did exactly that, so a bin promoted earlier could be
        overtaken and the proof described an order the code did not implement.
        Removing the ordering from `construct_regret2` makes this test serve node
        2 instead of node 1.
        """
        travel = VRP.TravelModel(
            np.array([[0.0, 20000.0, 5000.0],
                      [20000.0, 0.0, 18000.0],
                      [5000.0, 18000.0, 0.0]]), default_speed_kmh=20.0)
        # Long enough for the far bin alone, not for both.
        vehicle = VRP.VehicleSpec(vehicle_id=0, depot_index=0, shift_minutes=145.0)

        def task(node_id, index, wait_h):
            tiered = AG.effective_priorities({node_id: 0.30}, {node_id: wait_h},
                                             {node_id: 0.0})
            return SC.make_tasks(
                [node_id], {node_id: 0.5}, {node_id: tiered[node_id].score},
                index_of={node_id: index}, hazards={node_id: False},
                tiers={node_id: tiered[node_id].tier},
                overdue_pressures={node_id: tiered[node_id].pressure},
                densities={node_id: 220.0}, capacities_l={node_id: 1100.0},
            )[0]

        older = task(1, 1, 60.0)      # waiting longer, 20 km out
        younger = task(2, 2, 49.0)    # waiting less, 5 km out
        self.assertEqual(older.tier, AG.TIER_OVERDUE)
        self.assertEqual(younger.tier, AG.TIER_OVERDUE)
        self.assertGreater(older.prize, younger.prize)

        orders, _ = VRP.construct_regret2(
            [younger, older], [vehicle], travel, self.WEIGHTS)
        served = [t.node_id for order in orders.values() for t in order]
        self.assertEqual(served, [1],
                         "the longest-waiting overdue bin was overtaken by a "
                         "younger one that happened to be cheaper to reach")

    def test_ordering_score_is_strictly_increasing_in_wait(self):
        """Lemma 1 of the proof, checked directly rather than assumed."""
        scores = [AG.overdue_score(w, AG.DEFAULT_TAU_H)
                  for w in (48, 49, 60, 96, 240, 1000)]
        self.assertEqual(scores, sorted(scores))
        self.assertEqual(len(set(scores)), len(scores), "two waits tied")


class ReservedHeadTests(SimpleTestCase):
    """
    The reservation is what the wait bound rests on, so it is pinned directly.

    A price is a preference and a search can always decline it.  These cases
    check the three things the proof needs and nothing else supplies: the queue
    has one order wherever it is read, the reserved bins are served in every
    plan, and the bound they give is attained on an instance built to attain it.
    """

    WEIGHTS = VRP.ObjectiveWeights()
    TAU, CYCLE = 48.0, 12.0

    def _tasks(self, waits, priority=0.30, hazards=None, windows=None,
               escalating_price=True):
        """Bins 1..n at matrix rows 1..n, built through the production path."""
        ids = sorted(waits)
        hazards = hazards or {}
        tiered = AG.effective_priorities(
            {i: priority for i in ids}, waits,
            {i: (1.0 if hazards.get(i) else 0.0) for i in ids},
            tau_h=self.TAU, escalating_price=escalating_price)
        return SC.make_tasks(
            ids, {i: 0.5 for i in ids}, {i: tiered[i].score for i in ids},
            index_of={i: i for i in ids},
            hazards={i: bool(hazards.get(i)) for i in ids},
            tiers={i: tiered[i].tier for i in ids},
            overdue_pressures={i: tiered[i].pressure for i in ids},
            waits={i: tiered[i].wait_hours for i in ids},
            overdue={i: tiered[i].overdue for i in ids},
            tto_hours={i: (2.0 if hazards.get(i) else float("inf")) for i in ids},
            densities={i: 220.0 for i in ids},
            capacities_l={i: 1100.0 for i in ids},
            windows=windows or {},
        )

    def _star(self, n, spoke_km=10.0):
        """
        A depot with ``n`` bins, each on its own spoke.

        Travel between two bins passes the depot, so a round serving ``k`` bins
        is as long as ``k`` separate round trips and the shift length alone
        decides how many fit.
        """
        size = n + 1
        matrix = np.full((size, size), 2.0 * spoke_km * 1000.0)
        matrix[0, :] = matrix[:, 0] = spoke_km * 1000.0
        np.fill_diagonal(matrix, 0.0)
        return VRP.TravelModel(matrix, default_speed_kmh=20.0)

    def _shift_for(self, k, spoke_km=10.0):
        """Shift that fits exactly ``k`` spokes: legs, service and one tip."""
        leg = 60.0 * spoke_km / 20.0
        return 2.0 * k * leg + 4.0 * k + 15.0 + 2.0

    def test_queue_is_ordered_by_wait_then_by_id(self):
        tasks = self._tasks({3: 60.0, 1: 60.0, 2: 72.0, 4: 24.0})
        queue = VRP.overdue_queue(list(reversed(tasks)))
        self.assertEqual([t.node_id for t in queue], [2, 1, 3],
                         "the queue must not depend on the order of the list")

    def test_queue_order_survives_the_bounded_price(self):
        """With the price switched off the queue must still know who waited."""
        tasks = self._tasks({1: 60.0, 2: 96.0, 3: 49.0}, escalating_price=False)
        self.assertTrue(all(t.overdue_pressure == 0.0 for t in tasks))
        self.assertEqual([t.node_id for t in VRP.overdue_queue(tasks)], [2, 1, 3])

    def test_a_lone_overdue_bin_is_served_at_the_deadline(self):
        """
        The case the controlled sweep cannot show, because it switches this off.

        Forty kilometres out a round trip costs about 218 units against a skip
        penalty of 90 at the deadline, so the price alone declines it.  Reserved,
        it is served in the first cycle it is overdue.
        """
        travel = VRP.TravelModel(np.array([[0.0, 40000.0], [40000.0, 0.0]]),
                                 default_speed_kmh=20.0)
        vehicle = VRP.VehicleSpec(vehicle_id=0, depot_index=0, shift_minutes=1200.0)
        priced = VRP.solve(self._tasks({1: self.TAU}), [vehicle], travel,
                           self.WEIGHTS, time_budget_s=0.2, reserve_overdue=0)
        self.assertEqual(priced.metrics["bins_served"], 0)
        reserved = VRP.solve(self._tasks({1: self.TAU}), [vehicle], travel,
                             self.WEIGHTS, time_budget_s=0.2, reserve_overdue=1)
        self.assertEqual(reserved.metrics["bins_served"], 1)
        self.assertEqual(reserved.metrics["heads_reserved"], 1)
        self.assertEqual(reserved.metrics["head_repairs"], 0)

    def test_the_certificate_is_a_feasible_prefix_of_the_queue(self):
        tasks = self._tasks({i: 48.0 + 12.0 * i for i in range(1, 7)})
        travel = self._star(6)
        vehicle = VRP.VehicleSpec(vehicle_id=0, depot_index=0,
                                  shift_minutes=self._shift_for(3))
        queue = VRP.overdue_queue(tasks)
        heads, routes, unservable = VRP.certify_heads(
            queue, [vehicle], travel, self.WEIGHTS, reserve=5)
        # Five were asked for and three fit, so three are reserved.
        self.assertEqual([t.node_id for t in heads],
                         [t.node_id for t in queue[:3]])
        self.assertEqual(unservable, [])
        self.assertIsNotNone(VRP.evaluate_route(routes[0], vehicle, travel))
        self.assertEqual(sorted(t.node_id for t in routes[0]),
                         sorted(t.node_id for t in heads))

    def test_every_reserved_head_is_served_and_the_marks_are_cleared(self):
        rng = np.random.default_rng(11)
        for reserve in (1, 2, 4):
            waits = {i: float(rng.choice([0.0, 24.0, 48.0, 60.0, 72.0, 96.0]))
                     for i in range(1, 19)}
            tasks = self._tasks(waits, hazards={2: True, 5: True, 9: True})
            coords = [(23.8069, 90.3687)] + [
                (23.780 + float(rng.uniform(0, 0.055)),
                 90.348 + float(rng.uniform(0, 0.042))) for _ in range(18)]
            travel = SC.build_travel(
                coords, datetime(2026, 3, 3, 7, 30, tzinfo=timezone.utc),
                provider=TR.SyntheticTrafficProvider(seed=42))
            vehicle = VRP.VehicleSpec(vehicle_id=0, depot_index=0,
                                      shift_minutes=150.0)
            plan = VRP.solve(tasks, [vehicle], travel, self.WEIGHTS,
                             time_budget_s=0.5, reserve_overdue=reserve)
            served = {s.task.node_id for r in plan.routes for s in r.stops}
            self.assertGreaterEqual(plan.metrics["heads_reserved"], 1)
            self.assertLessEqual(plan.metrics["heads_reserved"], reserve)
            self.assertTrue(set(plan.metrics["head_ids"]) <= served,
                            "a reserved head was left out of the plan")
            self.assertEqual(plan.metrics["heads_served"],
                             plan.metrics["heads_reserved"])
            self.assertEqual(plan.metrics["head_repairs"], 0)
            self.assertFalse(any(t.mandatory for t in tasks),
                             "a mandatory mark outlived the call that set it")
            self.assertLess(plan.objective, VRP.MANDATORY_SKIP_COST / 2,
                            "the mandatory cost leaked into the reported objective")

    def test_a_reserved_head_outlasts_a_flood_of_hazards(self):
        """
        The hazard tier outranks the overdue tier in the ranking, and with a
        short shift the hazards alone fill the round.  The head is still served,
        because it is in the plan before the hazards are inserted.
        """
        waits = {i: 0.0 for i in range(1, 9)}
        waits[9] = 120.0
        tasks = self._tasks(waits, hazards={i: True for i in range(1, 9)})
        travel = self._star(9, spoke_km=4.0)
        vehicle = VRP.VehicleSpec(vehicle_id=0, depot_index=0,
                                  shift_minutes=self._shift_for(3, spoke_km=4.0))
        plan = VRP.solve(tasks, [vehicle], travel, self.WEIGHTS,
                         time_budget_s=0.3, reserve_overdue=1)
        served = {s.task.node_id for r in plan.routes for s in r.stops}
        self.assertIn(9, served)
        self.assertEqual(plan.metrics["bins_served"], 3)

    def test_a_bin_nobody_can_serve_does_not_block_the_queue(self):
        """Its window opens after the shift ends, so no policy can collect it."""
        tasks = self._tasks({1: 200.0, 2: 96.0, 3: 60.0},
                            windows={1: (400.0, 480.0)})
        travel = self._star(3, spoke_km=2.0)
        vehicle = VRP.VehicleSpec(vehicle_id=0, depot_index=0, shift_minutes=120.0)
        heads, _routes, unservable = VRP.certify_heads(
            VRP.overdue_queue(tasks), [vehicle], travel, self.WEIGHTS, reserve=1)
        self.assertEqual([t.node_id for t in unservable], [1])
        self.assertEqual([t.node_id for t in heads], [2])

    def test_a_dropped_head_is_restored(self):
        """A solver that ignores the reservation loses the head; the repair must not."""
        tasks = self._tasks({i: 48.0 + 12.0 * i for i in range(1, 5)})
        travel = self._star(4)
        vehicle = VRP.VehicleSpec(vehicle_id=0, depot_index=0,
                                  shift_minutes=self._shift_for(2))
        queue = VRP.overdue_queue(tasks)
        heads, certificate, _ = VRP.certify_heads(queue, [vehicle], travel,
                                                  self.WEIGHTS, reserve=2)
        head_ids = [t.node_id for t in heads]
        others = [t for t in tasks if t.node_id not in head_ids]
        for head in heads:
            head.mandatory = True
        try:
            orders, unserved = VRP.restore_heads(
                heads, certificate, {0: others}, list(heads),
                [vehicle], travel, self.WEIGHTS)
        finally:
            for head in heads:
                head.mandatory = False
        self.assertEqual(sorted(t.node_id for t in orders[0]), sorted(head_ids))
        self.assertEqual(sorted(t.node_id for t in unserved),
                         sorted(t.node_id for t in others))

    def test_queue_bound_values(self):
        bound = AG.queue_wait_bound
        self.assertEqual(bound(48.0, 12.0, 9, 1), 144.0)
        self.assertEqual(bound(48.0, 12.0, 60, 1), 756.0)
        self.assertEqual(bound(48.0, 12.0, 60, 4), 216.0)
        self.assertEqual(bound(48.0, 12.0, 60, 8), 132.0)
        # A deadline that is not a multiple of the cycle is first seen late.
        self.assertEqual(bound(50.0, 12.0, 1, 1), 60.0)
        self.assertEqual(bound(48.0, 12.0, 5, 0), math.inf)
        # It is the pricing form with the pricing delay left out.
        self.assertEqual(
            bound(48.0, 12.0, 17, 3),
            AG.worst_case_wait_bound(tau_h=48.0, cycle_h=12.0, max_overdue=17,
                                     served_overdue_per_cycle=3))

    def test_the_bound_is_attained_on_a_star_network(self):
        """
        Tightness, by construction.

        Six bins on six spokes and a shift that fits exactly ``r`` of them.
        Every bin starts at zero, so all six reach the deadline together and the
        backlog is the whole network.  The last bin of the first pass is then
        served at exactly the bound, for one reserved head and for two.
        """
        n = 6
        for reserve in (1, 2):
            travel = self._star(n)
            vehicle = VRP.VehicleSpec(vehicle_id=0, depot_index=0,
                                      shift_minutes=self._shift_for(reserve))
            waits = {i: 0.0 for i in range(1, n + 1)}
            worst_served, worst_backlog = 0.0, 0
            for _cycle in range(16):
                tasks = self._tasks(waits)
                worst_backlog = max(worst_backlog,
                                    len(VRP.overdue_queue(tasks)))
                plan = VRP.solve(tasks, [vehicle], travel, self.WEIGHTS,
                                 time_budget_s=0.2, reserve_overdue=reserve)
                served = {s.task.node_id for r in plan.routes for s in r.stops}
                self.assertEqual(plan.metrics["heads_served"],
                                 plan.metrics["heads_reserved"])
                for node_id in waits:
                    if node_id in served:
                        worst_served = max(worst_served, waits[node_id])
                        waits[node_id] = self.CYCLE
                    else:
                        waits[node_id] += self.CYCLE
            self.assertEqual(worst_backlog, n)
            self.assertEqual(
                worst_served,
                AG.queue_wait_bound(self.TAU, self.CYCLE, n, reserve),
                f"bound not attained with {reserve} reserved")


class RoadNetworkTests(SimpleTestCase):
    """
    The street-graph distance model.

    These run against the cached study areas committed under ``data/roadnet``.
    They never touch the network, so a failure here is a defect in the code and
    not an outage at Overpass.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.net = RN.load("dhaka")

    def test_graph_is_strongly_connected(self):
        """Every stop must be reachable from every other, or a plan is a fiction."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        adj = csr_matrix((self.net.length_m, (self.net.src, self.net.dst)),
                         shape=(self.net.n_nodes, self.net.n_nodes))
        n_comp, _ = connected_components(adj, directed=True, connection="strong")
        self.assertEqual(n_comp, 1)

    def test_distances_exceed_great_circle(self):
        """A road path can never be shorter than the straight line it spans."""
        coords = [(23.8069, 90.3687), (23.7910, 90.3550), (23.8203, 90.3650),
                  (23.7780, 90.3800)]
        distance, _freeflow, _snap = self.net.matrices(coords)
        for i in range(len(coords)):
            for j in range(len(coords)):
                if i == j:
                    continue
                straight = GEO.haversine(coords[i][0], coords[i][1],
                                         coords[j][0], coords[j][1])
                self.assertGreaterEqual(distance[i][j], straight - 1.0)

    def test_matrix_is_asymmetric(self):
        """One-way streets make the matrix directed; a symmetric one hides them."""
        coords = [(23.8069, 90.3687), (23.7910, 90.3550), (23.8203, 90.3650),
                  (23.8100, 90.3780), (23.7890, 90.3740)]
        distance, _freeflow, _snap = self.net.matrices(coords)
        self.assertFalse(np.allclose(distance, distance.T, atol=1.0))

    def test_diagonal_is_zero_distance_and_undefined_speed(self):
        coords = [(23.8069, 90.3687), (23.7910, 90.3550)]
        distance, freeflow, _snap = self.net.matrices(coords)
        self.assertEqual(distance[0][0], 0.0)
        self.assertTrue(math.isnan(freeflow[0][0]))

    def test_freeflow_is_within_the_arc_speed_range(self):
        """A path speed is a weighted mean, so it cannot leave the arc range."""
        coords = [(23.8069, 90.3687), (23.7910, 90.3550), (23.8203, 90.3650)]
        _distance, freeflow, _snap = self.net.matrices(coords)
        finite = freeflow[np.isfinite(freeflow)]
        self.assertGreater(finite.size, 0)
        self.assertGreaterEqual(finite.min(), self.net.speed_kmh.min() - 1e-6)
        self.assertLessEqual(finite.max(), self.net.speed_kmh.max() + 1e-6)

    def test_circuity_excludes_near_coincident_pairs(self):
        """Two stops metres apart give a ratio that says nothing about the grid."""
        coords = [(23.8069, 90.3687), (23.80691, 90.36871), (23.7910, 90.3550)]
        ratio = self.net.circuity(coords, min_straight_m=100.0)
        self.assertTrue(math.isnan(ratio[0][1]))
        self.assertTrue(np.isfinite(ratio[0][2]))

    def test_snapping_stays_close_to_the_requested_point(self):
        coords = [(23.8069, 90.3687), (23.7910, 90.3550), (23.8203, 90.3650)]
        _idx, snap_m = self.net.snap(coords)
        self.assertLess(float(snap_m.max()), 300.0)

    def test_maxspeed_parsing(self):
        self.assertAlmostEqual(RN._parse_maxspeed("50"), 50.0)
        self.assertAlmostEqual(RN._parse_maxspeed("30 mph"), 48.28032, places=4)
        self.assertIsNone(RN._parse_maxspeed("signals"))
        self.assertIsNone(RN._parse_maxspeed(None))
        self.assertIsNone(RN._parse_maxspeed("none"))

    def test_oneway_tag_reading(self):
        self.assertEqual(RN._is_oneway({"oneway": "yes"}), 1)
        self.assertEqual(RN._is_oneway({"oneway": "-1"}), -1)
        self.assertEqual(RN._is_oneway({"highway": "residential"}), 0)
        self.assertEqual(RN._is_oneway({"junction": "roundabout"}), 1)
        # An explicit oneway=no must beat the motorway default.
        self.assertEqual(RN._is_oneway({"highway": "motorway", "oneway": "no"}), 0)


class RoadNetworkTravelTests(SimpleTestCase):
    """The travel model must use the per-leg free-flow speed the graph supplies."""

    def test_per_leg_freeflow_changes_the_derated_speed(self):
        coords = [(23.8069, 90.3687), (23.7910, 90.3550)]
        distance = np.array([[0.0, 4000.0], [4000.0, 0.0]])
        fast = np.array([[np.nan, 60.0], [60.0, np.nan]])
        slow = np.array([[np.nan, 15.0], [15.0, np.nan]])
        when = datetime(2026, 3, 3, 7, 30, tzinfo=timezone.utc)
        provider = TR.SyntheticTrafficProvider(seed=42)

        ctx_fast = TR.build_travel_context(coords, distance, provider, when,
                                           freeflow_matrix=fast)
        ctx_slow = TR.build_travel_context(coords, distance, provider, when,
                                           freeflow_matrix=slow)
        self.assertGreater(ctx_fast.speed_kmh(0, 1), ctx_slow.speed_kmh(0, 1))

    def test_directed_cache_does_not_merge_the_two_directions(self):
        """With a directed graph the cache key must keep i->j apart from j->i."""
        coords = [(23.8069, 90.3687), (23.7910, 90.3550)]
        distance = np.array([[0.0, 4000.0], [4200.0, 0.0]])
        freeflow = np.array([[np.nan, 60.0], [15.0, np.nan]])
        when = datetime(2026, 3, 3, 7, 30, tzinfo=timezone.utc)
        ctx = TR.build_travel_context(coords, distance,
                                      TR.SyntheticTrafficProvider(seed=42), when,
                                      freeflow_matrix=freeflow)
        self.assertNotAlmostEqual(ctx.speed_kmh(0, 1), ctx.speed_kmh(1, 0))

    def test_absent_matrix_keeps_the_single_constant(self):
        coords = [(23.8069, 90.3687), (23.7910, 90.3550)]
        distance = np.array([[0.0, 4000.0], [4000.0, 0.0]])
        when = datetime(2026, 3, 3, 7, 30, tzinfo=timezone.utc)
        ctx = TR.build_travel_context(coords, distance,
                                      TR.SyntheticTrafficProvider(seed=42), when,
                                      freeflow_kmh=34.0)
        self.assertEqual(ctx.leg_freeflow_kmh(0, 1), 34.0)

    def test_scenario_builds_road_distances_when_given_a_network(self):
        net = RN.load("dhaka")
        coords = [(23.8069, 90.3687), (23.7910, 90.3550), (23.8203, 90.3650)]
        when = datetime(2026, 3, 3, 7, 30, tzinfo=timezone.utc)
        road = SC.build_travel(coords, when, road_network=net)
        great_circle = SC.build_travel(coords, when)
        # The 1.30 constant is an approximation of the real street pattern, so
        # the two must not agree to the metre on every pair.
        self.assertFalse(np.allclose(road.distance_m, great_circle.distance_m,
                                     atol=1.0))
        self.assertGreater(road.distance(0, 1), 0.0)


class ConstructionCacheTests(SimpleTestCase):
    """
    The insertion table is memoised between iterations, not approximated.

    Inserting a bin changes one vehicle's route, so every other pairing keeps
    its costed answer.  If that reasoning is ever wrong the constructor silently
    becomes a different heuristic, so it is pinned rather than argued.
    """

    def _instance(self, n_bins: int, seed: int):
        rng = np.random.default_rng(seed)
        coords = [(23.8069, 90.3687)]
        tasks = []
        for i in range(n_bins):
            coords.append((23.780 + float(rng.uniform(0, 0.055)),
                           90.348 + float(rng.uniform(0, 0.042))))
            tasks.append(VRP.BinTask(
                node_id=i + 1, index=i + 1,
                prize=float(rng.uniform(0.05, 0.95)),
                load_kg=float(rng.uniform(40, 260)),
                service_minutes=float(rng.uniform(2.5, 6.0)),
                window_start_min=float(rng.choice([0, 0, 120, 180])),
                window_end_min=480.0,
                stream=str(rng.choice(["general", "general", "recyclable", "organic"])),
                density_kg_per_m3=float(rng.uniform(180, 260)),
                tier=int(rng.choice([1, 2, 2, 2])),
                overdue_pressure=float(rng.uniform(0.0, 3.0)),
            ))
        when = datetime(2026, 3, 3, 7, 30, tzinfo=timezone.utc)
        travel = SC.build_travel(coords, when,
                                 provider=TR.SyntheticTrafficProvider(seed=42))
        fleet = SC.make_fleet(3, depot_index=0, capacity_kg=4500.0,
                              shift_minutes=480.0)
        fleet[0].accepts_streams = ("general", "organic", "recyclable")
        fleet[1].accepts_streams = ("general", "recyclable")
        fleet[2].accepts_streams = ("general", "organic")
        return tasks, fleet, travel

    def test_cached_and_uncached_construction_agree(self):
        for seed in (1, 7, 13):
            tasks, fleet, travel = self._instance(24, seed)
            weights = VRP.ObjectiveWeights()
            cached = VRP.construct_regret2(tasks, fleet, travel, weights,
                                           use_cache=True)
            plain = VRP.construct_regret2(tasks, fleet, travel, weights,
                                          use_cache=False)
            for vid in cached[0]:
                self.assertEqual([t.node_id for t in cached[0][vid]],
                                 [t.node_id for t in plain[0][vid]],
                                 msg=f"vehicle {vid} diverged at seed {seed}")
            self.assertEqual(sorted(t.node_id for t in cached[1]),
                             sorted(t.node_id for t in plain[1]))

    def test_cache_is_invalidated_for_the_touched_vehicle(self):
        """A stale entry for the changed vehicle would reorder the insertions."""
        tasks, fleet, travel = self._instance(18, seed=3)
        weights = VRP.ObjectiveWeights()
        orders, _unserved = VRP.construct_regret2(tasks, fleet, travel, weights)
        # Every routed bin must be feasible in the order it was placed, which is
        # only true if each insertion was costed against the route as it stood.
        for vehicle in fleet:
            order = orders[vehicle.vehicle_id]
            if order:
                self.assertIsNotNone(VRP.evaluate_route(order, vehicle, travel))


class WeatherTests(SimpleTestCase):
    """
    Each mapping from weather to a model parameter against a hand-computed value.

    The numbers come from published functions, so a wrong coefficient would not
    fail anywhere else: the planner would simply plan for different weather.
    The provider cases never touch the network, and the documents they parse are
    written here to the published schema with invented values.
    """

    HOT = WX.Conditions(temperature_c=38.0, relative_humidity=45.0, uv_index=9.0)

    def test_rain_classes_follow_the_published_bounds(self):
        cases = [(0.0, "dry"), (0.09, "dry"), (0.1, "slight"), (2.49, "slight"),
                 (2.5, "moderate"), (9.99, "moderate"), (10.0, "heavy"),
                 (49.9, "heavy"), (50.0, "violent")]
        for rate, expected in cases:
            self.assertEqual(WX.rain_class(rate), expected, msg=f"{rate} mm/h")

    def test_speed_factor_never_rises_with_rain(self):
        order = ["dry", "slight", "moderate", "heavy", "violent"]
        for table in (WX.RAIN_SPEED_FACTOR, WX.RAIN_SPEED_FACTOR_SEVERE):
            values = [table[k] for k in order]
            self.assertEqual(values, sorted(values, reverse=True))
            self.assertEqual(table["dry"], 1.0)

    def test_standing_water_caps_the_speed(self):
        self.assertTrue(math.isinf(WX.flood_speed_cap_kmh(0.0)))
        # 0.0009 * 150^2 - 0.5529 * 150 + 86.9448
        self.assertAlmostEqual(WX.flood_speed_cap_kmh(150.0), 24.2598, places=4)
        self.assertAlmostEqual(WX.flood_speed_cap_kmh(100.0), 40.6548, places=4)
        self.assertEqual(WX.flood_speed_cap_kmh(300.0), 0.0)
        self.assertFalse(WX.effects(WX.Conditions(standing_water_mm=300.0)).passable)

    def test_wet_bulb_matches_the_worked_example(self):
        """Stull gives 13.7 C for 20 C at 50 percent relative humidity."""
        self.assertAlmostEqual(WX.wet_bulb_c(20.0, 50.0), 13.70, places=2)

    def test_work_capacity_is_one_half_at_the_fitted_midpoint(self):
        self.assertAlmostEqual(WX.work_capacity(33.63), 0.5, places=12)
        self.assertAlmostEqual(WX.work_capacity(40.0), 0.2501, places=4)
        values = [WX.work_capacity(w) for w in (15, 20, 25, 30, 35, 40)]
        self.assertEqual(values, sorted(values, reverse=True))
        # The function is fitted from 12 to 40 C and is held flat outside that.
        self.assertEqual(WX.work_capacity(5.0), WX.work_capacity(12.0))
        self.assertEqual(WX.work_capacity(48.0), WX.work_capacity(40.0))

    def test_nominal_conditions_change_nothing(self):
        effect = WX.effects(WX.NOMINAL)
        self.assertTrue(effect.neutral)
        self.assertEqual(effect.rain_class, "dry")
        self.assertFalse(effect.sun_exposed)
        # 0.7 * wet bulb + 0.3 * air temperature, in shade.
        self.assertAlmostEqual(effect.wbgt_c, 0.7 * WX.wet_bulb_c(28.0, 60.0) + 8.4, places=9)

    def test_heat_slows_service_and_sun_slows_it_further(self):
        shade = WX.effects(replace_conditions(self.HOT, uv_index=2.0))
        sun = WX.effects(self.HOT)
        night = WX.effects(replace_conditions(self.HOT, is_daytime=False))
        self.assertGreater(shade.service_factor, 1.0)
        self.assertGreater(sun.service_factor, shade.service_factor)
        self.assertAlmostEqual(sun.wbgt_c - shade.wbgt_c, 3.0, places=9)
        self.assertEqual(night.service_factor, shade.service_factor,
                         "a high index after dark carries no solar load")
        self.assertEqual(sun.uv_category, "very high")
        self.assertAlmostEqual(sun.service_factor,
                               WX.work_capacity(WX.wbgt_c(WX.NOMINAL)) / sun.work_capacity,
                               places=12)

    def test_guideline_capacity_is_the_published_form(self):
        self.assertEqual(WX.work_capacity_guideline(25.0), 1.0)
        self.assertAlmostEqual(WX.work_capacity_guideline(26.0), 0.75, places=12)
        # 100 - 25 * (33 - 25)^(2/3) = 100 - 25 * 4 = 0, held at the floor.
        self.assertEqual(WX.work_capacity_guideline(33.0), 0.10)

    def test_gas_follows_the_measured_ratio_inside_its_range(self):
        self.assertAlmostEqual(WX.DECOMPOSITION_Q10 ** 2.5, 28.0, places=9)
        self.assertEqual(WX.decomposition_factor(28.0), 1.0)
        self.assertAlmostEqual(WX.decomposition_factor(35.0) / WX.decomposition_factor(10.0),
                               28.0, places=9)
        self.assertEqual(WX.decomposition_factor(42.0), WX.decomposition_factor(35.0))
        self.assertEqual(WX.decomposition_factor(2.0), WX.decomposition_factor(10.0))

    def test_litter_demand_uses_the_estimated_coefficients(self):
        warm = WX.Conditions(temperature_c=33.0)
        self.assertAlmostEqual(WX.litter_demand_factor(warm), math.exp(0.1025 * 0.5), places=12)
        wet = WX.Conditions(temperature_c=28.0, rain_mm_h=4.0)
        self.assertAlmostEqual(WX.litter_demand_factor(wet), math.exp(-0.0903), places=12)
        # The daily maximum, when given, replaces the temperature of the hour.
        evening = WX.Conditions(temperature_c=24.0, daily_max_c=33.0)
        self.assertEqual(WX.litter_demand_factor(evening), WX.litter_demand_factor(warm))
        # Held at the edge of the range the record covers.
        self.assertEqual(WX.litter_demand_factor(WX.Conditions(temperature_c=45.0)),
                         WX.litter_demand_factor(WX.Conditions(temperature_c=37.8)))

    def _line(self):
        """Depot and two bins on a line, 10 km apart, at 30 km/h."""
        matrix = np.array([[0.0, 10e3, 20e3], [10e3, 0.0, 10e3], [20e3, 10e3, 0.0]])
        return VRP.TravelModel(matrix, default_speed_kmh=30.0)

    def test_travel_model_scales_speed_and_keeps_distance(self):
        base = self._line()
        rain = WX.travel_under(base, WX.effects(WX.Conditions(rain_mm_h=20.0)))
        self.assertEqual(rain.distance(0, 1), base.distance(0, 1))
        self.assertAlmostEqual(rain.speed_kmh(0, 1), 30.0 * 0.88, places=9)
        self.assertAlmostEqual(rain.minutes(0, 1), base.minutes(0, 1) / 0.88, places=9)
        self.assertEqual(rain.friction(0, 1), base.friction(0, 1))

        flooded = WX.travel_under(
            base, WX.effects(WX.Conditions(rain_mm_h=60.0, standing_water_mm=150.0)))
        # 30 * 0.82 = 24.6 km/h, above the 24.26 km/h the water allows.
        self.assertAlmostEqual(flooded.speed_kmh(0, 1), 24.2598, places=4)

        self.assertIs(WX.travel_under(base, WX.NEUTRAL), base)
        with self.assertRaises(ValueError):
            WX.travel_under(
                base, WX.effects(WX.Conditions(standing_water_mm=320.0))).minutes(0, 1)

    def test_a_route_that_just_fits_no_longer_fits_in_rain(self):
        base = self._line()
        task = VRP.BinTask(node_id=1, index=1, load_kg=100.0, service_minutes=4.0)
        vehicle = VRP.VehicleSpec(vehicle_id=1, shift_minutes=60.0, tipping_minutes=15.0)
        # 20 minutes out, 4 of service, 20 back and 15 to tip: 59 of 60 minutes.
        self.assertIsNotNone(VRP.evaluate_route([task], vehicle, base))
        rain = WX.travel_under(base, WX.effects(WX.Conditions(rain_mm_h=20.0)))
        self.assertIsNone(VRP.evaluate_route([task], vehicle, rain))

    def test_service_time_follows_the_work_capacity(self):
        tasks = [VRP.BinTask(node_id=1, index=1, service_minutes=4.0)]
        effect = WX.effects(self.HOT)
        slowed = WX.tasks_under(tasks, effect)
        self.assertAlmostEqual(slowed[0].service_minutes, 4.0 * effect.service_factor,
                               places=12)
        self.assertEqual(tasks[0].service_minutes, 4.0, "the caller's tasks are not altered")
        self.assertEqual(WX.tasks_under(tasks, WX.NEUTRAL)[0].service_minutes, 4.0)

    def _two_stops(self):
        return [VRP.BinTask(node_id=1, index=1, load_kg=100.0, service_minutes=4.0),
                VRP.BinTask(node_id=2, index=2, load_kg=100.0, service_minutes=4.0)]

    def test_driving_in_nominal_weather_reproduces_the_plan(self):
        base, tasks = self._line(), self._two_stops()
        vehicle = VRP.VehicleSpec(vehicle_id=1, shift_minutes=240.0)
        planned = VRP.evaluate_route(tasks, vehicle, base)
        driven, dropped, overrun = WX.drive_route(tasks, vehicle, base, [WX.NEUTRAL] * 4)
        self.assertEqual(dropped, [])
        self.assertEqual(overrun, 0.0)
        for name in ("distance_m", "co2_kg", "duration_min", "load_kg", "trips"):
            self.assertAlmostEqual(getattr(driven, name), getattr(planned, name),
                                   places=9, msg=name)
        self.assertEqual([s.start_service_min for s in driven.stops],
                         [s.start_service_min for s in planned.stops])
        # 20 + 4 + 20 + 4 + 40 minutes on the road and at stops, 15 to tip.
        self.assertAlmostEqual(driven.duration_min, 103.0, places=9)

    def test_a_route_driven_in_rain_is_cut_to_what_fits(self):
        base, tasks = self._line(), self._two_stops()
        vehicle = VRP.VehicleSpec(vehicle_id=1, shift_minutes=104.0)
        rain = WX.effects(WX.Conditions(rain_mm_h=20.0))
        self.assertEqual(rain.service_factor, 1.0)
        driven, dropped, overrun = WX.drive_route(tasks, vehicle, base, [rain] * 2)
        self.assertEqual([s.task.node_id for s in driven.stops], [1])
        self.assertEqual([t.node_id for t in dropped], [2])
        # The whole plan: 80 minutes of driving at 0.88 of the speed, 8 of
        # service and 15 to tip, against a shift of 104.
        self.assertAlmostEqual(overrun, 80.0 / 0.88 + 8.0 + 15.0 - 104.0, places=9)
        self.assertAlmostEqual(driven.duration_min, 40.0 / 0.88 + 4.0 + 15.0, places=9)
        self.assertAlmostEqual(driven.distance_m, 20e3, places=6)
        self.assertLessEqual(driven.duration_min, vehicle.shift_minutes)

    def test_a_stop_that_does_not_fit_is_skipped_and_the_next_is_tried(self):
        base = self._line()
        far_first = list(reversed(self._two_stops()))
        vehicle = VRP.VehicleSpec(vehicle_id=1, shift_minutes=100.0)
        rain = WX.effects(WX.Conditions(rain_mm_h=20.0))
        # The far stop alone needs 80 / 0.88 + 4 + 15 = 109.9 minutes.  The near
        # one, taken from the depot, needs 40 / 0.88 + 4 + 15 = 64.5.
        driven, dropped, _overrun = WX.drive_route(far_first, vehicle, base, [rain] * 2)
        self.assertEqual([s.task.node_id for s in driven.stops], [1])
        self.assertEqual([t.node_id for t in dropped], [2])
        self.assertAlmostEqual(driven.duration_min, 40.0 / 0.88 + 4.0 + 15.0, places=9)

    def test_a_protected_stop_is_kept_at_the_cost_of_the_others(self):
        base, tasks = self._line(), self._two_stops()
        vehicle = VRP.VehicleSpec(vehicle_id=1, shift_minutes=112.0)
        rain = WX.effects(WX.Conditions(rain_mm_h=20.0))
        # Both stops need 113.9 minutes in the rain, so one has to go.
        plain, lost, _overrun = WX.drive_route(tasks, vehicle, base, [rain] * 2)
        self.assertEqual([s.task.node_id for s in plain.stops], [1])
        self.assertEqual([t.node_id for t in lost], [2])
        kept, lost, _overrun = WX.drive_route(tasks, vehicle, base, [rain] * 2,
                                              protect=[2])
        self.assertEqual([s.task.node_id for s in kept.stops], [2])
        self.assertEqual([t.node_id for t in lost], [1])
        self.assertAlmostEqual(kept.duration_min, 80.0 / 0.88 + 4.0 + 15.0, places=9)
        # When everything fits, protection changes nothing.
        roomy = VRP.VehicleSpec(vehicle_id=1, shift_minutes=240.0)
        both, lost, _overrun = WX.drive_route(tasks, roomy, base, [rain] * 4, protect=[2])
        self.assertEqual([s.task.node_id for s in both.stops], [1, 2])
        self.assertEqual(lost, [])

    def test_a_protected_stop_that_cannot_be_reached_is_reported(self):
        base, tasks = self._line(), self._two_stops()
        vehicle = VRP.VehicleSpec(vehicle_id=1, shift_minutes=90.0)
        rain = WX.effects(WX.Conditions(rain_mm_h=20.0))
        driven, lost, _overrun = WX.drive_route(tasks, vehicle, base, [rain] * 2,
                                                protect=[2])
        # The protected stop alone needs 109.9 minutes, so it cannot be reached
        # at all.  It is lost, and the other stop is not held back for it.
        self.assertEqual([t.node_id for t in lost], [2])
        self.assertEqual([s.task.node_id for s in driven.stops], [1])

    def test_a_leg_takes_the_weather_of_the_hour_it_starts_in(self):
        base = self._line()
        task = VRP.BinTask(node_id=2, index=2, load_kg=100.0, service_minutes=4.0,
                           window_start_min=70.0)
        vehicle = VRP.VehicleSpec(vehicle_id=1, shift_minutes=240.0)
        rain = WX.effects(WX.Conditions(rain_mm_h=20.0))
        driven, _dropped, _overrun = WX.drive_route(
            [task], vehicle, base, [WX.NEUTRAL, rain, rain, rain])
        # Out in the dry first hour: 40 minutes.  Waits to minute 70, serves to
        # minute 74, and drives home in the rain of the second hour.
        self.assertAlmostEqual(driven.stops[0].arrival_min, 40.0, places=9)
        self.assertAlmostEqual(driven.duration_min, 74.0 + 40.0 / 0.88 + 15.0, places=9)

    def test_heat_lengthens_the_stop_and_not_the_drive(self):
        base, tasks = self._line(), self._two_stops()
        vehicle = VRP.VehicleSpec(vehicle_id=1, shift_minutes=240.0)
        hot = WX.effects(self.HOT)
        driven, _dropped, _overrun = WX.drive_route(tasks, vehicle, base, [hot] * 4)
        self.assertAlmostEqual(driven.duration_min,
                               80.0 + 8.0 * hot.service_factor + 15.0, places=9)

    def test_hours_that_agree_combine_to_their_common_value(self):
        rain = WX.effects(WX.Conditions(rain_mm_h=20.0))
        self.assertEqual(WX.combine_effects([rain] * 3).speed_factor, 0.88)
        mixed = WX.combine_effects([WX.NEUTRAL, rain, rain, rain])
        # Travel time adds up, so the speed factor is the harmonic mean.
        self.assertAlmostEqual(mixed.speed_factor, 4.0 / (1.0 + 3.0 / 0.88), places=12)
        self.assertIs(WX.combine_effects([]), WX.NEUTRAL)

    def test_mean_conditions_keep_the_worst_water(self):
        hours = [WX.Conditions(temperature_c=30.0, rain_mm_h=0.0),
                 WX.Conditions(temperature_c=34.0, rain_mm_h=12.0, standing_water_mm=120.0)]
        mean = WX.mean_conditions(hours)
        self.assertAlmostEqual(mean.temperature_c, 32.0)
        self.assertAlmostEqual(mean.rain_mm_h, 6.0)
        self.assertEqual(mean.standing_water_mm, 120.0)
        self.assertIs(WX.mean_conditions([]), WX.NOMINAL)

    # -- providers -----------------------------------------------------------
    DOCUMENT = {
        "currentTime": "2026-01-15T03:20:11.123456789Z",
        "isDaytime": True,
        "temperature": {"degrees": 31.5, "unit": "CELSIUS"},
        "relativeHumidity": 70,
        "uvIndex": 7,
        "precipitation": {"probability": {"percent": 40, "type": "RAIN"},
                          "qpf": {"quantity": 3.2, "unit": "MILLIMETERS"}},
    }

    def test_a_feed_document_is_read_into_conditions(self):
        c = WX.parse_google(self.DOCUMENT)
        self.assertEqual((c.temperature_c, c.relative_humidity, c.rain_mm_h, c.uv_index),
                         (31.5, 70.0, 3.2, 7.0))
        self.assertEqual(c.observed_at,
                         datetime(2026, 1, 15, 3, 20, 11, 123456, tzinfo=timezone.utc))
        self.assertEqual(c.source, "google")

    def test_imperial_units_are_converted(self):
        document = dict(self.DOCUMENT,
                        temperature={"degrees": 86.0, "unit": "FAHRENHEIT"},
                        precipitation={"qpf": {"quantity": 0.5, "unit": "INCHES"}})
        c = WX.parse_google(document)
        self.assertAlmostEqual(c.temperature_c, 30.0, places=9)
        self.assertAlmostEqual(c.rain_mm_h, 12.7, places=9)

    def test_a_document_without_temperature_is_rejected(self):
        with self.assertRaises(ValueError):
            WX.parse_google({"relativeHumidity": 50})

    def test_a_reading_is_reused_inside_the_cache_period_only(self):
        calls = []

        def fetch(url):
            calls.append(url)
            return self.DOCUMENT

        provider = WX.GoogleWeatherProvider(api_key="k", cache_seconds=600.0, fetch=fetch)
        provider.current(23.75, 90.38)
        provider.current(23.75, 90.38)
        self.assertEqual(len(calls), 1)
        self.assertEqual(provider.cache_hits, 1)

        uncached = WX.GoogleWeatherProvider(api_key="k", cache_seconds=0.0, fetch=fetch)
        uncached.current(23.75, 90.38)
        uncached.current(23.75, 90.38)
        self.assertEqual(uncached.requests, 2)
        self.assertEqual(uncached._cache, {}, "an expired reading must be deleted")

    def test_the_cache_period_cannot_exceed_one_hour(self):
        provider = WX.GoogleWeatherProvider(api_key="k", cache_seconds=86400.0,
                                            fetch=lambda url: self.DOCUMENT)
        self.assertEqual(provider.cache_seconds, 3600.0)

    def test_a_failed_request_falls_back_and_hides_the_key(self):
        def fetch(url):
            raise RuntimeError(f"HTTP 403 for {url}")

        provider = WX.GoogleWeatherProvider(api_key="SECRET-KEY", fetch=fetch)
        self.assertEqual(provider.current(23.75, 90.38), WX.NOMINAL)
        self.assertEqual(provider.fallbacks, 1)
        self.assertNotIn("SECRET-KEY", provider.last_error)
        self.assertNotIn("SECRET-KEY", json.dumps(provider.describe()))

    def test_no_key_means_no_request(self):
        def fetch(url):
            raise AssertionError("the service must not be called without a key")

        with mock.patch.dict("os.environ", {WX.KEY_VARIABLE: ""}):
            provider = WX.GoogleWeatherProvider(api_key="", fetch=fetch)
        self.assertEqual(provider.current(23.75, 90.38), WX.NOMINAL)
        self.assertEqual(provider.requests, 0)

    def test_hourly_forecast_is_read_per_hour(self):
        hour = dict(self.DOCUMENT)
        hour.pop("currentTime")
        hour["interval"] = {"startTime": "2026-01-15T04:00:00Z"}
        provider = WX.GoogleWeatherProvider(
            api_key="k", fetch=lambda url: {"forecastHours": [hour, hour, hour]})
        hours = provider.next_hours(23.75, 90.38, 2)
        self.assertEqual(len(hours), 2)
        self.assertEqual(hours[0].observed_at,
                         datetime(2026, 1, 15, 4, 0, tzinfo=timezone.utc))

    def test_open_feed_turns_an_interval_total_into_a_rate(self):
        document = {"current": {"time": "2026-01-15T03:15", "interval": 900,
                                "temperature_2m": 29.0, "relative_humidity_2m": 80,
                                "precipitation": 1.5, "uv_index": 4.0, "is_day": 1}}
        provider = WX.OpenMeteoProvider(fetch=lambda url: document)
        c = provider.current(23.75, 90.38)
        self.assertAlmostEqual(c.rain_mm_h, 6.0, places=9)
        self.assertEqual(c.source, "open-meteo")

    def test_key_is_read_from_the_environment_then_the_file(self):
        with mock.patch.dict("os.environ", {WX.KEY_VARIABLE: "from-env"}):
            self.assertEqual(WX.api_key_from_environment(), "from-env")
        import pathlib
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / ".env"
            path.write_text(f"OTHER=1\n{WX.KEY_VARIABLE}='from-file'\n", encoding="utf-8")
            with mock.patch.dict("os.environ", {WX.KEY_VARIABLE: ""}):
                self.assertEqual(WX.api_key_from_environment(path), "from-file")
                self.assertEqual(
                    WX.api_key_from_environment(pathlib.Path(folder) / "absent"), "")

    def test_default_configuration_is_nominal_weather(self):
        provider = WX.make_provider({})
        self.assertEqual(provider.current(0.0, 0.0), WX.NOMINAL)
        self.assertIsInstance(WX.make_provider({"PROVIDER": "open-meteo"}),
                              WX.OpenMeteoProvider)


def replace_conditions(conditions, **changes):
    from dataclasses import replace
    return replace(conditions, **changes)
