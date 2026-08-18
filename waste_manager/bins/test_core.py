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
from wastebins_core import ledger as LG
from wastebins_core import scenario as SC
from wastebins_core import stats as ST
from wastebins_core import traffic as TR
from wastebins_core import vrp as VRP


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
