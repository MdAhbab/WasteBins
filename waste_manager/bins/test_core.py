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

import math

import numpy as np
from django.test import SimpleTestCase

from wastebins_core import continual as CL
from wastebins_core import emissions as EM
from wastebins_core import faults as F
from wastebins_core import ledger as LG
from wastebins_core import stats as ST


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
