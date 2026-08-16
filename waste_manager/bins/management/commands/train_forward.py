"""
Train the forward-looking prediction bundle offline.

This is the *only* place a gradient-boosting model is fitted.  The running
application never does it, so an operator using the dashboard never pays for a
training job.
"""
from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = ("Train the time-to-overflow regressor, quantile heads and calibrated "
            "hazard classifier from stored telemetry.")

    def add_arguments(self, parser):
        parser.add_argument("--test-frac", type=float, default=0.2,
                            help="Fraction held out as the temporal test set.")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--n-iter", type=int, default=25,
                            help="Randomised-search draws for the regressor.")
        parser.add_argument("--n-splits", type=int, default=5,
                            help="GroupKFold folds for the inner cross-validation.")
        parser.add_argument("--quick", action="store_true",
                            help="Small search budget, for smoke tests and CI.")
        parser.add_argument("--no-importance", action="store_true",
                            help="Skip permutation importance (faster).")

    def handle(self, *args, **options):
        from bins.utils.ai.train_forward import train_forward

        self.stdout.write("Building the supervised dataset from stored telemetry...")
        try:
            meta = train_forward(
                test_frac=float(options["test_frac"]),
                random_state=int(options["seed"]),
                n_iter=int(options["n_iter"]),
                n_splits=int(options["n_splits"]),
                quick=bool(options["quick"]),
                compute_importance=not options["no_importance"],
            )
        except ValueError as exc:
            raise CommandError(str(exc))

        metrics = meta["metrics"]
        self.stdout.write(self.style.SUCCESS(f"\nTrained {meta['version']}"))
        self.stdout.write(f"  rows            {metrics['n_records']:,} "
                          f"({metrics['n_train']:,} train / {metrics['n_test']:,} test) "
                          f"across {metrics['n_bins']} bins")
        self.stdout.write(f"  validation      {meta['validation']}")
        self.stdout.write("")
        self.stdout.write("  Time to overflow")
        self.stdout.write(f"    test R2       {metrics['reg_r2']:.4f}   "
                          f"(mean-predictor baseline {metrics['baseline_mean_r2']:.4f})")
        self.stdout.write(f"    test MAE      {metrics['reg_mae_h']:.3f} h  "
                          f"(baseline {metrics['baseline_mean_mae_h']:.3f} h)")
        self.stdout.write(f"    grouped CV R2 {metrics['reg_cv_r2_mean']:.4f} "
                          f"+/- {metrics['reg_cv_r2_std']:.4f}")
        self.stdout.write(f"    P10-P90 cover {metrics['interval_coverage_p10_p90']:.3f} "
                          f"(nominal {metrics['interval_nominal_coverage']:.2f}), "
                          f"mean width {metrics['interval_mean_width_h']:.2f} h")

        if "hazard_roc_auc" in metrics:
            self.stdout.write("")
            self.stdout.write("  Hazard within horizon")
            self.stdout.write(f"    ROC AUC       {metrics['hazard_roc_auc']:.4f}  "
                              f"(grouped CV {metrics.get('hazard_cv_auc_mean', 0):.4f} "
                              f"+/- {metrics.get('hazard_cv_auc_std', 0):.4f})")
            self.stdout.write(f"    avg precision {metrics['hazard_average_precision']:.4f} "
                              f"(positive rate {metrics['hazard_positive_rate']:.4f})")
            self.stdout.write(f"    Brier         {metrics['hazard_brier']:.4f} "
                              f"(skill score {metrics['hazard_brier_skill_score']:.4f})")
        elif "hazard_note" in metrics:
            self.stdout.write(self.style.WARNING(f"\n  {metrics['hazard_note']}"))

        params = meta["hyperparameters"]["regressor"]
        self.stdout.write("")
        self.stdout.write("  Selected hyperparameters")
        for key in sorted(params):
            self.stdout.write(f"    {key:<20} {params[key]}")

        importance = meta.get("global_importance") or []
        if importance:
            self.stdout.write("")
            self.stdout.write("  Top features by permutation importance")
            for row in importance[:8]:
                self.stdout.write(f"    {row['feature']:<20} {row['share']:.3f}")

        if options.get("quick"):
            self.stdout.write(self.style.WARNING(
                "\n  Quick mode: reduced search budget. Re-run without --quick for "
                "the tuned configuration reported in the paper."))
