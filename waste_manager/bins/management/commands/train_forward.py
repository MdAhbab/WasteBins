"""
Management command: train the forward-looking (non-circular) priority model.

    python manage.py train_forward

Builds forward labels (time-to-overflow, hazard-within-6h) from the stored
SensorReading history, trains a histogram gradient-boosting regressor + P10/P50/P90
quantile models + a calibrated hazard classifier under a temporal split, and
saves the bundle to the model store.
"""
from django.core.management.base import BaseCommand
from bins.utils.ai.train_forward import train_forward


class Command(BaseCommand):
    help = "Train the forward-looking priority model (time-to-overflow + hazard)."

    def add_arguments(self, parser):
        parser.add_argument("--test-frac", type=float, default=0.2,
                            help="Fraction of the latest data held out for temporal validation.")

    def handle(self, *args, **opts):
        self.stdout.write("Training forward-looking model ...")
        try:
            meta = train_forward(test_frac=opts["test_frac"])
        except Exception as e:  # pragma: no cover - surfaced to operator
            self.stderr.write(self.style.ERROR(f"Training failed: {e}"))
            return
        m = meta["metrics"]
        self.stdout.write(self.style.SUCCESS(
            f"Saved {meta['version']}  |  n={m['n_records']}  "
            f"time-to-overflow R2={m['reg_r2']:.3f} (MAE {m['reg_mae_h']:.2f} h)"))
        if "hazard_roc_auc" in m:
            self.stdout.write(self.style.SUCCESS(
                f"hazard ROC-AUC={m['hazard_roc_auc']:.3f}  "
                f"AP={m['hazard_avg_precision']:.3f}  Brier={m['hazard_brier']:.3f}"))
