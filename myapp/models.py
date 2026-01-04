from django.db import models

class Violation(models.Model):
    VIOLATION_TYPES = [
        ("no_helmet", "No Helmet"),
        ("overload", "Overload"),
    ]
    violation_type = models.CharField(max_length=20, choices=VIOLATION_TYPES)
    plate_text = models.CharField(max_length=32, blank=True)
    confidence = models.FloatField(default=0.0)
    timestamp = models.DateTimeField(auto_now_add=True)
    snapshot = models.ImageField(upload_to="snapshots/", blank=True, null=True)
    meta = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        return f"{self.get_violation_type_display()} @ {self.timestamp:%Y-%m-%d %H:%M:%S}"