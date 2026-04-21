import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'waste_manager.settings')
django.setup()

from bins.models import Node

def relocate():
    # 1. Cleanup N-series
    print("Deleting N-series nodes...")
    deleted, _ = Node.objects.filter(name__startswith='N').delete()
    print(f"  Deleted {deleted} nodes.")

    # 2. Update specific bins to Mirpur
    updates = [
        ("Bin-A (TSC Gate)",       "Bin-A (Mirpur 10)",      23.8103, 90.3644),
        ("Bin-B (Arts Faculty)",   "Bin-B (Sony Square)",    23.7910, 90.3550),
        ("Bin-C (Central Mosque)", "Bin-C (Mirpur Stadium)", 23.8050, 90.3630),
        ("Bin-D (Library Road)",   "Bin-D (Mirpur 11)",      23.8203, 90.3650),
        ("Bin-E (Curzon Hall)",    "Bin-E (Mirpur 14)",      23.8100, 90.3780),
        ("Bin-F (Shahbag)",        "Bin-F (Pallabi)",        23.8340, 90.3650),
    ]

    for old_name, new_name, lat, lng in updates:
        try:
            node = Node.objects.get(name=old_name)
            node.name = new_name
            node.latitude = lat
            node.longitude = lng
            node.save()
            print(f"  Updated '{old_name}' -> '{new_name}'")
        except Node.DoesNotExist:
            # Maybe it's already updated or has a different name
            try:
                node = Node.objects.get(name=new_name)
                node.latitude = lat
                node.longitude = lng
                node.save()
                print(f"  Refreshed coordinates for '{new_name}'")
            except Node.DoesNotExist:
                # Create if it doesn't exist
                Node.objects.create(name=new_name, latitude=lat, longitude=lng)
                print(f"  Created new node '{new_name}'")

    print("\nRelocation complete.")

if __name__ == "__main__":
    relocate()
