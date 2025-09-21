from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('bins', '0002_alter_aicost_id_alter_bingroup_id_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='sensorreading',
            name='waste_level',
            field=models.FloatField(null=True, blank=True),
        ),
    ]
