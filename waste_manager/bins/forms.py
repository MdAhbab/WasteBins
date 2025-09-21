from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import UserSetting

class SignupForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')

class ProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'email')

class SettingsForm(forms.ModelForm):
    latitude = forms.FloatField(
        required=False,
        widget=forms.NumberInput(attrs={
            'step': 'any',
            'placeholder': 'e.g., 23.7806',
            'class': 'form-control',
            'id': 'id_latitude'
        }),
        help_text='Your latitude coordinate (GPS location)'
    )
    longitude = forms.FloatField(
        required=False,
        widget=forms.NumberInput(attrs={
            'step': 'any',
            'placeholder': 'e.g., 90.2794',
            'class': 'form-control',
            'id': 'id_longitude'
        }),
        help_text='Your longitude coordinate (GPS location)'
    )
    location_name = forms.CharField(
        required=False,
        max_length=255,
        widget=forms.TextInput(attrs={
            'placeholder': 'e.g., Dhaka University, Dhaka',
            'class': 'form-control'
        }),
        help_text='Optional: Name or address of your location'
    )
    
    class Meta:
        model = UserSetting
        fields = (
            'notify_email', 
            'polling_interval_sec', 
            'latitude', 
            'longitude', 
            'location_name',
            'auto_update_location'
        )
        widgets = {
            'notify_email': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'polling_interval_sec': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 1,
                'max': 300
            }),
            'auto_update_location': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
        
    def clean(self):
        cleaned_data = super().clean()
        latitude = cleaned_data.get('latitude')
        longitude = cleaned_data.get('longitude')
        
        # Validate coordinate ranges
        if latitude is not None:
            if not (-90 <= latitude <= 90):
                raise forms.ValidationError("Latitude must be between -90 and 90 degrees")
                
        if longitude is not None:
            if not (-180 <= longitude <= 180):
                raise forms.ValidationError("Longitude must be between -180 and 180 degrees")
        
        # Both coordinates should be provided together
        if (latitude is None) != (longitude is None):
            raise forms.ValidationError("Please provide both latitude and longitude, or leave both empty")
            
        return cleaned_data

class LocationForm(forms.Form):
    """Standalone form for quick location updates on dashboard"""
    latitude = forms.FloatField(
        widget=forms.NumberInput(attrs={
            'step': 'any',
            'placeholder': 'Latitude',
            'class': 'form-control form-control-sm',
            'id': 'quick_latitude'
        })
    )
    longitude = forms.FloatField(
        widget=forms.NumberInput(attrs={
            'step': 'any', 
            'placeholder': 'Longitude',
            'class': 'form-control form-control-sm',
            'id': 'quick_longitude'
        })
    )
    
    def clean(self):
        cleaned_data = super().clean()
        latitude = cleaned_data.get('latitude')
        longitude = cleaned_data.get('longitude')
        
        if latitude is not None and not (-90 <= latitude <= 90):
            raise forms.ValidationError("Latitude must be between -90 and 90 degrees")
            
        if longitude is not None and not (-180 <= longitude <= 180):
            raise forms.ValidationError("Longitude must be between -180 and 180 degrees")
            
        return cleaned_data