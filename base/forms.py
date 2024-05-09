from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from .models import Profile
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserChangeForm
from django.contrib.auth.forms import PasswordChangeForm
from django.utils.translation import gettext_lazy as _

class ImageUploadForm(forms.Form):
    image = forms.ImageField()



class ChangeUsernameForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['username']
        labels = {
            'username': _('Nazwa użytkownika'),
        }
        help_texts = {
            'username': None,  # Usunięcie domyślnego komunikatu help_text
        }
        error_messages = {
            'username': {
                'required': _("To pole jest wymagane"),
                'max_length': _("Nazwa użytkownika może zawierać maksymalnie %(max)d znaków"),
                'invalid': _("Nazwa użytkownika może zawierać tylko litery, cyfry oraz znaki @/./+/-/_"),
            },
        }



class CustomPasswordChangeForm(PasswordChangeForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['old_password'].label = _('Stare hasło')
        self.fields['new_password1'].label = _('Nowe hasło')
        self.fields['new_password2'].label = _('Potwierdź nowe hasło')
        self.fields['new_password1'].help_text = None  # Usunięcie domyślnego komunikatu help_text
        self.fields['new_password2'].help_text = None  # Usunięcie domyślnego komunikatu help_text
        self.fields['new_password1'].widget.attrs['autocomplete'] = 'new-password'  # Wyłączenie automatycznego uzupełniania w przeglądarkach

    error_messages = {
        'password_mismatch': _('Hasła nie pasują do siebie.'),
        'password_incorrect': _('Podane stare hasło jest nieprawidłowe.'),
        'password_too_similar': _('Twoje hasło nie może być zbyt podobne do innych Twoich danych osobowych.'),
        'password_too_short': _('Twoje hasło musi mieć co najmniej 8 znaków.'),
        'password_common': _('Twoje hasło nie może być powszechnie używanym hasłem.'),
        'password_entirely_numeric': _('Twoje hasło nie może składać się wyłącznie z cyfr.'),
    }