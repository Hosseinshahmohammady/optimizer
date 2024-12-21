from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User

class SignUpForm(UserCreationForm):
    first_name = forms.CharField(max_length=100, help_text= 'First Name')
    last_name = forms.CharField(max_length=100, help_text= 'Last Name')
    email = forms.EmailField(max_length=150, help_text= 'Email')

    class Meta :
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')


class LoginForm(AuthenticationForm):
    username = forms.CharField(label='Username or Email') 

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if '@' in username :
            try:
                user = User.objects.get(email=username)
                return user.username
            except User.DoesNotExist:
                raise forms.ValidationError("No user found with this email.")
        return username