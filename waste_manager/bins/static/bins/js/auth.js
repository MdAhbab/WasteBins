document.addEventListener('submit', (e) => {
  const form = e.target;
  const btn = form.querySelector('button[type=submit]');
  
  if (btn) {
    btn.disabled = true;
    
    // Show loading state
    const btnText = btn.querySelector('.btn-text');
    const btnLoading = btn.querySelector('.btn-loading');
    
    if (btnText && btnLoading) {
      btnText.style.display = 'none';
      btnLoading.style.display = 'inline';
    } else {
      btn.dataset.originalText = btn.textContent;
      btn.textContent = 'Working...';
    }
    
    // Re-enable button after timeout (fallback)
    setTimeout(() => {
      btn.disabled = false;
      if (btnText && btnLoading) {
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
      } else if (btn.dataset.originalText) {
        btn.textContent = btn.dataset.originalText;
      }
    }, 10000);
  }
});

// Enhanced form validation and UX
document.addEventListener('DOMContentLoaded', function() {
  // Add focus/blur animations to form inputs
  const inputs = document.querySelectorAll('.form-input');
  inputs.forEach(input => {
    input.addEventListener('focus', function() {
      this.parentElement.classList.add('focused');
    });
    
    input.addEventListener('blur', function() {
      this.parentElement.classList.remove('focused');
      if (this.value) {
        this.parentElement.classList.add('filled');
      } else {
        this.parentElement.classList.remove('filled');
      }
    });
    
    // Initialize filled state
    if (input.value) {
      input.parentElement.classList.add('filled');
    }
  });
  
  // Enhanced password visibility toggle
  const passwordInputs = document.querySelectorAll('input[type="password"]');
  passwordInputs.forEach(input => {
    const wrapper = input.parentElement;
    const toggleBtn = document.createElement('button');
    toggleBtn.type = 'button';
    toggleBtn.className = 'password-toggle';
    toggleBtn.innerHTML = '👁️';
    toggleBtn.title = 'Show password';
    
    wrapper.style.position = 'relative';
    wrapper.appendChild(toggleBtn);
    
    toggleBtn.addEventListener('click', function() {
      if (input.type === 'password') {
        input.type = 'text';
        this.innerHTML = '🙈';
        this.title = 'Hide password';
      } else {
        input.type = 'password';
        this.innerHTML = '👁️';
        this.title = 'Show password';
      }
    });
  });
  
  // Form validation feedback
  const forms = document.querySelectorAll('form');
  forms.forEach(form => {
    form.addEventListener('submit', function(e) {
      const requiredInputs = form.querySelectorAll('input[required]');
      let isValid = true;
      
      requiredInputs.forEach(input => {
        if (!input.value.trim()) {
          isValid = false;
          input.classList.add('error');
          showValidationMessage(input, 'This field is required');
        } else {
          input.classList.remove('error');
          hideValidationMessage(input);
        }
      });
      
      if (!isValid) {
        e.preventDefault();
      }
    });
  });
});

function showValidationMessage(input, message) {
  hideValidationMessage(input); // Remove existing message
  
  const errorDiv = document.createElement('div');
  errorDiv.className = 'validation-error';
  errorDiv.textContent = message;
  
  input.parentElement.appendChild(errorDiv);
}

function hideValidationMessage(input) {
  const existingError = input.parentElement.querySelector('.validation-error');
  if (existingError) {
    existingError.remove();
  }
}