# Setup Guide - Smart Waste Management System

Complete installation guide for Windows and macOS.

---

## 📋 Table of Contents

- [System Requirements](#system-requirements)
- [Windows Setup](#windows-setup)
- [macOS Setup](#macos-setup)
- [Database Configuration](#database-configuration)
- [Initial Setup](#initial-setup)
- [Running the Server](#running-the-server)
- [Troubleshooting](#troubleshooting)
- [Network Access (LAN)](#network-access-lan)

---

## 💻 System Requirements

### Minimum Requirements
- **Operating System:** Windows 10+ or macOS 10.15+
- **Python:** 3.8 or higher (3.13 recommended)
- **MySQL:** 8.0 or higher
- **RAM:** 4GB minimum, 8GB recommended
- **Disk Space:** 1GB free space
- **Internet:** Required for downloading dependencies

### Software Prerequisites
- Python 3.8+
- MySQL Server 8.0+
- pip (Python package manager)
- Git (optional, for cloning)

---

## 🪟 Windows Setup

### Step 1: Install Python

1. **Download Python:**
   - Visit https://www.python.org/downloads/
   - Download Python 3.13.x for Windows
   - **Important:** Check "Add Python to PATH" during installation

2. **Verify Installation:**
   ```cmd
   python --version
   pip --version
   ```
   Should show Python 3.13.x and pip version.

### Step 2: Install MySQL

1. **Download MySQL:**
   - Visit https://dev.mysql.com/downloads/installer/
   - Download MySQL Installer for Windows
   - Choose "mysql-installer-community" version

2. **Install MySQL:**
   - Run the installer
   - Choose "Developer Default" setup type
   - Set root password (remember this!)
   - Complete installation

3. **Verify MySQL:**
   ```cmd
   mysql --version
   ```

4. **Start MySQL Service:**
   - Open Services (Win + R → `services.msc`)
   - Find "MySQL80" service
   - Right-click → Start (if not running)

### Step 3: Create Virtual Environment

1. **Navigate to project directory:**
   ```cmd
   cd C:\path\to\Wastebins
   ```

2. **Create virtual environment:**
   ```cmd
   python -m venv venv
   ```

3. **Activate virtual environment:**
   ```cmd
   venv\Scripts\activate
   ```
   You should see `(venv)` prefix in your command prompt.

### Step 4: Install Dependencies

1. **Upgrade pip and build tools:**
   ```cmd
   python -m pip install --upgrade pip setuptools wheel
   ```

2. **Install project dependencies:**
   ```cmd
   cd waste_manager
   pip install -r requirements.txt
   ```

3. **If scikit-learn fails:**
   ```cmd
   pip install scikit-learn --no-cache-dir
   ```

### Step 5: Install mysqlclient (Windows-specific)

**Option A: Using wheel file (Recommended)**

1. Download the appropriate `.whl` file for your Python version from:
   https://www.lfd.uci.edu/~gohlke/pythonlibs/#mysqlclient

2. Install:
   ```cmd
   pip install mysqlclient-2.2.4-cp313-cp313-win_amd64.whl
   ```

**Option B: Using Microsoft C++ Build Tools**

1. Download Visual Studio Build Tools:
   https://visualstudio.microsoft.com/visual-cpp-build-tools/

2. Install "Desktop development with C++"

3. Install mysqlclient:
   ```cmd
   pip install mysqlclient
   ```

---

## 🍎 macOS Setup

### Step 1: Install Homebrew (if not installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install Python

1. **Install Python via Homebrew:**
   ```bash
   brew install python@3.13
   ```

2. **Verify installation:**
   ```bash
   python3 --version
   pip3 --version
   ```

### Step 3: Install MySQL

1. **Install MySQL:**
   ```bash
   brew install mysql
   ```

2. **Start MySQL service:**
   ```bash
   brew services start mysql
   ```

3. **Secure MySQL installation:**
   ```bash
   mysql_secure_installation
   ```
   - Set root password
   - Remove anonymous users: Yes
   - Disallow root login remotely: Yes
   - Remove test database: Yes
   - Reload privilege tables: Yes

4. **Verify MySQL:**
   ```bash
   mysql --version
   ```

### Step 4: Create Virtual Environment

1. **Navigate to project directory:**
   ```bash
   cd /Users/yourusername/Downloads/MicroLab/Wastebins
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   ```

3. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```
   You should see `(venv)` prefix in your terminal.

### Step 5: Install Dependencies

1. **Upgrade pip:**
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. **Install MySQL development files:**
   ```bash
   brew install mysql-client pkg-config
   ```

3. **Set environment variables for mysqlclient:**
   ```bash
   export PATH="/opt/homebrew/opt/mysql-client/bin:$PATH"
   export LDFLAGS="-L/opt/homebrew/opt/mysql-client/lib"
   export CPPFLAGS="-I/opt/homebrew/opt/mysql-client/include"
   export PKG_CONFIG_PATH="/opt/homebrew/opt/mysql-client/lib/pkgconfig"
   ```

4. **Install project dependencies:**
   ```bash
   cd waste_manager
   pip install -r requirements.txt
   ```

---

## 🗄️ Database Configuration

### Step 1: Create Database

**Windows:**
```cmd
mysql -u root -p
```

**macOS:**
```bash
mysql -u root -p
```

Enter your MySQL root password, then run:

```sql
CREATE DATABASE waste_manager_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'root'@'localhost' IDENTIFIED BY 'root';
GRANT ALL PRIVILEGES ON waste_manager_db.* TO 'root'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

**Note:** For production, use a strong password and different username.

### Step 2: Configure Django Settings (Optional)

If you need custom database settings, create `local_settings.py`:

**Windows:**
```cmd
cd waste_manager\waste_manager
copy local_settings.py.example local_settings.py
notepad local_settings.py
```

**macOS:**
```bash
cd waste_manager/waste_manager
cp local_settings.py.example local_settings.py
nano local_settings.py
```

Edit the database configuration if needed:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'waste_manager_db',
        'USER': 'your_username',
        'PASSWORD': 'your_password',
        'HOST': 'localhost',
        'PORT': '3306',
    }
}
```

---

## 🚀 Initial Setup

### Step 1: Run Migrations

**Windows:**
```cmd
cd waste_manager
python manage.py migrate
```

**macOS:**
```bash
cd waste_manager
python manage.py migrate
```

Expected output:
```
Operations to perform:
  Apply all migrations: admin, auth, bins, contenttypes, sessions
Running migrations:
  Applying bins.0001_initial... OK
  Applying bins.0002_alter_aicost_id_alter_bingroup_id... OK
  ...
```

### Step 2: Create Superuser

**Windows/macOS:**
```bash
python manage.py createsuperuser
```

Enter:
- Username: (your choice, e.g., `admin`)
- Email: (your email)
- Password: (strong password)
- Password confirmation: (same password)

### Step 3: Load Sample Data (Optional)

Load sample nodes and sensor readings for testing:

```bash
python manage.py load_sample_data
```

This creates:
- 5 waste bin nodes with GPS coordinates
- 20 sensor readings with realistic data
- Sample for testing AI model and routing

### Step 4: System Health Check

```bash
python manage.py check_system
```

Verifies:
- ✓ Database connectivity
- ✓ Sample data loaded
- ✓ Node count
- ✓ Sensor readings count

---

## 🌐 Running the Server

### Local Access Only

**Windows:**
```cmd
python manage.py runserver
```

**macOS:**
```bash
python manage.py runserver
```

Access at: http://localhost:8000/

### LAN Access (All devices on network)

**Windows:**
```cmd
python manage.py runserver 0.0.0.0:8000
```

**macOS:**
```bash
python manage.py runserver 0.0.0.0:8000
```

Find your IP address:

**Windows:**
```cmd
ipconfig
```
Look for "IPv4 Address" (e.g., 192.168.1.100)

**macOS:**
```bash
ifconfig | grep "inet "
```
Look for your local IP (e.g., 192.168.1.100)

Access from other devices: http://192.168.1.100:8000/

### Stop the Server

Press `Ctrl + C` in the terminal.

---

## 🧪 Testing the System

### 1. Access Dashboard

Visit: http://localhost:8000/dashboard/

**First-time users:**
- Click "Sign Up" to create an account
- Or use superuser credentials from Step 2

### 2. View Admin Interface

Visit: http://localhost:8000/admin/

Login with superuser credentials.

### 3. Test API Endpoints

**Using curl (Windows/macOS):**

```bash
# Get latest readings
curl http://localhost:8000/api/readings/?limit=5

# Submit sensor data
curl -X POST http://localhost:8000/api/readings/submit/ \
  -H "Content-Type: application/json" \
  -d "{\"node_id\": 1, \"temperature\": 28.5, \"humidity\": 65.0, \"gas_level\": 0.45, \"waste_level\": 0.78}"
```

**Using Postman:**
1. Download Postman: https://www.postman.com/downloads/
2. Import the API collection (if provided)
3. Test endpoints individually

### 4. Train AI Model

```bash
python manage.py shell
```

In the Python shell:
```python
from bins.utils.ai.train_model import train_from_db
result = train_from_db()
print(f"Model trained! R² = {result['validation_metrics']['r2']:.3f}")
exit()
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError: No module named 'MySQLdb'"

**Solution:**
```bash
pip install mysqlclient
```

If fails on Windows, use wheel file method (see Windows Step 5).

#### 2. "Access denied for user 'root'@'localhost'"

**Solution:**
```sql
mysql -u root -p
ALTER USER 'root'@'localhost' IDENTIFIED BY 'root';
FLUSH PRIVILEGES;
```

#### 3. "django.db.utils.OperationalError: (2003, "Can't connect to MySQL server")"

**Solution:**

**Windows:** Start MySQL service via Services app

**macOS:**
```bash
brew services restart mysql
```

#### 4. "pip install fails with 'error: Microsoft Visual C++ 14.0 is required'"

**Solution (Windows):**
1. Install Visual Studio Build Tools
2. Or use pre-compiled wheel files

#### 5. Virtual environment activation fails

**Windows (PowerShell restriction):**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\activate
```

**macOS (permission denied):**
```bash
chmod +x venv/bin/activate
source venv/bin/activate
```

#### 6. "Port 8000 already in use"

**Solution:**

**Windows:**
```cmd
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F
```

**macOS:**
```bash
lsof -ti:8000 | xargs kill -9
```

Or use different port:
```bash
python manage.py runserver 8001
```

#### 7. Static files not loading

**Solution:**
```bash
python manage.py collectstatic
```

#### 8. "ImportError: No module named 'sklearn'"

**Solution:**
```bash
pip install scikit-learn
```

Note: Package is `scikit-learn`, import is `sklearn`.

#### 9. Database migration conflicts

**Solution:**
```bash
python manage.py migrate --fake bins zero
python manage.py migrate bins
```

---

## 🌐 Network Access (LAN)

### Enable LAN Access

1. **Update settings (already configured):**
   - `ALLOWED_HOSTS = ['*']` in `settings.py`

2. **Run server on all interfaces:**
   ```bash
   python manage.py runserver 0.0.0.0:8000
   ```

3. **Find your local IP:**

   **Windows:**
   ```cmd
   ipconfig
   ```

   **macOS:**
   ```bash
   ifconfig | grep "inet " | grep -v 127.0.0.1
   ```

4. **Access from other devices:**
   ```
   http://YOUR_IP:8000/
   ```
   Example: http://192.168.1.100:8000/

### Firewall Configuration

**Windows:**
1. Open Windows Defender Firewall
2. Click "Advanced settings"
3. Inbound Rules → New Rule
4. Port → TCP → 8000
5. Allow the connection
6. Apply to all profiles
7. Name: "Django Dev Server"

**macOS:**
1. System Preferences → Security & Privacy
2. Firewall → Firewall Options
3. Add Python app
4. Allow incoming connections

---

## 📱 Mobile/Tablet Access

Once LAN access is enabled:

1. Connect mobile device to same WiFi network
2. Open browser on mobile
3. Navigate to: http://YOUR_COMPUTER_IP:8000/
4. Login with your credentials
5. Test location-based routing features

---

## 🔄 Updating the System

### Pull Latest Changes (if using Git)

```bash
git pull origin main
```

### Update Dependencies

```bash
pip install -r requirements.txt --upgrade
```

### Run New Migrations

```bash
python manage.py migrate
```

### Collect Static Files

```bash
python manage.py collectstatic --noinput
```

---

## 🛑 Deactivating Virtual Environment

When done working:

**Windows:**
```cmd
deactivate
```

**macOS:**
```bash
deactivate
```

---

## 📝 Daily Workflow

### Starting Work

**Windows:**
```cmd
cd C:\path\to\Wastebins
venv\Scripts\activate
cd waste_manager
python manage.py runserver 0.0.0.0:8000
```

**macOS:**
```bash
cd /path/to/Wastebins
source venv/bin/activate
cd waste_manager
python manage.py runserver 0.0.0.0:8000
```

### Stopping Work

1. Press `Ctrl + C` to stop server
2. Type `deactivate` to exit virtual environment
3. Close terminal

---

## 🆘 Getting Help

### Check System Status

```bash
python manage.py check
python manage.py check_system
```

### View Django Debug Info

Set `DEBUG = True` in `settings.py` (already enabled by default).

Errors will show detailed traceback in browser.

### Check Logs

Look at terminal output for error messages.

### Database Console

```bash
python manage.py dbshell
```

Access MySQL directly for debugging.

---

## ✅ Verification Checklist

After setup, verify these work:

- [ ] `python --version` shows 3.8+
- [ ] `mysql --version` shows 8.0+
- [ ] Virtual environment activates
- [ ] `pip list` shows all dependencies
- [ ] Database migrations complete
- [ ] Superuser created
- [ ] Server starts without errors
- [ ] Dashboard loads at http://localhost:8000/dashboard/
- [ ] Can login with superuser
- [ ] Admin panel accessible at /admin/
- [ ] Sample data loaded (if desired)
- [ ] LAN access works (if configured)

---

## 🎯 Quick Start Commands

### Complete Setup (Fresh Install)

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
cd waste_manager
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py load_sample_data
python manage.py runserver 0.0.0.0:8000
```

**macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
cd waste_manager
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py load_sample_data
python manage.py runserver 0.0.0.0:8000
```

---

## 📚 Next Steps

After successful setup:

1. **Read the main README.md** for system architecture and algorithms
2. **Train the AI model** with your sensor data
3. **Test API endpoints** using Postman or curl
4. **Integrate IoT sensors** for real sensor data
5. **Customize priority weights** for your use case
6. **Set up production deployment** (see DEPLOYMENT.md if available)

---

**Setup complete! 🎉**

For detailed system information, algorithms, and usage, see [README.md](README.md)

