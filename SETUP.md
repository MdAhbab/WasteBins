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
- Python 3.13 (or 3.8+)
- MySQL Server / MariaDB 8.0+
- Node.js & npm
- Git (optional)

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

### Step 3: Run the Setup Orchestrator (New)

The easiest way to bootstrap the entire environment (Backend, Frontend, and Dummy Simulation) is via the unified setup script. This script automatically:
- Creates a virtual environment (`.venv`).
- Installs Python dependencies (`pip install`).
- Installs Frontend dependencies (`npm install`).
- Executes database migrations.
- Bootstraps the Mirpur bin location data.
- Launches all services in parallel.

1. **Open a terminal in the project root directory**.
2. **Run the script**:
   ```cmd
   python run_setup.py
   ```
3. That's it! Access the dashboard at `http://localhost:5173/login`.

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

### Step 4: Run the Setup Orchestrator (New)

Just like on Windows, you can automate your setup:

1. **Navigate to project directory:**
   ```bash
   cd /Users/yourusername/Downloads/MicroLab/Wastebins
   ```

2. **Run the script:**
   ```bash
   python3 run_setup.py
   ```

The script will automatically configure your `.venv`, install all Python/Node packages, run migrations, and launch the platform at `http://localhost:5173/login`.

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

### Step 3: Run Setup Script

The `run_setup.py` script automatically runs `manage.py migrate` and initializes the Mirpur operational nodes for you. It replaces manual data initialization.

---

## 🌐 Running the Server

### Using the Orchestrator (Zero-Config)

**Windows & macOS:**
```cmd
python run_setup.py
```
This single command spins up three services simultaneously:
- Django Backend
- React / Vite Frontend
- Mirpur Dummy Data Sender

Access the dashboard at: http://localhost:5173/login

### Stop the Servers
Press `Ctrl + C` in the orchestrator terminal. It will cleanly shut down all subprocesses.

---

## 🧪 Testing the System

### 1. Access Dashboard

Visit: http://localhost:5173/login

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

*(Note: `mysqlclient` and Visual Studio Build Tools are no longer required, as the project now uses pure-Python `pymysql`.)*

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

From the project root:
```cmd
python run_setup.py
```

### Stopping Work

1. Press `Ctrl + C` in the orchestrator terminal.
2. Close terminal.

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

**Windows & macOS:**
Just clone the repo, ensure MySQL is running, and execute:
```cmd
python run_setup.py
```
*(The script will detect missing virtual environments, run NPM/PIP installs, make migrations, and launch!).*

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

