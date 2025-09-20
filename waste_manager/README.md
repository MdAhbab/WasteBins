# Smart Waste Management System

A Django-based intelligent waste management system with AI-powered route optimization and real-time sensor monitoring.

## Features

- **Real-time Sensor Monitoring**: Track temperature, humidity, and gas levels from waste bins
- **AI-Powered Cost Prediction**: Machine learning models predict collection costs
- **Optimized Route Planning**: Dijkstra algorithm with AI cost weighting for efficient collection routes
- **User Authentication**: Secure login system with user profiles and settings
- **Admin Interface**: Comprehensive data management through Django admin
- **API Endpoints**: RESTful APIs for data submission and retrieval
- **Real-time Dashboard**: Live updates with notification system

## Technology Stack

- **Backend**: Django 4.2
- **Database**: MySQL 8+
- **Machine Learning**: scikit-learn, pandas, numpy
- **Frontend**: HTML, CSS, JavaScript (Vanilla)
- **Model Storage**: joblib

## Installation

### Prerequisites

1. Python 3.8+
2. MySQL 8+
3. pip

### Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Database Setup**
   - Create MySQL database and user:
   ```sql
   CREATE DATABASE waste_manager_db;
   CREATE USER 'root'@'localhost' IDENTIFIED BY 'root';
   GRANT ALL PRIVILEGES ON waste_manager_db.* TO 'root'@'localhost';
   FLUSH PRIVILEGES;
   ```

3. **Run Migrations**
   ```bash
   python manage.py migrate
   ```

4. **Load Sample Data**
   ```bash
   python manage.py load_sample_data
   ```

5. **Create Superuser**
   ```bash
   python manage.py createsuperuser
   ```

6. **Start Development Server**
   ```bash
   python manage.py runserver
   ```

## Usage

### Dashboard
Access the main dashboard at `http://localhost:8000/dashboard/` to view:
- Real-time sensor readings
- Latest collection routes
- System health status
- Route optimization controls

### API Endpoints

- `GET /api/readings/` - Get latest sensor readings
- `POST /api/readings/submit/` - Submit new sensor data
- `POST /api/predict-cost/` - Generate AI cost predictions
- `POST /api/compute-route/` - Calculate optimized collection routes
- `POST /api/train-model/` - Train machine learning model (admin only)
- `GET /api/notifications/` - Get user notifications

### Admin Interface
Access Django admin at `http://localhost:8000/admin/` to manage:
- Bin groups and nodes
- Sensor readings
- AI cost predictions
- Collection routes
- User notifications

## Configuration

### Settings
Key configuration options in `settings.py`:

- **Database**: MySQL connection settings
- **Time Zone**: Set to 'Asia/Dhaka'
- **Model Storage**: AI models stored in `model_store/` directory
- **Static Files**: CSS/JS assets in `bins/static/`
- **Templates**: HTML templates in `bins/templates/`

### Routing Algorithm
The system uses a weighted Dijkstra algorithm that combines:
- Haversine distance between nodes
- AI-predicted collection costs
- Configurable alpha parameter for cost weighting

## Development

### Project Structure
```
waste_manager/
├── manage.py
├── requirements.txt
├── README.md
├── waste_manager/          # Django project settings
├── bins/                   # Main application
│   ├── models.py          # Database models
│   ├── views.py           # Views and API endpoints
│   ├── urls.py            # URL routing
│   ├── forms.py           # User forms
│   ├── admin.py           # Admin interface
│   ├── utils/             # Utility functions
│   │   ├── dijkstra.py    # Route optimization
│   │   └── ai/            # Machine learning
│   ├── templates/         # HTML templates
│   ├── static/            # CSS/JS assets
│   └── management/        # Custom commands
└── fixtures/              # Sample data
```

### Running Tests
```bash
python manage.py test
```

## License

This project is developed for educational and research purposes.