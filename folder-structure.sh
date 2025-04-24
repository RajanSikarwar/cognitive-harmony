cognitive_harmony/
│
├── app.py              # Main Flask application
├── config.py           # Configuration settings
├── requirements.txt    # Project dependencies
├── run.py              # Application entry point
│
├── static/             # Static files
│   ├── css/
│   │   ├── main.css
│   │   └── responsive.css
│   ├── js/
│   │   ├── main.js
│   │   └── form-validation.js
│   └── images/
│       └── logo.svg
│
├── templates/          # HTML templates
│   ├── base.html       # Base template with common elements
│   ├── index.html      # Landing page
│   ├── register.html   # Registration page
│   ├── login.html      # Login page
│   ├── dashboard.html  # User dashboard
│   └── components/     # Reusable template components
│       ├── navbar.html
│       └── footer.html
│
├── models/             # Database models
│   ├── __init__.py
│   ├── user.py
│   ├── journal.py
│   └── mood.py
│
├── controllers/        # Route controllers
│   ├── __init__.py
│   ├── auth.py         # Authentication routes
│   ├── dashboard.py    # Dashboard routes
│   └── api.py          # API endpoints
│
└── utils/              # Utility functions
    ├── __init__.py
    ├── db.py           # Database utilities
    └── security.py     # Security utilities
