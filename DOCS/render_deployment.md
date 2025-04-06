# Deploying Sentiment Analysis Application to Render

This guide explains how to deploy the Sentiment Analysis application to Render, a cloud platform for web applications.

## Prerequisites

- A Render account (sign up at [render.com](https://render.com))
- Git repository with your Sentiment Analysis application code

## Deployment Overview

The application now consists of seven components that need to be deployed separately:

1. **Mock API**: Provides simulated incident data
2. **Model API**: Serves sentiment predictions
3. **Main Dashboard**: Visualizes the sentiment analysis
4. **Social Media Batch Processor**: Processes social media data in batches
5. **Social Media Authentication UI**: Manages social media platform authentication
6. **Social Media Data Visualization**: Displays social media sentiment analysis
7. **CSV Data Upload Page**: Allows uploading social media data via CSV

## Step 1: Prepare Your Repository

Ensure your repository includes these required files:

- `requirements.txt`: Lists all dependencies
- `render.yaml`: Defines the Render Blueprint (see below)

Update your `requirements.txt` to include all necessary packages:

```
streamlit>=1.24.0
pandas>=1.5.3
numpy>=1.24.2
scikit-learn>=1.2.2
matplotlib>=3.7.1
altair>=5.0.1
requests>=2.28.2
sqlite3>=3.41.2
python-dotenv>=1.0.0
```

Create a `render.yaml` file in your root directory:

```yaml
services:
  # Mock API Service
  - name: sentiment-mock-api
    type: web
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app/api/mock_api.py
    envVars:
      - key: API_PORT
        value: 10000
      - key: PYTHON_VERSION
        value: 3.9.0

  # Model API Service
  - name: sentiment-model-api
    type: web
    env: python
    buildCommand: pip install -r requirements.txt && python app/models/train_model.py
    startCommand: python app/api/model_api.py
    envVars:
      - key: API_PORT
        value: 10001
      - key: PYTHON_VERSION
        value: 3.9.0
      - key: API_HOST
        fromService:
          name: sentiment-mock-api
          type: web
          property: host

  # Main Dashboard Service
  - name: sentiment-dashboard
    type: web
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app/dashboard/dashboard.py
    envVars:
      - key: API_HOST
        fromService:
          name: sentiment-mock-api
          type: web
          property: host
      - key: MODEL_API_HOST
        fromService:
          name: sentiment-model-api
          type: web
          property: host
      - key: PYTHON_VERSION
        value: 3.9.0

  # Social Media Batch Processor
  - name: sentiment-social-processor
    type: worker
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app/social_media/data_processor.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.0
      - key: MODEL_API_HOST
        fromService:
          name: sentiment-model-api
          type: web
          property: host

  # Social Media Authentication UI
  - name: sentiment-social-auth
    type: web
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app/social_media/social_media_auth.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.0
      - key: PORT
        value: 10002

  # Social Media Data Visualization
  - name: sentiment-social-data
    type: web
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app/social_media/social_media_data.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.0
      - key: PORT
        value: 10003
      - key: MODEL_API_HOST
        fromService:
          name: sentiment-model-api
          type: web
          property: host

  # CSV Data Upload Page
  - name: sentiment-csv-upload
    type: web
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app/social_media/csv_data_upload.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.0
      - key: PORT
        value: 10004
      - key: MODEL_API_HOST
        fromService:
          name: sentiment-model-api
          type: web
          property: host
```

## Step 2: Set Up a Shared Database

Since multiple services need to access the same databases, we need to set up a shared database. Render provides a PostgreSQL database service:

1. Log in to your Render account
2. Go to the Dashboard and click "New"
3. Select "PostgreSQL"
4. Set a name (e.g., "sentiment-analysis-db")
5. Choose an appropriate plan
6. Click "Create Database"

Once created, note the following connection details:
- Internal Database URL
- External Database URL
- Database Name
- Username
- Password

Update your `render.yaml` to reference this database for all services:

```yaml
# Add this to each service that needs database access
envVars:
  - key: DATABASE_URL
    fromDatabase:
      name: sentiment-analysis-db
      property: connectionString
```

## Step 3: Deploy Using Render Blueprint

1. Log in to your Render account
2. Go to the Dashboard and click "New"
3. Select "Blueprint"
4. Connect your Git repository
5. Render will detect the `render.yaml` file and set up the services
6. Click "Apply" to create and deploy all services

## Step 4: Configure Environment Variables

After deployment, verify and update these environment variables in each service:

- `API_HOST`: URL of the Mock API service
- `API_PORT`: Port for each service
- `MODEL_API_URL`: Full URL of the Model API service
- `DATABASE_URL`: Connection string to your PostgreSQL database

## Step 5: Initialize the Databases

Both the main database and social media database need to be initialized:

### Initialize Main Database

1. In your Render dashboard, go to the Model API service
2. Click "Shell" to open a terminal
3. Run: `python -c "from app.data.database import init_db; init_db()"`
4. Then run: `python app/data/data_ingestion.py --count 50`

### Initialize Social Media Database

1. Go to the Social Media Batch Processor service
2. Click "Shell" to open a terminal
3. Run: `python -c "from app.social_media.database import init_db; init_db()"`

## Step 6: Set Up Database Migration

To adapt the application to use PostgreSQL instead of SQLite:

1. Add this script to your repository as `scripts/migrate_to_postgres.py`:

```python
"""
Database Migration Script: SQLite to PostgreSQL
This script migrates data from SQLite to PostgreSQL for the Render deployment.
"""
import os
import sqlite3
import psycopg2
from psycopg2.extras import execute_values
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import database connections
from app.data.database import get_db_connection as get_incidents_conn
from app.social_media.database import get_db_connection as get_social_conn

def migrate_incidents_db(pg_conn_string):
    """Migrate incidents database to PostgreSQL"""
    print("Migrating incidents database...")
    
    # Connect to SQLite
    sqlite_conn = get_incidents_conn()
    sqlite_cursor = sqlite_conn.cursor()
    
    # Connect to PostgreSQL
    pg_conn = psycopg2.connect(pg_conn_string)
    pg_cursor = pg_conn.cursor()
    
    # Create tables in PostgreSQL
    pg_cursor.execute('''
    CREATE TABLE IF NOT EXISTS incidents (
        id SERIAL PRIMARY KEY,
        incident_id TEXT UNIQUE NOT NULL,
        report TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        sentiment INTEGER,
        confidence REAL,
        model_used TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Get data from SQLite
    sqlite_cursor.execute("SELECT incident_id, report, timestamp, sentiment, confidence, model_used, created_at FROM incidents")
    rows = sqlite_cursor.fetchall()
    
    if rows:
        # Insert data into PostgreSQL
        execute_values(
            pg_cursor,
            "INSERT INTO incidents (incident_id, report, timestamp, sentiment, confidence, model_used, created_at) VALUES %s ON CONFLICT (incident_id) DO NOTHING",
            rows
        )
    
    # Commit and close
    pg_conn.commit()
    sqlite_conn.close()
    pg_conn.close()
    
    print(f"Migrated {len(rows)} incident records")

def migrate_social_media_db(pg_conn_string):
    """Migrate social media database to PostgreSQL"""
    print("Migrating social media database...")
    
    # Connect to SQLite
    sqlite_conn = get_social_conn()
    sqlite_cursor = sqlite_conn.cursor()
    
    # Connect to PostgreSQL
    pg_conn = psycopg2.connect(pg_conn_string)
    pg_cursor = pg_conn.cursor()
    
    # Create tables in PostgreSQL
    # social_media_accounts table
    pg_cursor.execute('''
    CREATE TABLE IF NOT EXISTS social_media_accounts (
        id SERIAL PRIMARY KEY,
        platform TEXT NOT NULL,
        account_name TEXT NOT NULL,
        account_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(platform, account_id)
    )
    ''')
    
    # social_media_posts table
    pg_cursor.execute('''
    CREATE TABLE IF NOT EXISTS social_media_posts (
        id SERIAL PRIMARY KEY,
        post_id TEXT NOT NULL,
        platform TEXT NOT NULL,
        account_id INTEGER REFERENCES social_media_accounts(id),
        content TEXT NOT NULL,
        author TEXT,
        timestamp TIMESTAMP NOT NULL,
        url TEXT,
        engagement_count INTEGER DEFAULT 0,
        raw_data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(platform, post_id)
    )
    ''')
    
    # social_media_sentiment table
    pg_cursor.execute('''
    CREATE TABLE IF NOT EXISTS social_media_sentiment (
        id SERIAL PRIMARY KEY,
        post_id INTEGER REFERENCES social_media_posts(id),
        sentiment INTEGER NOT NULL,
        confidence REAL,
        model_name TEXT,
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Migrate accounts
    sqlite_cursor.execute("SELECT platform, account_name, account_id, created_at, last_updated FROM social_media_accounts")
    account_rows = sqlite_cursor.fetchall()
    
    if account_rows:
        execute_values(
            pg_cursor,
            "INSERT INTO social_media_accounts (platform, account_name, account_id, created_at, last_updated) VALUES %s ON CONFLICT (platform, account_id) DO NOTHING",
            account_rows
        )
    
    # Migrate posts
    sqlite_cursor.execute("SELECT post_id, platform, account_id, content, author, timestamp, url, engagement_count, raw_data, created_at FROM social_media_posts")
    post_rows = sqlite_cursor.fetchall()
    
    if post_rows:
        execute_values(
            pg_cursor,
            "INSERT INTO social_media_posts (post_id, platform, account_id, content, author, timestamp, url, engagement_count, raw_data, created_at) VALUES %s ON CONFLICT (platform, post_id) DO NOTHING",
            post_rows
        )
    
    # Migrate sentiment
    sqlite_cursor.execute("SELECT post_id, sentiment, confidence, model_name, processed_at FROM social_media_sentiment")
    sentiment_rows = sqlite_cursor.fetchall()
    
    if sentiment_rows:
        execute_values(
            pg_cursor,
            "INSERT INTO social_media_sentiment (post_id, sentiment, confidence, model_name, processed_at) VALUES %s",
            sentiment_rows
        )
    
    # Commit and close
    pg_conn.commit()
    sqlite_conn.close()
    pg_conn.close()
    
    print(f"Migrated {len(account_rows)} accounts, {len(post_rows)} posts, and {len(sentiment_rows)} sentiment records")

if __name__ == "__main__":
    # Get PostgreSQL connection string from environment
    pg_conn_string = os.environ.get("DATABASE_URL")
    
    if not pg_conn_string:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)
    
    # Migrate both databases
    migrate_incidents_db(pg_conn_string)
    migrate_social_media_db(pg_conn_string)
    
    print("Migration complete!")
```

2. Update all database connection functions to use PostgreSQL when deployed:
   - Modify `app/data/database.py` and `app/social_media/database.py` to check for `DATABASE_URL` environment variable
   - Use psycopg2 for PostgreSQL connection when available

3. Run the migration script in the shell of any service:
   ```
   python scripts/migrate_to_postgres.py
   ```

## Step 7: Verify Deployment

1. Check that all seven services are running (green status in Render dashboard)
2. Test each service by visiting its URL:
   - Mock API: `https://sentiment-mock-api.onrender.com/health`
   - Model API: `https://sentiment-model-api.onrender.com/health`
   - Main Dashboard: `https://sentiment-dashboard.onrender.com/`
   - Social Media Auth UI: `https://sentiment-social-auth.onrender.com/`
   - Social Media Data Visualization: `https://sentiment-social-data.onrender.com/`
   - CSV Data Upload: `https://sentiment-csv-upload.onrender.com/`

## Step 8: Set Up Navigation Links

Update the navigation links in each Streamlit application to point to the correct URLs:

1. In your Render dashboard, note the URLs of all deployed services
2. For each Streamlit app, add a `.streamlit/config.toml` file in its directory:

```toml
[browser]
serverAddress = "sentiment-dashboard.onrender.com"
serverPort = 443
```

3. Update navigation links in each app to use the full URLs of the deployed services

## Troubleshooting

- **Service Crashes**: Check the logs in the Render dashboard
- **Database Issues**: Verify the DATABASE_URL environment variable is correctly set
- **Connection Errors**: Check if services can communicate with each other
- **CSS/JS Loading Issues**: Ensure Streamlit base URL is configured correctly
- **Persistent Storage**: Note that Render's default disk storage is ephemeral; use the database for persistent data

## Maintenance

- Render automatically redeploys services when you push changes to your repository
- You can manually restart services from the Render dashboard
- Set up Render's built-in monitoring for each service
- For database backups, use Render's automatic backup feature for the PostgreSQL database 