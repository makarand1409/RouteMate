# RouteMATE

RouteMATE is an ML-based ride-sharing simulation platform. It matches ride requests with vehicles, supports vehicle pooling, and provides a FastAPI backend with a React frontend.


## Prerequisites

- Python 3.10 or newer
- Node.js and npm
- Git

## Setup

Clone the repository:

```bash
git clone https://github.com/makarand1409/RouteMate.git
cd RouteMate
```

Create and activate a Python virtual environment.

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install Python dependencies from the repository root:

```bash
python -m pip install -r requirements_phase1.txt
python -m pip install -r requirements_phase2.txt
python -m pip install -r requirements_phase4.txt
```

Install frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

## Run the Application

Start the backend in one terminal from the repository root:

```powershell
.\venv\Scripts\Activate.ps1
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend URLs:

- API: http://localhost:8000
- Swagger documentation: http://localhost:8000/docs

Start the frontend in a second terminal from the repository root:

```bash
cd frontend
npm start
```

Frontend URL: http://localhost:3000

