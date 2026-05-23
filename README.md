# RouteMate

ML-based ride-sharing optimization system using reinforcement learning and vehicle pooling.

---

## Features

- Ride-sharing optimization using PPO reinforcement learning
- Vehicle pooling support
- Real-time simulation environment
- Performance comparison with heuristic baselines
- REST API integration using FastAPI
- Interactive frontend visualization using React

---

## Tech Stack

- Python
- Gymnasium
- Stable-Baselines3
- FastAPI
- React
- NumPy
- Pandas
- Matplotlib

---

## Results

- Improved vehicle-request matching efficiency compared to heuristic approaches
- Simulated and evaluated 10,000+ ride requests
- Compared PPO agent against greedy and random baselines
- Implemented vehicle pooling for efficient passenger allocation

---

## Architecture

```text
User Requests
      ↓
Simulation Environment
      ↓
RL Agent (PPO)
      ↓
Vehicle Assignment Engine
      ↓
Metrics & Visualization
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/makarand1409/RouteMate.git
cd RouteMate
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Project

```bash
python src/simulator/simulation_engine.py
```

---

## Future Improvements

- Real-time traffic-aware routing
- Multi-agent reinforcement learning
- Live map integration
- Advanced analytics dashboard

---

## Contributors

- Adithya Madivala
- Makarand

