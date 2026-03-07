# RouteMATE: ML-Based Dynamic Ride-Sharing with Vehicle Pooling

## 🎯 Project Overview

RouteMATE is a machine learning-based ride-sharing optimization system that uses reinforcement learning to improve vehicle-request matching compared to traditional heuristic approaches. The system supports vehicle pooling (multiple passengers per vehicle) and provides a complete simulation environment for training and evaluation.


## 📊 Current Status (Phases 1-3 Complete)

✅ **Phase 1:** Core Simulator with Pooling  
✅ **Phase 2:** OpenAI Gym Environment Wrapper  
✅ **Phase 3:** RL Agent Training (PPO)  
🔄 **Phase 4:** FastAPI Backend (In Progress)  
🔄 **Phase 5:** React Frontend (In Progress)

---

## 🏗️ Project Structure

```
routemate/
├── src/
│   ├── simulator/              # Core simulation engine (Phase 1)
│   │   ├── __init__.py
│   │   ├── grid_city.py       # 10x10 grid environment
│   │   ├── request.py         # Ride request handling
│   │   ├── vehicle.py         # Vehicle with pooling (capacity=4)
│   │   ├── request_generator.py  # Poisson request generation
│   │   ├── matching_policy.py    # Baseline policies
│   │   ├── simulation_engine.py  # Main simulator
│   │   └── metrics_and_viz.py    # Metrics & visualization
│   │
│   ├── environment/            # Gym environment (Phase 2)
│   │   ├── __init__.py
│   │   └── gym_environment.py  # RL environment wrapper
│   │
│   ├── agents/                 # RL training (Phase 3)
│   │   ├── __init__.py
│   │   └── train_ppo_agent.py  # PPO training script
│   │
│   ├── backend/                # FastAPI backend (Phase 4) - TO DO
│   │   └── main.py
│   │
│   └── frontend/               # React frontend (Phase 5) - TO DO
│       └── src/
│
├── tests/                      # Test scripts
│   ├── test_basic_components.py
│   ├── test_random_agent.py
│   ├── test_greedy_baseline.py
│   └── compare_baselines.py
│
├── outputs/                    # Generated outputs
│   ├── models/                 # Trained ML models
│   │   └── ppo_routemate_final.zip
│   ├── logs/                   # Training logs
│   └── *.png                   # Visualization charts
│
├── requirements_phase1.txt     # Simulator dependencies
├── requirements_phase2.txt     # ML dependencies
├── requirements_phase4.txt     # Backend dependencies (TO DO)
├── requirements_phase5.txt     # Frontend dependencies (TO DO)
└── README.md                   # This file
```

---

## 🚀 Quick Start for New Team Members

### **Prerequisites**

- Python 3.8 or higher
- Git
- Code editor (VS Code recommended)

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/YOUR_USERNAME/routemate.git
cd routemate
```

### **Step 2: Create Virtual Environment**

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### **Step 3: Install Dependencies**

**For Phases 1-3 (Simulator + ML):**
```bash
pip install -r requirements_phase1.txt
pip install -r requirements_phase2.txt
```

This installs:
- `numpy`, `matplotlib`, `pandas` (data processing)
- `gymnasium` (RL environment framework)
- `stable-baselines3` (RL algorithms)
- `torch` (PyTorch for neural networks)
- `tensorboard` (training visualization)

---

## 🧪 Testing the Setup

### **1. Test Core Simulator**

```bash
python src/simulator/simulation_engine.py
```

**Expected output:**
```
Simulation complete!
Total requests: 150
Completed: 120
Completion rate: 80.0%
```

### **2. Test Gym Environment**

```bash
python src/environment/gym_environment.py
```

**Expected output:**
```
✓ Gym environment working correctly!
Episode complete!
Total Episode Reward: +250.30
```

### **3. Test Baseline Policies**

```bash
# Test greedy baseline
python tests/test_greedy_baseline.py

# Test random baseline  
python tests/test_random_agent.py

# Compare both
python tests/compare_baselines.py
```

### **4. Load Trained ML Model**

```python
from stable_baselines3 import PPO

# Load the trained model
model = PPO.load("outputs/models/ppo_routemate_final")

# Model is ready to use!
print("✓ Trained model loaded successfully")
```

---

## 📊 Current Results (Phase 3)

| Policy | Average Reward | Completion Rate | Notes |
|--------|---------------|-----------------|-------|
| Random | +287.10 | 20.4% | Worst case baseline |
| Greedy (Nearest Vehicle) | +498.90 | 27.7% | **Best performing** |
| PPO Agent (200k steps) | +300.10 | 20.6% | Needs more training |

**Key Insight:** The simple greedy heuristic outperforms the RL agent with current training. Future work includes longer training, hyperparameter tuning, or trying DQN.

---

## 🛠️ Phase 4: Backend Development (FastAPI)

**Assigned to:** [Team Member Names]

### **Objectives:**

1. Create REST API to serve the trained ML model
2. Implement endpoints for simulation and prediction
3. Add metrics and monitoring endpoints
4. Enable CORS for frontend integration

### **Required Files:**

```
src/backend/
├── main.py              # FastAPI application
├── models.py            # Pydantic models
├── routes/
│   ├── simulation.py    # Simulation endpoints
│   ├── prediction.py    # ML prediction endpoints
│   └── metrics.py       # Metrics endpoints
└── utils/
    └── model_loader.py  # Load trained models
```

### **Key Endpoints to Implement:**

```python
POST /api/simulate
- Body: { "policy": "greedy|random|ml", "num_steps": 100 }
- Returns: Simulation results with metrics

POST /api/predict
- Body: { "observation": [...] }
- Returns: { "action": 2, "vehicle_id": 2 }

GET /api/models
- Returns: List of available trained models

GET /api/metrics
- Returns: Performance comparison of all policies
```

### **Setup:**

```bash
# Install backend dependencies
pip install fastapi uvicorn python-multipart

# Run backend
cd src/backend
uvicorn main:app --reload

# API docs available at: http://localhost:8000/docs
```

---

## 🎨 Phase 5: Frontend Development (React)

**Assigned to:** [Team Member Names]

### **Objectives:**

1. Create interactive dashboard for simulation
2. Visualize vehicle movements and requests in real-time
3. Display performance metrics and comparisons
4. Allow policy selection and parameter tuning

### **Required Components:**

```
src/frontend/
├── public/
├── src/
│   ├── components/
│   │   ├── SimulationGrid.jsx    # 10x10 grid display
│   │   ├── VehicleMarker.jsx     # Vehicle visualization
│   │   ├── MetricsPanel.jsx      # Performance metrics
│   │   ├── ControlPanel.jsx      # Simulation controls
│   │   └── ComparisonChart.jsx   # Policy comparison
│   ├── services/
│   │   └── api.js                # Backend API calls
│   ├── App.jsx
│   └── index.js
└── package.json
```

### **Features to Implement:**

- ✅ Grid visualization (10x10 with vehicles and requests)
- ✅ Real-time simulation playback
- ✅ Policy selector (Random, Greedy, ML)
- ✅ Metrics dashboard (reward, completion rate)
- ✅ Comparison charts (all three policies)
- ✅ Adjustable parameters (vehicles, request rate)

### **Setup:**

```bash
# Create React app
npx create-react-app frontend
cd frontend

# Install dependencies
npm install axios recharts react-icons

# Start development server
npm start

# App runs at: http://localhost:3000
```

### **Connecting to Backend:**

```javascript
// src/services/api.js
import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

export const runSimulation = async (policy, numSteps) => {
  const response = await axios.post(`${API_BASE}/simulate`, {
    policy,
    num_steps: numSteps
  });
  return response.data;
};

export const getPrediction = async (observation) => {
  const response = await axios.post(`${API_BASE}/predict`, {
    observation
  });
  return response.data;
};
```

---

## 📚 Key Concepts to Understand

### **1. Simulation Engine**

- **Grid City:** 10x10 grid representing city locations
- **Vehicles:** 5 vehicles with capacity of 4 passengers each
- **Requests:** Generated using Poisson process (λ=1.5 per step)
- **Pooling:** Multiple passengers can share one vehicle

### **2. Policies**

**Random Policy:**
- Randomly selects any vehicle for assignment
- Baseline for worst-case performance

**Greedy Policy (Nearest Vehicle):**
- Assigns request to closest available vehicle
- Simple heuristic, surprisingly effective

**ML Policy (PPO):**
- Trained reinforcement learning agent
- Learns from 200,000 simulation steps
- Uses neural network to predict best vehicle

### **3. Gym Environment**

**Observation Space (29 values):**
- Request: pickup_x, pickup_y, dropoff_x, dropoff_y (4 values)
- Each vehicle: x, y, occupancy, queue_length, distance (5 × 5 = 25 values)

**Action Space:**
- Discrete(5) - Select which vehicle (0-4)

**Reward Function:**
- -0.01 per step (time penalty)
- +2.0 for successful pickup
- +5.0 for completed trip
- -1.0 for failed assignment

---

## 🐛 Common Issues & Solutions

### **Issue 1: ModuleNotFoundError**

```
ModuleNotFoundError: No module named 'src'
```

**Solution:**
```bash
# Make sure you're in project root
cd /path/to/routemate

# Run from root, not from subdirectories
python src/simulator/simulation_engine.py  # ✅ Correct
```

### **Issue 2: Import Errors**

```
ImportError: cannot import name 'GridCity'
```

**Solution:**
```bash
# Check __init__.py exists in src/simulator/
# Reinstall in development mode
pip install -e .
```

### **Issue 3: Gymnasium API Errors**

```
ValueError: too many values to unpack
```

**Solution:**
Make sure you're using the latest versions:
- `gymnasium>=0.29.0`
- `stable-baselines3>=2.0.0`

All environment files have been updated for Gymnasium API.

---

## 📖 Documentation & Resources

### **Project Documentation:**
- `PHASE1_COMPLETE.md` - Phase 1 completion guide
- `PHASE2_GUIDE.md` - Gym environment setup
- `PHASE3_GUIDE.md` - RL training guide

### **External Resources:**
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [React Documentation](https://react.dev/)

### **Research Papers:**
- Proximal Policy Optimization (PPO): [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Ride-sharing optimization surveys

---

## 🎯 Development Workflow

### **For Backend Developer:**

1. **Branch strategy:**
   ```bash
   git checkout -b feature/backend-api
   ```

2. **Daily workflow:**
   ```bash
   git pull origin main
   # Make changes
   git add .
   git commit -m "Added simulation endpoint"
   git push origin feature/backend-api
   ```

3. **When complete:**
   - Create Pull Request
   - Request review from team
   - Merge to main

### **For Frontend Developer:**

1. **Branch strategy:**
   ```bash
   git checkout -b feature/react-dashboard
   ```

2. **Daily workflow:**
   ```bash
   git pull origin main
   # Make changes
   git add .
   git commit -m "Added simulation grid component"
   git push origin feature/react-dashboard
   ```

3. **When complete:**
   - Create Pull Request
   - Test with backend
   - Merge to main

---

## 🧪 Testing Guidelines

### **Backend Tests:**

```bash
# Install testing framework
pip install pytest httpx

# Run tests
pytest tests/test_backend.py
```

### **Frontend Tests:**

```bash
# Install testing library
npm install --save-dev @testing-library/react

# Run tests
npm test
```

---

## 📊 Performance Targets

### **Backend (Phase 4):**
- ✅ API response time < 100ms
- ✅ Simulation endpoint < 2 seconds for 100 steps
- ✅ Concurrent requests handling
- ✅ Proper error handling

### **Frontend (Phase 5):**
- ✅ Smooth grid rendering (60 FPS)
- ✅ Real-time updates without lag
- ✅ Responsive design (mobile + desktop)
- ✅ Accessible UI (WCAG compliance)

---

## 🤝 Team Coordination

### **Communication Channels:**
- WhatsApp/Discord: Daily updates
- GitHub Issues: Track bugs and features
- GitHub Projects: Kanban board for tasks

### **Meeting Schedule:**
- Daily standup: 15 min sync
- Weekly review: 1 hour progress review
- Demo prep: Before submission

---

## 🎓 For Project Submission

### **Deliverables:**

1. ✅ **Source Code** (GitHub repository)
2. ✅ **Documentation** (README, API docs)
3. ✅ **Presentation** (PPT with results)
4. ✅ **Demo Video** (5-10 min walkthrough)
5. ✅ **Report** (PDF with methodology)

### **What to Highlight:**

- Complete ML pipeline (simulator → training → deployment)
- Comparative analysis (Random vs Greedy vs ML)
- Full-stack implementation (Backend + Frontend)
- Professional software engineering (Git, testing, docs)
- Real performance metrics and honest results

---

## 🚀 Quick Commands Reference

```bash
# Activate virtual environment
venv\Scripts\activate              # Windows
source venv/bin/activate           # Mac/Linux

# Run tests
python tests/test_greedy_baseline.py
python tests/compare_baselines.py

# Train new model
python src/agents/train_ppo_agent.py

# Start backend (Phase 4)
uvicorn src.backend.main:app --reload

# Start frontend (Phase 5)
cd src/frontend && npm start

# View training logs
tensorboard --logdir outputs/logs
```



## 🆘 Need Help?

**For Phase 1-3 (Simulator/ML):**
- Check documentation in `/outputs/`
- Review test scripts in `/tests/`
- Trained model available in `/outputs/models/`

**For Phase 4 (Backend):**
- Check FastAPI docs: https://fastapi.tiangolo.com
- Example endpoints in this README
- Test with Postman or curl
-starting backend

**For Phase 5 (Frontend):**
- Check React docs: https://react.dev
- Component examples in this README
- Use browser DevTools for debugging

**General Issues:**
- Open GitHub Issue
- Contact team members
- Review existing code examples

---

