"""
Quick training script using improved environment
Place in project root and run: python quick_train_improved.py
"""

import sys
sys.path.insert(0, '.')

from src.environment.gym_environment import RideSharingEnvImproved
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
import os

print("=" * 70)
print("TRAINING IMPROVED AGENT - 1M STEPS")
print("=" * 70)

# Create directories
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)

# Create environments
print("\n1. Creating improved environment...")
env = RideSharingEnvImproved(max_steps=200)
eval_env = RideSharingEnvImproved(max_steps=200)
print("   ✓ Done")

# Create model
print("\n2. Creating PPO model...")
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0001,
    n_steps=2048,
    batch_size=128,
    n_epochs=15,
    gamma=0.99,
    ent_coef=0.02,
    verbose=1,
    tensorboard_log="outputs/logs"
)
print("   ✓ Done")

# Setup callbacks
print("\n3. Setting up callbacks...")
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="outputs/models",
    log_path="outputs/logs",
    eval_freq=5000,
    n_eval_episodes=5,
    deterministic=True
)

checkpoint_callback = CheckpointCallback(
    save_freq=25000,
    save_path="outputs/models",
    name_prefix='ppo_improved_checkpoint'
)

callbacks = CallbackList([eval_callback, checkpoint_callback])
print("   ✓ Done")

# Train!
print("\n4. Starting training (1M steps = 3-4 hours)...")
print("   " + "-" * 66)

model.learn(
    total_timesteps=1000000,
    callback=callbacks,
    progress_bar=True
)

print("\n   " + "-" * 66)
print("   ✓ Training complete!")

# Save
print("\n5. Saving final model...")
model.save("outputs/models/ppo_routemate_improved_1M")
print("   ✓ Saved to: outputs/models/ppo_routemate_improved_1M.zip")

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print("\nNext: Evaluate with:")
print("  python tests/compare_all_policies.py")
