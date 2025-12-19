# Implementation Guide: R-PIP-constraint Modifications

This document provides a comprehensive guide to all modifications made to the original [PIP-constraint](https://github.com/jieyibi/PIP-constraint) repository. It is designed to enable someone to recreate these changes given the original reference code.

**Repository**: `https://github.com/jk5279/R-PIP-Constraint.git`  
**Original Repository**: `https://github.com/jieyibi/PIP-constraint.git`  
**Base Commit**: Original PIP-constraint repository state

---

## Table of Contents

1. [Problem Generation Changes: TSPTW → STSPTW](#1-problem-generation-changes-tsptw--stsptw)
2. [PIP Buffer Implementation](#2-pip-buffer-implementation)
3. [PID-Lagrangian Lambda Controller](#3-pid-lagrangian-lambda-controller)
4. [Validation Dataset Handling](#4-validation-dataset-handling)

---

## 1. Problem Generation Changes: TSPTW → STSPTW

### 1.1 Overview

The original TSPTW (Traveling Salesperson Problem with Time Windows) was converted to STSPTW (Stochastic Traveling Salesperson Problem with Time Windows) by adding stochastic noise to distance calculations. Additionally, the problem generation method was unified to use an α/β-based approach for all hardness levels, replacing the previous "hard" generation method that was incompatible with distance modifications.

### 1.2 Key Changes

#### 1.2.1 Rename TSPTW to STSPTW

**Files Modified:**
- `POMO+PIP/envs/TSPTWEnv.py` → `POMO+PIP/envs/STSPTWEnv.py`
- `POMO+PIP/train.py`
- `POMO+PIP/test.py`
- `POMO+PIP/models/SINGLEModel.py`
- `POMO+PIP/Trainer.py`
- `POMO+PIP/Tester.py`
- `POMO+PIP/utils.py`
- `POMO+PIP/generate_data.py`
- `README.md`

**Implementation Steps:**

1. **Rename Environment File:**
   ```bash
   mv POMO+PIP/envs/TSPTWEnv.py POMO+PIP/envs/STSPTWEnv.py
   ```

2. **Update Class Name and Problem String:**
   In `POMO+PIP/envs/STSPTWEnv.py`:
   ```python
   # Change:
   __all__ = ['TSPTWEnv']
   class TSPTWEnv:
       self.problem = "TSPTW"
   
   # To:
   __all__ = ['STSPTWEnv']
   class STSPTWEnv:
       self.problem = "STSPTW"
   ```

3. **Update All References:**
   - Replace `"TSPTW"` with `"STSPTW"` in all string comparisons
   - Replace `TSPTWEnv` with `STSPTWEnv` in imports
   - Update dataset paths: `data/TSPTW/` → `data/STSPTW/`
   - Update pretrained model paths: `pretrained/TSPTW/` → `pretrained/STSPTW/`

#### 1.2.2 Add Stochastic Noise to Distance Calculations

**Location:** `POMO+PIP/envs/STSPTWEnv.py`

**Implementation:**

1. **Add Noise Sampling Method:**
   ```python
   def _sample_distance_noise(self, shape):
       """Sample uniform noise U(0, sqrt(2)) for distance calculations"""
       return torch.rand(shape, device=self.device) * np.sqrt(2)
   ```

2. **Initialize Noise Parameters in `__init__`:**
   ```python
   # Stochastic noise parameters
   self.noise_max = np.sqrt(2)  # Maximum noise value U(0, sqrt(2))
   ```

3. **Add Noise to Distance Calculations in `step()` Method:**
   
   **Current step distance (to next node):**
   ```python
   # Original:
   new_length = (current_coord - self.current_coord).norm(p=2, dim=-1)
   
   # Modified:
   new_length = (current_coord - self.current_coord).norm(p=2, dim=-1)
   noise = self._sample_distance_noise(new_length.shape)
   new_length = new_length + noise
   ```

   **Travel time calculation (for unvisited nodes):**
   ```python
   # Original:
   travel_time = (self.current_coord[:, :, None, :] - self.node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1)
   
   # Modified:
   travel_time = (self.current_coord[:, :, None, :] - self.node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1)
   noise = self._sample_distance_noise(travel_time.shape)
   travel_time = travel_time + noise
   ```

4. **Add Noise to Travel Distance Calculation in `_get_travel_distance()`:**
   ```python
   # Original:
   segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
   
   # Modified:
   segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
   noise = self._sample_distance_noise(segment_lengths.shape)
   segment_lengths = segment_lengths + noise
   ```

**Important Notes:**
- Noise is sampled **independently at each step** for each distance measurement
- The **initial distance matrix** (depot distances) remains **noiseless**
- **PIP lookahead calculations** use **noiseless distances** (see Section 2 for PIP buffer details)
- Noise follows uniform distribution: **U(0, √2)**

#### 1.2.3 Unified Problem Generation Method

**Location:** `POMO+PIP/envs/STSPTWEnv.py` → `get_random_problems()` method

**Problem:** The original "hard" generation method was incompatible with distance modifications (e.g., adding noise). It used a different algorithm that would break if distances changed.

**Solution:** Replace all hardness-specific generation methods with a unified α/β-based approach.

**Implementation:**

1. **Remove Hard-Specific Generation Logic:**
   ```python
   # DELETE any blocks like:
   if self.hardness == "hard":
       # Special hard generation code
   ```

2. **Add Unified Hardness Mapping:**
   ```python
   def get_random_problems(self, batch_size, problem_size, coord_factor=100, max_tw_size=100):
       # Hardness to α/β mapping for time window duration
       # α and β represent the range [α, β] for time window duration as fraction of time_factor
       hardness_to_alpha_beta = {
           "easy": [0.5, 0.75],    # Wide windows, loose constraints
           "medium": [0.3, 0.48],  # Moderate constraints
           "hard": [0.1, 0.2]      # Narrow windows, tight constraints
       }
       
       # Get α/β values for current hardness level
       alpha_beta = hardness_to_alpha_beta[self.hardness]
       
       # Generate STSPTW data using unified method for all hardness levels
       tw = generate_stsptw_data(size=batch_size, graph_size=problem_size, 
                                time_factor=problem_size*55, tw_duration=alpha_beta)
       node_xy = torch.tensor(tw.node_loc).float()
       time_windows = torch.tensor(tw.node_tw)
       
       service_time = torch.zeros(size=(batch_size, problem_size))
       return node_xy, service_time, time_windows[:,:,0], time_windows[:,:,1]
   ```

**How It Works:**
- All hardness levels use the same `generate_stsptw_data()` function
- The `tw_duration` parameter is set to `[α, β]` based on hardness
- Time windows are sampled uniformly from `[α × time_factor, β × time_factor]`
- This makes the system robust to distance function modifications

**Hardness Parameters:**
- **Easy**: α = 0.5, β = 0.75 (Wide time windows, loose constraints)
- **Medium**: α = 0.3, β = 0.48 (Moderate constraints)
- **Hard**: α = 0.1, β = 0.2 (Narrow time windows, tight constraints)

#### 1.2.4 Update Data Generation Function Name

**Location:** `POMO+PIP/envs/STSPTWEnv.py`

**Change:**
```python
# Original:
def generate_tsptw_data(...):
    ...

# Modified:
def generate_stsptw_data(...):
    ...
```

Update all calls to this function accordingly.

---

## 2. PIP Buffer Implementation

### 2.1 Overview

A configurable buffer term (default: `√2`) is added to distance calculations within PIP's lookahead filtering mechanism. This buffer accounts for the worst-case stochastic noise uncertainty when performing feasibility checks.

### 2.2 Implementation Details

#### 2.2.1 Add PIP Buffer Parameter

**Location:** `POMO+PIP/envs/STSPTWEnv.py` → `__init__()`

```python
# PIP buffer parameter for distance calculations in lookahead
self.pip_buffer = env_params.get('pip_buffer', np.sqrt(2))  # Default to sqrt(2) for worst-case noise
```

#### 2.2.2 Pass Buffer from Training Script

**Location:** `POMO+PIP/train.py`

**Add Argument:**
```python
parser.add_argument('--pip_buffer', type=float, default=1.4142135623730951, 
                    help="Buffer term added to distance calculations in PIP mask (default: sqrt(2))")
```

**Pass to Environment:**
```python
env_params = {
    "problem_size": args.problem_size,
    "pomo_size": args.pomo_size,
    "hardness": args.hardness,
    # ... other params ...
    "pip_buffer": args.pip_buffer  # Add this
}
```

#### 2.2.3 Apply Buffer in PIP Lookahead Calculations

**Location:** `POMO+PIP/envs/STSPTWEnv.py` → `_calculate_PIP_mask()` method

**Key Principle:** The buffer is added to **distance before dividing by speed**, ensuring it represents a time buffer.

**Implementation Points:**

1. **PIP Step 0 (One-step lookahead):**
   ```python
   # Original:
   distance = (self.current_coord[:, :, None, :] - self.node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1)
   next_arrival_time = torch.max(self.current_time[:, :, None] + distance / self.speed, ...)
   
   # Modified:
   distance = (self.current_coord[:, :, None, :] - self.node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1)
   distance_with_buffer = distance + self.pip_buffer  # Add buffer to distance
   next_arrival_time = torch.max(self.current_time[:, :, None] + distance_with_buffer / self.speed, ...)
   ```

2. **PIP Step 1 (Two-step lookahead):**
   ```python
   # Around line 473-475:
   second_step_new_length = (two_step_current_coord - first_step_current_coord.unsqueeze(3).repeat(1, 1, 1, simulate_size - 1, 1)).norm(p=2, dim=-1)
   # Add PIP buffer to distance calculation for lookahead
   second_step_new_length = second_step_new_length + self.pip_buffer
   ```

3. **PIP Step 2 (Three-step lookahead):**
   ```python
   # Around line 524-526:
   second_step_new_length = (two_step_current_coord - first_step_current_coord.unsqueeze(3).repeat(1, 1, 1, unvisited_size - 1, 1)).norm(p=2, dim=-1)
   # Add PIP buffer to distance calculation for lookahead
   second_step_new_length = second_step_new_length + self.pip_buffer
   
   # Around line 552-554:
   third_step_new_length = (three_step_current_coord - two_step_current_coord.unsqueeze(4).expand(-1, -1, -1, -1, unvisited_size - 2, -1)).norm(p=2, dim=-1)
   # Add PIP buffer to distance calculation for lookahead
   third_step_new_length = third_step_new_length + self.pip_buffer
   ```

**Important Notes:**
- Buffer is added to **noiseless distances** used in PIP lookahead (PIP uses expected distances)
- Buffer defaults to `√2` to account for worst-case noise (since noise is U(0, √2))
- Buffer is configurable via `--pip_buffer` argument
- Buffer is added **before** dividing by speed (so it represents a distance buffer, not a time buffer directly)

#### 2.2.4 Update Test Script

**Location:** `POMO+PIP/test.py`

Add the same `--pip_buffer` argument and pass it to `env_params`:

```python
parser.add_argument('--pip_buffer', type=float, default=1.4142135623730951, 
                    help="Buffer term added to distance calculations in PIP mask (default: sqrt(2))")

# In args2dict or similar:
env_params = {
    # ... other params ...
    "pip_buffer": args.pip_buffer
}
```

---

## 3. PID-Lagrangian Lambda Controller

### 3.1 Overview

A PID (Proportional-Integral-Derivative) controller dynamically adjusts the Lagrange multiplier λ (which replaces the static `penalty_factor`) during training. The controller uses EMA-smoothed instance infeasibility rate as feedback, targeting zero violations.

### 3.2 Implementation Details

#### 3.2.1 Create PID Controller Module

**New File:** `POMO+PIP/pid_lagrangian.py`

**Complete Implementation:**

```python
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class PIDLambdaState:
    lambda_val: float
    ema_error: float
    error_integral: float
    prev_ema_error: float
    steps: int


class PIDLambdaController:
    """
    PID controller for adapting a non-negative Lagrange multiplier (lambda).
    
    Designed for noisy RL signals:
    - Uses EMA smoothing on the error signal (violation - target)
    - Integral windup protection via clamping
    - Projection to [lambda_min, lambda_max]
    """
    
    def __init__(
        self,
        lambda_init: float = 0.1,
        Kp: float = 0.1,
        Ki: float = 0.01,
        Kd: float = 0.0,
        target: float = 0.0,
        ema_beta: float = 0.9,
        lambda_min: float = 0.0,
        lambda_max: float = 10.0,
        integral_limit: Optional[float] = None,
        signal_clip: Tuple[float, float] = (0.0, 1.0),
    ):
        # Validation
        if lambda_max < lambda_min:
            raise ValueError(f"lambda_max ({lambda_max}) must be >= lambda_min ({lambda_min})")
        if ema_beta < 0.0 or ema_beta >= 1.0:
            raise ValueError(f"ema_beta must be in [0, 1); got {ema_beta}")
        
        # Store parameters
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.target = float(target)
        self.ema_beta = float(ema_beta)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.signal_clip = (float(signal_clip[0]), float(signal_clip[1]))
        
        # Windup protection: bound integral such that Ki * integral doesn't dwarf lambda_max
        if integral_limit is None:
            eps = 1e-8
            if abs(self.Ki) < eps:
                integral_limit = 100.0
            else:
                integral_limit = self.lambda_max / max(abs(self.Ki), eps)
                integral_limit = min(integral_limit, 100.0)
        self.integral_limit = float(abs(integral_limit))
        
        # Initialize state
        lambda_init = float(lambda_init)
        self.state = PIDLambdaState(
            lambda_val=_clamp(lambda_init, self.lambda_min, self.lambda_max),
            ema_error=0.0,
            error_integral=0.0,
            prev_ema_error=0.0,
            steps=0,
        )
    
    def step(self, signal: float) -> Dict[str, float]:
        """
        Update lambda given a scalar constraint signal (e.g., infeasible rate).
        
        Returns diagnostics dict with keys:
        - lambda_val, signal, error, ema_error, error_integral, delta
        """
        sig = float(signal)
        sig = _clamp(sig, self.signal_clip[0], self.signal_clip[1])
        
        # Positive error means violating constraint, should increase lambda
        error = sig - self.target
        
        # EMA smoothing
        if self.state.steps == 0:
            ema_error = error
        else:
            ema_error = self.ema_beta * self.state.ema_error + (1.0 - self.ema_beta) * error
        
        # Integral term with windup protection
        error_integral = self.state.error_integral + ema_error
        error_integral = _clamp(error_integral, -self.integral_limit, self.integral_limit)
        
        # Derivative term on the smoothed error
        d_error = ema_error - self.state.prev_ema_error
        
        # PID update
        delta = (self.Kp * ema_error) + (self.Ki * error_integral) + (self.Kd * d_error)
        
        lambda_val = self.state.lambda_val + delta
        lambda_val = _clamp(lambda_val, self.lambda_min, self.lambda_max)
        
        # Update state
        self.state = PIDLambdaState(
            lambda_val=lambda_val,
            ema_error=ema_error,
            error_integral=error_integral,
            prev_ema_error=ema_error,
            steps=self.state.steps + 1,
        )
        
        return {
            "lambda_val": self.state.lambda_val,
            "signal": sig,
            "error": error,
            "ema_error": self.state.ema_error,
            "error_integral": self.state.error_integral,
            "delta": delta,
        }
    
    def get_lambda(self) -> float:
        return float(self.state.lambda_val)
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "Kp": self.Kp,
                "Ki": self.Ki,
                "Kd": self.Kd,
                "target": self.target,
                "ema_beta": self.ema_beta,
                "lambda_min": self.lambda_min,
                "lambda_max": self.lambda_max,
                "integral_limit": self.integral_limit,
                "signal_clip": self.signal_clip,
            },
            "state": asdict(self.state),
        }
    
    def load_state_dict(self, d: Dict[str, Any]) -> None:
        state = d.get("state", d)
        self.state = PIDLambdaState(
            lambda_val=float(state.get("lambda_val", self.state.lambda_val)),
            ema_error=float(state.get("ema_error", self.state.ema_error)),
            error_integral=float(state.get("error_integral", self.state.error_integral)),
            prev_ema_error=float(state.get("prev_ema_error", self.state.prev_ema_error)),
            steps=int(state.get("steps", self.state.steps)),
        )
```

#### 3.2.2 Add CLI Arguments

**Location:** `POMO+PIP/train.py`

```python
# PID-Lagrangian lambda controller arguments
parser.add_argument('--pid_lambda', action='store_true', 
                    help="Enable PID controller for dynamic lambda adjustment")
parser.add_argument('--pid_lambda_init', type=float, default=0.1, 
                    help="Initial lambda value (default: 0.1)")
parser.add_argument('--pid_lambda_kp', type=float, default=0.1, 
                    help="Proportional gain (default: 0.1)")
parser.add_argument('--pid_lambda_ki', type=float, default=0.01, 
                    help="Integral gain (default: 0.01)")
parser.add_argument('--pid_lambda_kd', type=float, default=0.0, 
                    help="Derivative gain (default: 0.0)")
parser.add_argument('--pid_lambda_target', type=float, default=0.0, 
                    help="Target violation rate (default: 0.0)")
parser.add_argument('--pid_lambda_ema_beta', type=float, default=0.9, 
                    help="EMA smoothing factor for error signal (default: 0.9)")
parser.add_argument('--pid_lambda_max', type=float, default=10.0, 
                    help="Maximum lambda value (default: 10.0)")
```

**Pass to Trainer:**

```python
trainer_params = {
    # ... existing params ...
    "pid_lambda": args.pid_lambda,
    "pid_lambda_init": args.pid_lambda_init,
    "pid_lambda_kp": args.pid_lambda_kp,
    "pid_lambda_ki": args.pid_lambda_ki,
    "pid_lambda_kd": args.pid_lambda_kd,
    "pid_lambda_target": args.pid_lambda_target,
    "pid_lambda_ema_beta": args.pid_lambda_ema_beta,
    "pid_lambda_max": args.pid_lambda_max,
}
```

#### 3.2.3 Integrate into Trainer

**Location:** `POMO+PIP/Trainer.py`

**1. Import PID Controller:**

```python
from pid_lagrangian import PIDLambdaController
```

**2. Initialize Controller in `__init__()`:**

```python
# Initialize PID lambda controller if enabled
self.pid_lambda_controller = None
if self.trainer_params.get("pid_lambda", False):
    self.pid_lambda_controller = PIDLambdaController(
        lambda_init=self.trainer_params.get("pid_lambda_init", 0.1),
        Kp=self.trainer_params.get("pid_lambda_kp", 0.1),
        Ki=self.trainer_params.get("pid_lambda_ki", 0.01),
        Kd=self.trainer_params.get("pid_lambda_kd", 0.0),
        target=self.trainer_params.get("pid_lambda_target", 0.0),
        ema_beta=self.trainer_params.get("pid_lambda_ema_beta", 0.9),
        lambda_max=self.trainer_params.get("pid_lambda_max", 10.0),
    )
    # Set initial penalty_factor from lambda
    self.penalty_factor = self.pid_lambda_controller.get_lambda()
```

**3. Restore Controller State from Checkpoint:**

```python
# In checkpoint loading section (around line 39-53):
if args.checkpoint is not None:
    checkpoint_fullname = args.checkpoint
    checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
    # ... existing checkpoint loading ...
    
    # Restore PID lambda controller state if it exists
    if self.pid_lambda_controller is not None and 'pid_lambda_state' in checkpoint:
        self.pid_lambda_controller.load_state_dict(checkpoint['pid_lambda_state'])
        self.penalty_factor = self.pid_lambda_controller.get_lambda()
        print(">> PID Lambda Controller state restored (lambda = {:.4f})".format(self.penalty_factor))
```

**4. Update Lambda Each Epoch:**

```python
# In run() method, after _train_one_epoch() returns:
train_score, train_loss, infeasible = self._train_one_epoch(epoch)

# Update PID lambda controller if enabled
if self.pid_lambda_controller is not None:
    try:
        # Extract instance infeasibility rate
        if isinstance(infeasible, dict):
            ins_infeasible_rate = infeasible["ins_infeasible_rate"]
        else:
            sol_infeasible_rate, ins_infeasible_rate = infeasible
        
        # Update lambda
        diagnostics = self.pid_lambda_controller.step(ins_infeasible_rate)
        self.penalty_factor = diagnostics["lambda_val"]
        
        # Log lambda and diagnostics
        if self.tb_logger:
            self.tb_logger.log_value('pid_lambda/lambda', diagnostics["lambda_val"], epoch)
            self.tb_logger.log_value('pid_lambda/signal', diagnostics["signal"], epoch)
            self.tb_logger.log_value('pid_lambda/error', diagnostics["error"], epoch)
            self.tb_logger.log_value('pid_lambda/ema_error', diagnostics["ema_error"], epoch)
            self.tb_logger.log_value('pid_lambda/delta', diagnostics["delta"], epoch)
        
        if self.wandb_logger:
            wandb.log({'pid_lambda/lambda': diagnostics["lambda_val"]})
            wandb.log({'pid_lambda/signal': diagnostics["signal"]})
            wandb.log({'pid_lambda/error': diagnostics["error"]})
    except Exception as e:
        print(f">> Warning: PID lambda update failed: {e}")
```

**5. Save Controller State in Checkpoints:**

```python
# In checkpoint saving section (around line 132-142):
checkpoint_dict = {
    'epoch': epoch,
    'problem': self.args.problem,
    'model_state_dict': self.model.state_dict(),
    'optimizer_state_dict': self.optimizer.state_dict(),
    'scheduler_state_dict': self.scheduler.state_dict(),
    'result_log': self.result_log,
}

# Save PID lambda controller state if enabled
if self.pid_lambda_controller is not None:
    checkpoint_dict['pid_lambda_state'] = self.pid_lambda_controller.state_dict()

torch.save(checkpoint_dict, '{}/epoch-{}.pt'.format(self.log_path, epoch))
```

#### 3.2.4 Default Parameters

**Recommended Defaults:**
- `lambda_init = 0.1`: Start with small penalty
- `Kp = 0.1`: Proportional gain for responsiveness
- `Ki = 0.01`: Integral gain (smaller than Kp to avoid overshoot)
- `Kd = 0.0`: Derivative gain (set to 0 for simplicity, can be tuned)
- `target = 0.0`: Target zero violation rate
- `ema_beta = 0.9`: Strong smoothing for noisy RL signals
- `lambda_max = 10.0`: Prevent "death spiral" where agent freezes

**Rationale:**
- **Kp > Ki**: Proportional term reacts immediately to violations
- **Ki small**: Prevents integral windup from early exploration
- **EMA smoothing**: Essential for noisy RL signals
- **Bounds**: Prevent negative lambda (would reward violations) and excessive lambda (would freeze agent)

---

## 4. Validation Dataset Handling

### 4.1 Overview

Improved validation dataset loading to handle cases where the requested number of samples exceeds the dataset size, and to ensure consistency between default validation episodes and dataset sizes.

### 4.2 Implementation Details

#### 4.2.1 Handle Dataset Offset Exceeding Length

**Location:** `POMO+PIP/envs/STSPTWEnv.py` → `load_dataset()` method

**Problem:** When `offset >= len(all_data)`, the original code would return an empty tensor, causing indexing errors downstream.

**Solution:**

```python
def load_dataset(self, path, offset=0, num_samples=10000, disable_print=True):
    assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
    with open(path, 'rb') as f:
        all_data = pickle.load(f)
        # Handle case where offset exceeds dataset length
        if offset >= len(all_data):
            if not disable_print:
                print(f">> Load 0 data (<class 'list'>) from {path} (offset={offset} exceeds dataset length={len(all_data)})")
            return None  # Return None instead of empty tensor
        data = all_data[offset: offset+num_samples]
        if not disable_print:
            print(">> Load {} data ({}) from {}".format(len(data), type(data), path))
    if len(data) == 0:
        return None
    # ... rest of loading logic ...
```

#### 4.2.2 Update Trainer Validation Loop

**Location:** `POMO+PIP/Trainer.py` → `_val_and_stat()` method

**Problem:** The validation loop didn't handle `None` returns from `load_dataset()`, and used `val_episodes` instead of actual loaded batch size.

**Solution:**

```python
def _val_and_stat(self, dir, path, env, batch_size, val_episodes, epoch):
    # ... existing setup ...
    
    episode = 0
    while episode < val_episodes:
        # Load dataset batch
        data = env.load_dataset(path, offset=episode, num_samples=batch_size, disable_print=True)
        
        # Handle case where dataset is exhausted
        if data is None:
            break  # Exit loop if no more data
        
        # ... validation logic ...
        
        # Increment by actual loaded batch size (not requested batch_size)
        episode += data[-1].size(0)  # Use actual batch size from loaded data
    
    # ... compute statistics using actual episode count ...
    
    # When loading optimal solutions, use actual episode count
    opt_sol = load_dataset(sol_path, disable_print=True)[: episode]  # Use episode, not val_episodes
```

#### 4.2.3 Generate Consistent Validation Datasets

**Location:** `generate_validation_datasets.sh`

**Problem:** Default `val_episodes=10000` but original datasets had only 1000 samples.

**Solution:** Generate validation datasets with 10,000 samples to match default `val_episodes`.

```bash
#!/bin/bash
# Script to generate STSPTW validation datasets

BASE_DIR="/scratch/kimjong/R-PIP-constraint/POMO+PIP"
ENV_ACTIVATE="/scratch/kimjong/R-PIP-constraint/.venv/bin/activate"

# Default values, can be overridden by environment variables
NUM_SAMPLES="${NUM_SAMPLES:-10000}"  # Default to 10000 samples
SEED="${SEED:-2025}"  # Default seed

cd "$BASE_DIR" || exit 1

# Activate virtual environment
source "$ENV_ACTIVATE"

echo "Generating STSPTW50 datasets with ${NUM_SAMPLES} samples..."
python generate_data.py --problem=STSPTW --problem_size=50 --hardness=easy --num_samples="${NUM_SAMPLES}" --seed="${SEED}" --dir=../data --no_cuda
python generate_data.py --problem=STSPTW --problem_size=50 --hardness=medium --num_samples="${NUM_SAMPLES}" --seed="${SEED}" --dir=../data --no_cuda
python generate_data.py --problem=STSPTW --problem_size=50 --hardness=hard --num_samples="${NUM_SAMPLES}" --seed="${SEED}" --dir=../data --no_cuda

echo "Generating STSPTW100 datasets with ${NUM_SAMPLES} samples..."
python generate_data.py --problem=STSPTW --problem_size=100 --hardness=easy --num_samples="${NUM_SAMPLES}" --seed="${SEED}" --dir=../data --no_cuda
python generate_data.py --problem=STSPTW --problem_size=100 --hardness=medium --num_samples="${NUM_SAMPLES}" --seed="${SEED}" --dir=../data --no_cuda
python generate_data.py --problem=STSPTW --problem_size=100 --hardness=hard --num_samples="${NUM_SAMPLES}" --seed="${SEED}" --dir=../data --no_cuda

echo "All validation datasets generated!"
```

**Key Changes:**
- Default `NUM_SAMPLES=10000` to match `val_episodes`
- Generate all 6 datasets (sizes 50/100 × hardness easy/medium/hard)
- Use consistent seed for reproducibility

---

## Summary of File Changes

### New Files Created:
1. `POMO+PIP/pid_lagrangian.py` - PID controller implementation
2. `IMPLEMENTATION_GUIDE.md` - This document

### Files Modified:
1. `POMO+PIP/envs/TSPTWEnv.py` → `POMO+PIP/envs/STSPTWEnv.py` (renamed + major changes)
2. `POMO+PIP/train.py` - Added PID lambda and PIP buffer arguments
3. `POMO+PIP/test.py` - Added PIP buffer argument, updated problem references
4. `POMO+PIP/Trainer.py` - Integrated PID controller, updated validation handling
5. `POMO+PIP/Tester.py` - Updated problem references
6. `POMO+PIP/models/SINGLEModel.py` - Updated problem references
7. `POMO+PIP/utils.py` - Updated problem references, removed TSPDL
8. `POMO+PIP/generate_data.py` - Updated problem references
9. `README.md` - Updated documentation
10. `submit_training_jobs.sh` - Added PID+PIP-buffer variant
11. `generate_validation_datasets.sh` - Updated to generate 10K samples

### Files Removed:
- `POMO+PIP/envs/TSPDLEnv.py` - TSPDL problem removed
- All baseline implementations
- All AM+PIP related code
- All PIP-D (auxiliary decoder) related code
- Old TSPTW pretrained models and datasets

---

## References

- Original PIP-constraint repository: https://github.com/jieyibi/PIP-constraint
- PID-Lagrangian methods: Stooke et al., "Responsive Safety in Reinforcement Learning" (2020)
- POMO framework: https://github.com/yd-kwon/POMO

---

**Document Version:** 1.0  
**Last Updated:** December 2024

