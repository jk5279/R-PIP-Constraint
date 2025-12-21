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
    PID controller (Position Form) for adapting a non-negative Lagrange multiplier.
    
    Fixes double-integration issue by mapping PID output directly to lambda value.
    The 'Integral' term maintains the history necessary to enforce constraints.
    
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
        if lambda_max < lambda_min:
            raise ValueError(f"lambda_max ({lambda_max}) must be >= lambda_min ({lambda_min})")
        if ema_beta < 0.0 or ema_beta >= 1.0:
            raise ValueError(f"ema_beta must be in [0, 1); got {ema_beta}")

        # Store parameters
        self.lambda_init = float(lambda_init)  # Stored as bias
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.target = float(target)
        self.ema_beta = float(ema_beta)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.signal_clip = (float(signal_clip[0]), float(signal_clip[1]))

        # Windup protection
        # In Position Form, the integral term alone can drive lambda up to lambda_max.
        # So we limit the integral sum such that Ki * integral_limit approx lambda_max.
        if integral_limit is None:
            eps = 1e-8
            if abs(self.Ki) < eps:
                integral_limit = 100.0
            else:
                # Allow integral to cover the full range of lambda
                integral_limit = self.lambda_max / max(abs(self.Ki), eps)
        self.integral_limit = float(abs(integral_limit))

        # Initialize state
        # Note: We start with integral=0, so lambda starts at lambda_init
        self.state = PIDLambdaState(
            lambda_val=_clamp(self.lambda_init, self.lambda_min, self.lambda_max),
            ema_error=0.0,
            error_integral=0.0,
            prev_ema_error=0.0,
            steps=0,
        )

    def step(self, signal: float) -> Dict[str, float]:
        """
        Update lambda given a scalar constraint signal (e.g., infeasible rate).
        Uses Position Form: Lambda = Bias + P + I + D
        """
        sig = float(signal)
        sig = _clamp(sig, self.signal_clip[0], self.signal_clip[1])

        # Positive error = Violation (Need higher lambda)
        error = sig - self.target

        # EMA smoothing
        if self.state.steps == 0:
            ema_error = error
        else:
            ema_error = self.ema_beta * self.state.ema_error + (1.0 - self.ema_beta) * error

        # Integral term with windup protection
        # This accumulates the "pressure" needed to keep the agent safe
        error_integral = self.state.error_integral + ema_error
        error_integral = _clamp(error_integral, -self.integral_limit, self.integral_limit)

        # Derivative term
        d_error = ema_error - self.state.prev_ema_error

        # --- POSITION FORM CALCULATION ---
        # 1. Calculate the raw PID output
        p_term = self.Kp * ema_error
        i_term = self.Ki * error_integral
        d_term = self.Kd * d_error

        # 2. Add to the base bias (lambda_init)
        # This ensures that if all errors are 0, we return to the baseline (or stay high if I-term is high)
        raw_lambda = self.lambda_init + p_term + i_term + d_term

        # 3. Project to valid range
        lambda_val = _clamp(raw_lambda, self.lambda_min, self.lambda_max)

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
            "p_term": p_term,  # Helpful for debugging which term is dominant
            "i_term": i_term,
            "d_term": d_term,
            "delta": p_term + i_term + d_term,  # For backward compatibility
        }

    def get_lambda(self) -> float:
        return float(self.state.lambda_val)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "lambda_init": self.lambda_init,  # Added to config
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
        # Load config if present to ensure parameters match
        if "config" in d:
            config = d["config"]
            self.lambda_init = float(config.get("lambda_init", self.lambda_init))
            self.Kp = float(config.get("Kp", self.Kp))
            self.Ki = float(config.get("Ki", self.Ki))
            self.integral_limit = float(config.get("integral_limit", self.integral_limit))

        state = d.get("state", d)
        self.state = PIDLambdaState(
            lambda_val=float(state.get("lambda_val", self.state.lambda_val)),
            ema_error=float(state.get("ema_error", self.state.ema_error)),
            error_integral=float(state.get("error_integral", self.state.error_integral)),
            prev_ema_error=float(state.get("prev_ema_error", self.state.prev_ema_error)),
            steps=int(state.get("steps", self.state.steps)),
        )

