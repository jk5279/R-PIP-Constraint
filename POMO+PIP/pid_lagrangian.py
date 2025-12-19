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
        if lambda_max < lambda_min:
            raise ValueError(f"lambda_max ({lambda_max}) must be >= lambda_min ({lambda_min})")
        if ema_beta < 0.0 or ema_beta >= 1.0:
            raise ValueError(f"ema_beta must be in [0, 1); got {ema_beta}")

        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.target = float(target)
        self.ema_beta = float(ema_beta)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.signal_clip = (float(signal_clip[0]), float(signal_clip[1]))

        # Windup protection default: bound integral such that Ki * integral doesn't dwarf lambda_max.
        # Use integral_limit = lambda_max / max(Ki, eps), but keep a sensible hard cap for Ki ~ 0.
        if integral_limit is None:
            eps = 1e-8
            if abs(self.Ki) < eps:
                integral_limit = 100.0
            else:
                integral_limit = self.lambda_max / max(abs(self.Ki), eps)
                integral_limit = min(integral_limit, 100.0)
        self.integral_limit = float(abs(integral_limit))

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

        # Positive error means violating constraint, should increase lambda.
        error = sig - self.target

        if self.state.steps == 0:
            ema_error = error
        else:
            ema_error = self.ema_beta * self.state.ema_error + (1.0 - self.ema_beta) * error

        # Integral term with windup protection
        error_integral = self.state.error_integral + ema_error
        error_integral = _clamp(error_integral, -self.integral_limit, self.integral_limit)

        # Derivative term on the smoothed error
        d_error = ema_error - self.state.prev_ema_error

        delta = (self.Kp * ema_error) + (self.Ki * error_integral) + (self.Kd * d_error)

        lambda_val = self.state.lambda_val + delta
        lambda_val = _clamp(lambda_val, self.lambda_min, self.lambda_max)

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

