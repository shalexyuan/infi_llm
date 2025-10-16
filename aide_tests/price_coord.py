from typing import Tuple

# ------------ price_coord.py ------------
class PriceCoordinator:
    def __init__(self, B_comm_bytes: float, B_H: float,
                 eta_p: float = 1e-3, eta_tau: float = 1e-3, ema: float = 0.6):
        self.p = 0.0; self.tau = 0.0
        self.Bc = float(B_comm_bytes); self.Bh = float(B_H)
        self.eta_p = eta_p; self.eta_tau = eta_tau; self.ema = ema
        self._C_ema = 0.0; self._H_ema = 0.0

    def update(self, total_C_bytes: float, total_H: float) -> Tuple[float,float]:
        # 平滑观测，避免价格振荡
        self._C_ema = self.ema*self._C_ema + (1-self.ema)*float(total_C_bytes)
        self._H_ema = self.ema*self._H_ema + (1-self.ema)*float(total_H)
        # 次梯度上调：超预算就涨价
        self.p   = max(0.0, self.p   + self.eta_p  * (self._C_ema - self.Bc))
        self.tau = max(0.0, self.tau + self.eta_tau* (self._H_ema - self.Bh))
        return self.p, self.tau
