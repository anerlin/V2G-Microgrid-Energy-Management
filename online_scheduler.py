import numpy as np
from v2g_model import V2GModel

class OnlineScheduler:
    def __init__(self, n_evs, T, ev_params, price, reg_signal, horizon=30):
        self.n_evs = n_evs
        self.T = T
        self.ev_params = ev_params
        self.price = price
        self.reg_signal = reg_signal
        self.horizon = horizon
        self.history = {'P': [], 'S': []}

    def run(self, external_load_forecast):
        print("\n" + "="*50)
        print("Online Rolling Horizon Scheduling")
        print("="*50)

        all_P = np.zeros((self.n_evs, self.T))
        all_S = np.zeros((self.n_evs, self.T))

        for t_current in range(self.T):
            price_current = self.price[t_current]
            reg_current = self.reg_signal[t_current]

            # 修复：检查是否为 None
            if external_load_forecast is not None:
                load_current = external_load_forecast[t_current]
            else:
                load_current = 0

            horizon_end = min(t_current + self.horizon, self.T)
            price_horizon = self.price[t_current:horizon_end]
            reg_horizon = self.reg_signal[t_current:horizon_end]

            if external_load_forecast is not None:
                load_horizon = external_load_forecast[t_current:horizon_end]
            else:
                load_horizon = None

            v2g = V2GModel(
                n_evs=self.n_evs,
                T=horizon_end - t_current,
                ev_params=self.ev_params,
                price=price_horizon,
                reg_signal=reg_horizon
            )

            cost, P_window, S_window = v2g.solve(
                external_load=load_horizon,
                is_online=True,
                t_current=0
            )

            if cost is not None:
                for i in range(self.n_evs):
                    all_P[i, t_current] = 0
                    all_S[i, t_current] = 0
                print(f"  t={t_current}: cost=${cost:.2f}")
            else:
                print(f"  t={t_current}: No solution")

        return all_P, all_S