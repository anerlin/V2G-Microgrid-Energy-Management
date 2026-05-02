import pulp
import numpy as np

class V2GModel:
    def __init__(self, n_evs, T, ev_params, price, reg_signal):
        self.n_evs = n_evs
        self.T = T
        self.ev_params = ev_params
        self.price = price
        self.reg_signal = reg_signal
        self.beta_degradation = 0.01
        self.R_base = 40
        self.xi = 0.1

    def build_model(self, external_load=None, is_online=False, t_current=0):
        model = pulp.LpProblem("V2G_Microgrid", pulp.LpMinimize)

        P = {}
        S = {}

        for i in range(self.n_evs):
            for t in range(self.T):
                P[(i, t)] = pulp.LpVariable(
                    f"P_{i}_{t}",
                    self.ev_params['P_dch_max'],
                    self.ev_params['P_ch_max']
                )
                S[(i, t)] = pulp.LpVariable(
                    f"S_{i}_{t}",
                    self.ev_params['S_min'],
                    self.ev_params['S_max']
                )

        # 目标函数：充电成本
        charging_cost = 0
        for i in range(self.n_evs):
            for t in range(self.T):
                charging_cost += self.price[t] * P[(i, t)]

        # 总功率
        total_power = pulp.lpSum([P[(i, t)] for i in range(self.n_evs) for t in range(self.T)])
        if external_load is not None:
            total_load = sum(external_load) + total_power
        else:
            total_load = total_power

        # 绝对值处理
        total_load_abs = pulp.LpVariable("total_load_abs", lowBound=0)
        model += total_load_abs >= total_load
        model += total_load_abs >= -total_load

        # 奖励
        reward = (self.R_base * self.T / 1000) - self.xi * total_load_abs

        # 退化惩罚
        degradation = 0
        for i in range(self.n_evs):
            for t in range(self.T):
                P_abs = pulp.LpVariable(f"P_abs_{i}_{t}", lowBound=0)
                model += P_abs >= P[(i, t)]
                model += P_abs >= -P[(i, t)]
                degradation += self.beta_degradation * P_abs

        model += charging_cost - reward + degradation

        dt = 1.0 / 60.0

        # SoC 约束
        for i in range(self.n_evs):
            S_init = np.random.uniform(0.4, 0.5)
            model += S[(i, 0)] == S_init

            for t in range(1, self.T):
                eta_ch = self.ev_params['eta_ch']
                eta_dch = self.ev_params['eta_dch']
                C = self.ev_params['battery_capacity']

                P_pos = pulp.LpVariable(f"P_pos_{i}_{t}", 0, self.ev_params['P_ch_max'])
                P_neg = pulp.LpVariable(f"P_neg_{i}_{t}", 0, -self.ev_params['P_dch_max'])
                model += P[(i, t-1)] == P_pos - P_neg

                # 修复除法：将 P_neg / eta_dch 改写成 P_neg * (1/eta_dch)
                inv_eta_dch = 1.0 / eta_dch
                model += C * (S[(i, t)] - S[(i, t-1)]) == dt * (
                    eta_ch * P_pos + inv_eta_dch * P_neg
                )

        # 充电需求约束
        for i in range(self.n_evs):
            S_req = np.random.uniform(0.85, 0.95)
            model += S[(i, self.T-1)] >= S_req

        # 功率平衡约束
        if external_load is not None:
            for t in range(self.T):
                model += self.reg_signal[t] + pulp.lpSum([P[(i, t)] for i in range(self.n_evs)]) + external_load[t] == 0

        return model, P, S

    def solve(self, external_load=None, is_online=False, t_current=0):
        try:
            model, P, S = self.build_model(external_load, is_online, t_current)
        except Exception as e:
            print(f"Build model error: {e}")
            return None, None, None

        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=30)
        model.solve(solver)

        if model.status == pulp.LpStatusOptimal:
            cost = pulp.value(model.objective)
            print(f"  V2G optimal cost: ${cost:.2f}")
            return cost, P, S
        else:
            print(f"  V2G: No feasible solution, status={model.status}")
            return None, None, None