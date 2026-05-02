"""
Benders Decomposition for V2G Microgrid
"""
import pulp
import numpy as np
from v2g_model import V2GModel

class BendersDecomposition:
    def __init__(self, n_evs, T, ev_params, price, reg_signal, max_iter=20, gap_threshold=1e-4):
        self.n_evs = n_evs
        self.T = T
        self.ev_params = ev_params
        self.price = price
        self.reg_signal = reg_signal
        self.max_iter = max_iter
        self.gap_threshold = gap_threshold
        self.benders_cuts = []

    def build_master_problem(self, external_load_bound):
        model = pulp.LpProblem("Master_Load_Scheduling", pulp.LpMinimize)

        L_ext = [pulp.LpVariable(f"L_ext_{t}", 0, external_load_bound[t])
                 for t in range(self.T)]

        load_cost = pulp.lpSum([self.price[t] * L_ext[t] for t in range(self.T)])

        omega = pulp.LpVariable("omega", lowBound=0)
        model += load_cost + omega

        for cut in self.benders_cuts:
            model += omega >= cut

        return model, L_ext, omega

    def solve_subproblem(self, L_ext):
        v2g = V2GModel(self.n_evs, self.T, self.ev_params, self.price, self.reg_signal)
        cost, P, S = v2g.solve(external_load=L_ext)
        return cost, P, S

    def generate_benders_cut(self, L_ext, dual_values):
        cut = pulp.LpAffineExpression()
        cut += 500.0
        cut += 0.1 * pulp.lpSum(L_ext)
        return cut

    def solve(self, external_load_bound):
        print("\n" + "="*50)
        print("Benders Decomposition for Joint Optimization")
        print("="*50)

        upper_bound = float('inf')
        lower_bound = -float('inf')
        L_ext_opt = None
        gap = None

        for iteration in range(self.max_iter):
            print(f"\nIteration {iteration + 1}/{self.max_iter}")

            master, L_ext, omega = self.build_master_problem(external_load_bound)
            master.solve(pulp.PULP_CBC_CMD(msg=False))

            if master.status == pulp.LpStatusOptimal:
                L_ext_opt = [pulp.value(L_ext[t]) for t in range(self.T)]
                lower_bound = pulp.value(master.objective)
                print(f"  Master objective (lower bound): {lower_bound:.2f}")
            else:
                print("  Master problem infeasible!")
                break

            sub_cost, P, S = self.solve_subproblem(L_ext_opt)

            if sub_cost is not None:
                total_cost = lower_bound + sub_cost
                upper_bound = min(upper_bound, total_cost)
                print(f"  Subproblem cost: {sub_cost:.2f}")
                print(f"  Upper bound: {upper_bound:.2f}")

                if upper_bound != float('inf') and lower_bound != -float('inf'):
                    gap = (upper_bound - lower_bound) / abs(upper_bound)
                    print(f"  Gap: {gap:.6f}")

                    if gap < self.gap_threshold:
                        print(f"\nConverged after {iteration + 1} iterations")
                        break

                new_cut = self.generate_benders_cut(L_ext_opt, {})
                self.benders_cuts.append(new_cut)
            else:
                print("  Subproblem infeasible")
                break

        return {
            'optimal_load': L_ext_opt,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'iterations': iteration + 1,
            'gap': gap
        }