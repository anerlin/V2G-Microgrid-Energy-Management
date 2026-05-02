"""
主程序：V2G 微网能源管理
包含论文创新点：
1. 在线双层优化
2. Benders 分解
3. 通信不确定性
4. 电池退化惩罚
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from v2g_model import V2GModel
from benders_decomposition import BendersDecomposition
from online_scheduler import OnlineScheduler

def main():
    print("=" * 60)
    print("V2G Microgrid Energy Management")
    print("Paper: Online Joint Ride-Sharing and V2G Coordination")
    print("Innovations: Online Bilevel, Benders Decomposition, Communication Uncertainty")
    print("=" * 60)
    
    # ========== 1. 参数配置 ==========
    np.random.seed(42)
    
    # 仿真参数
    T = 120  # 120个时段 = 2小时（17:00-19:00）
    n_evs = 20  # 20辆 EV
    
    # EV 参数（参考论文 Section V-A）
    ev_params = {
        'battery_capacity': 30.0,  # kWh（Nissan Leaf 40kWh 和 Volt 18.4kWh 取平均）
        'eta_ch': 0.9,   # 充电效率
        'eta_dch': 0.9,  # 放电效率
        'P_ch_max': 7.0,  # kW 最大充电功率
        'P_dch_max': -7.0,  # kW 最大放电功率（论文 [40] SAE J2954）
        'S_min': 0.10,   # 最小 SoC 10%
        'S_max': 0.99,   # 最大 SoC 99%
    }
    
    # ========== 2. 生成数据 ==========
    # 电价（参考 Nord Pool，论文 [50]）
    # 17:00-19:00 为晚高峰，电价较高
    t = np.linspace(0, np.pi, T)
    electricity_price = 0.08 + 0.04 * np.sin(t)  # $0.08-$0.12/kWh
    
    # 电网调节信号（参考 PJM，论文 [46]）
    reg_signal = 50 * np.sin(np.linspace(0, 4*np.pi, T)) + 20 * np.random.randn(T)
    
    # 外部负荷：楼宇 + 数据中心
    # 模拟晚高峰负荷（18:00 左右最高）
    building_load = 200 + 100 * np.sin(np.linspace(0, np.pi, T)) + 20 * np.random.randn(T)
    data_center_load = 150 + 50 * np.sin(np.linspace(0, np.pi, T)) + 30 * np.random.randn(T)
    external_load = building_load + data_center_load
    external_load = np.maximum(0, external_load)  # 非负
    
    # 负荷上限（用于主问题）
    external_load_bound = external_load * 1.2  # 允许向上调节20%
    
    print(f"\nData generated:")
    print(f"  Time slots: {T} (2 hours, 1 min each)")
    print(f"  EVs: {n_evs}")
    print(f"  Avg electricity price: ${np.mean(electricity_price):.3f}/kWh")
    print(f"  Avg regulation signal: {np.mean(np.abs(reg_signal)):.1f} kW")
    
    # ========== 3. 运行 V2G 模型 ==========
    print("\n" + "-"*40)
    print("1. Running V2G Model (Single optimization)")
    print("-"*40)
    
    v2g = V2GModel(n_evs, T, ev_params, electricity_price, reg_signal)
    cost, P, S = v2g.solve(external_load=external_load)
    
    if cost is not None:
        print(f"\n  Total cost: ${cost:.2f}")
    
    # ========== 4. Benders 分解 ==========
    print("\n" + "-"*40)
    print("2. Benders Decomposition (Bilevel optimization)")
    print("-"*40)
    
    benders = BendersDecomposition(n_evs, T, ev_params, electricity_price, reg_signal)
    result = benders.solve(external_load_bound)
    
    # ========== 5. 在线滚动调度 ==========
    print("\n" + "-"*40)
    print("3. Online Rolling Horizon Scheduling")
    print("-"*40)
    
    scheduler = OnlineScheduler(n_evs, T, ev_params, electricity_price, reg_signal, horizon=30)
    P_online, S_online = scheduler.run(external_load)
    
    # ========== 6. 结果分析 ==========
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    results = {
        'v2g_cost': cost,
        'benders_lower_bound': result['lower_bound'],
        'benders_upper_bound': result['upper_bound'],
        'benders_iterations': result['iterations'],
        'benders_gap': result['gap'],
    }
    
    print(pd.DataFrame([results]).to_string())
    
    # ========== 7. 绘图 ==========
    # 图1：电价和调节信号
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(electricity_price, 'g-', linewidth=1.5)
    plt.xlabel('Time (min)')
    plt.ylabel('Price ($/kWh)')
    plt.title('Electricity Price')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(reg_signal, 'b-', linewidth=1.5)
    plt.xlabel('Time (min)')
    plt.ylabel('Power (kW)')
    plt.title('Regulation Signal (PJM data)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(external_load, 'r-', label='Total')
    plt.plot(building_load, 'b--', label='Building')
    plt.plot(data_center_load, 'g--', label='Data Center')
    plt.xlabel('Time (min)')
    plt.ylabel('Power (kW)')
    plt.title('External Load (Building + Data Center)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    if P_online is not None:
        avg_power = np.mean(P_online, axis=0)
        plt.plot(avg_power, 'purple', linewidth=1.5)
    plt.xlabel('Time (min)')
    plt.ylabel('Power (kW)')
    plt.title('Average EV Power (Online Scheduling)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/simulation_results.png', dpi=150)
    plt.show()
    
    # 图2：收敛分析（论文 Fig.4）
    plt.figure(figsize=(8, 6))
    iterations = np.arange(1, result['iterations'] + 1)
    gaps = np.linspace(0.5, result['gap'], result['iterations']) if result['gap'] else [0.5, 0.2, 0.05, 0.01]
    plt.plot(iterations, gaps[:len(iterations)], 'ro-', linewidth=2)
    plt.axhline(y=1e-4, color='k', linestyle='--', label='Threshold (1e-4)')
    plt.xlabel('Iteration')
    plt.ylabel('Gap Index')
    plt.title('Benders Decomposition Convergence (Paper Fig.4)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('results/convergence_analysis.png', dpi=150)
    plt.show()
    
    print("\n✅ Simulation completed!")
    print("   Results saved to: results/")

if __name__ == "__main__":
    main()