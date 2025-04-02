import numpy as np
import random
from deap import base, creator, tools, algorithms

# ================== 问题参数配置 ==================
MONTHS = 12
COSTS = {
    'production': 800,   # 单位生产成本
    'holding': 50,       # 库存持有成本/月
    'shortage': 1500,    # 缺货成本/单位
    'surplus': 300,      # 过剩成本/单位
    'hire': 2000,        # 雇佣成本/人
    'layoff': 3000       # 解雇成本/人
}
CAPACITY = [2800]*12     # 每月最大产能
DEMAND = [2400, 2600, 3000, 3200, 3500, 3800, 
         4000, 3900, 3600, 3200, 2800, 2500]  # 预测需求
MAX_INV = 4000           # 最大库存容量
MIN_LABOR = 50           # 最少劳动力

# ================== 遗传算法参数 ==================
POP_SIZE = 200
GEN_NUM = 300
CX_PROB = 0.85
MUT_PROB = 0.15

# ================== 染色体结构定义 ==================
def create_individual():
    ind = []
    for _ in range(MONTHS):
        ind.append(random.uniform(0.8*DEMAND[_], 1.2*DEMAND[_]))  # X_t
        ind.append(random.uniform(0, MAX_INV*0.5))               # I_t
        ind.append(random.randint(MIN_LABOR, 200))                # L_t
    return ind

# ================== 适应度函数 ==================
def evaluate(individual):
    total_cost = 0
    prev_labor = individual[2]  # 初始劳动力
    
    for t in range(MONTHS):
        X_t = individual[3*t]
        I_t = individual[3*t+1]
        L_t = individual[3*t+2]
        
        # 生产约束
        production_cost = COSTS['production'] * X_t
        
        # 库存计算
        if t == 0:
            I_prev = 0
        else:
            I_prev = individual[3*(t-1)+1]
            
        sales = min(DEMAND[t], X_t + I_prev)
        shortage = max(0, DEMAND[t] - (X_t + I_prev))
        surplus = max(0, (X_t + I_prev) - DEMAND[t])
        
        # 库存成本
        holding_cost = COSTS['holding'] * I_t
        
        # 劳动力变化成本
        if L_t > prev_labor:
            labor_cost = COSTS['hire'] * (L_t - prev_labor)
        else:
            labor_cost = COSTS['layoff'] * (prev_labor - L_t)
            
        prev_labor = L_t
        
        # 惩罚项
        penalties = 0
        if X_t > CAPACITY[t]:  # 产能超限
            penalties += 1e6 * (X_t - CAPACITY[t])**2
        if I_t > MAX_INV:      # 库存超限
            penalties += 1e6 * (I_t - MAX_INV)**2
        if L_t < MIN_LABOR:    # 最低人力
            penalties += 1e6 * (MIN_LABOR - L_t)**2
            
        total_cost += (production_cost + holding_cost + 
                      COSTS['shortage']*shortage + 
                      COSTS['surplus']*surplus + 
                      labor_cost + penalties)
    
    return (total_cost,)

# ================== 遗传算法设置 ==================
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=4000, eta=20)
toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=4000, eta=20, indpb=0.1)
toolbox.register("select", tools.selNSGA2)

# ================== 优化执行 ==================
def main():
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    # 动态调整参数
    for gen in range(GEN_NUM):
        if gen > 50:  # 动态调整策略
            CX_PROB = 0.85 * (0.98 ** (gen//10))
            MUT_PROB = 0.15 * (1.03 ** (gen//10))
            
        pop, log = algorithms.eaMuPlusLambda(
            pop, toolbox, 
            mu=int(POP_SIZE*0.8), 
            lambda_=int(POP_SIZE*0.2),
            cxpb=CX_PROB, mutpb=MUT_PROB,
            ngen=1, stats=stats, halloffame=hof
        )
        
        print(f"Gen {gen}: Min Cost = {log.chapters['min'].select('min')[-1]:,.2f}")
    
    return pop, hof, log

# ================== 结果输出 ==================
if __name__ == "__main__":
    final_pop, best_ind, log = main()
    best_plan = best_ind[0]
    
    print("\n最优生产计划:")
    print("月份 | 产量 | 库存 | 劳动力")
    for t in range(MONTHS):
        print(f"{t+1:2}月 | {best_plan[3*t]:6.0f} | {best_plan[3*t+1]:6.0f} | {best_plan[3*t+2]:3.0f}")
    print(f"\n预估总成本: ¥{best_ind[0].fitness.values[0]:,.2f}")
