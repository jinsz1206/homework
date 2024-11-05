# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import random
from Knapsack import Knapsack
from TSP import TSP
import time
import matplotlib.pyplot as plt



random.seed(42)  # 设置随机种子
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

class my_number():
    def __init__(self, num):
        self.max_weight = 100  # 背包的最大承重
        self.num_items = num  # 物品数量
        self.item_weights = [random.randint(1, 20) for _ in range(num)]  # 每个物品的重量
        self.item_values = [random.randint(1, 100) for _ in range(num)]  # 每个物品的价值


def run_Knapsack(object):
    new_knapsack = Knapsack(object.max_weight, object.num_items, object.item_weights, object.item_values)
    backtrack_start_time = time.time()
    result_backtrack = new_knapsack.backtrack_max()
    backtrack_end_time = time.time()
    backtrack_time = backtrack_end_time - backtrack_start_time
    print(f"回溯算法耗时：{backtrack_time}秒")
    print(f"回溯算法最优值：{result_backtrack}")

    branch_bound_start_time = time.time()
    result_branch_bound = new_knapsack.branch_bound_max()
    branch_bound_end_time = time.time()
    branch_bound_time = branch_bound_end_time - branch_bound_start_time
    print(f"分支定界算法耗时：{branch_bound_time}秒")
    print(f"分支定界算法最优值：{result_branch_bound}")

    greedy_start_time = time.time()
    result_greedy_max = new_knapsack.greedy_max()
    greedy_end_time = time.time()
    greedy_time = greedy_end_time - greedy_start_time
    print(f"贪心算法耗时：{greedy_time}秒")
    print(f"贪心算法最优值：{result_greedy_max}")

    GA_start_time = time.time()
    best_individual, best_fitness = new_knapsack.run_genetic_algorithm(100, 50)
    GA_end_time = time.time()
    GA_time = GA_end_time - GA_start_time
    print(f"GA算法耗时：{GA_time}秒")
    print(f"GA最佳个体：{best_individual}")
    print(f"GA最佳个体适应度：{best_fitness}")

    PSO_start_time = time.time()
    best_solution_pso, best_fitness_pso = new_knapsack.particle_swarm_optimizer(100,50)
    PSO_end_time = time.time()
    PSO_time = PSO_end_time - PSO_start_time
    print(f"PSO算法耗时：{PSO_time}秒")
    print("PSO最优解:", best_solution_pso)
    print("PSO最佳适应度:", best_fitness_pso)

    x = ["回溯算法", "分支定界算法", "贪心算法", "GA算法", "PSO算法"]
    y = [backtrack_time, branch_bound_time, greedy_time, GA_time, PSO_time]
    plt.bar(x, y, color=['red', 'blue', 'green', 'yellow', 'purple'])
    plt.title("不同算法耗时对比")
    plt.xlabel("算法")
    plt.ylabel("耗时(秒)")
    plt.savefig(f"D:/study/当n={object.num_items}时，不同算法耗时对比.png")
    plt.show()

    y_1 = [result_backtrack, result_branch_bound, result_greedy_max, best_fitness, best_fitness_pso]
    plt.bar(x, y_1, color=['red', 'blue', 'green', 'yellow', 'purple'])
    plt.title("不同算法最优值对比")
    plt.xlabel("算法")
    plt.ylabel("最优值")
    plt.savefig(f"D:/study/当n={object.num_items}时，不同算法最优值对比.png")
    plt.show()




if __name__ == '__main__':
    plt.ion()  # 开启交互模式
    mytest_1 = my_number(10)
    run_Knapsack(mytest_1)
    mytest_2 = my_number(15)
    run_Knapsack(mytest_2)
    # mytest_3 = my_number(20)
    # run_Knapsack(mytest_3)
    # mytest_4 = my_number(21)
    # run_Knapsack(mytest_4)
    # mytest_5 = my_number(22)
    # run_Knapsack(mytest_5)
    # mytest_6 = my_number(23)
    # run_Knapsack(mytest_6)
    plt.ioff()  # 关闭交互模式

