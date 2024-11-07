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

class my_Karpsack():
    def __init__(self, num):
        self.max_weight = 100  # 背包的最大承重
        self.num_items = num  # 物品数量
        self.item_weights = [random.randint(1, 20) for _ in range(num)]  # 每个物品的重量
        self.item_values = [random.randint(1, 100) for _ in range(num)]  # 每个物品的价值


class my_TSP():
    def __init__(self, num):
        self.num_cities = num  # 城市数量
        self.city_positions = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(num)]  # 城市坐标





def execute_algorithm(algorithm, algorithm_name):
    start_time = time.time()
    result = algorithm()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{algorithm_name}耗时：{elapsed_time}秒")
    print(f"{algorithm_name}最优值：{result}")
    return elapsed_time, result

def run_Knapsack(object):
    new_knapsack = Knapsack(object.max_weight, object.num_items, object.item_weights, object.item_values)

    backtrack_time, result_backtrack = execute_algorithm(new_knapsack.backtrack_max, "回溯算法")

    branch_bound_time, result_branch_bound = execute_algorithm(new_knapsack.branch_bound_max, "分支定界算法")

    greedy_time, result_greedy_max = execute_algorithm(new_knapsack.greedy_max, "贪心算法")

    GA_time, best_fitness = execute_algorithm(new_knapsack.run_genetic_algorithm, "GA算法")

    PSO_time, best_fitness_pso = execute_algorithm(new_knapsack.particle_swarm_optimizer, "PSO算法")


    x = ["回溯算法", "分支定界算法", "贪心算法", "GA算法", "PSO算法"]
    y = [backtrack_time, branch_bound_time, greedy_time, GA_time, PSO_time]
    plt.bar(x, y, color=['red', 'blue', 'green', 'yellow', 'purple'])
    plt.title("不同算法耗时对比")
    plt.xlabel("算法")
    plt.ylabel("耗时(秒)")
    plt.savefig(f"D:/study/HomeWork/01背包/当n={object.num_items}时，不同算法耗时对比.png")
    plt.show()

    y_1 = [result_backtrack, result_branch_bound, result_greedy_max, best_fitness, best_fitness_pso]
    plt.bar(x, y_1, color=['red', 'blue', 'green', 'yellow', 'purple'])
    plt.title("不同算法最优值对比")
    plt.xlabel("算法")
    plt.ylabel("最优值")
    plt.savefig(f"D:/study/HomeWork/01背包/当n={object.num_items}时，不同算法最优值对比.png")
    plt.show()


def run_TSP(object):
    new_tsp = TSP(object.city_positions)

    backtrack_time, result_backtrack = execute_algorithm(new_tsp.backtrack_min, "回溯算法")

    branch_bound_time, result_branch_bound = execute_algorithm(new_tsp.branch_and_bound_min, "分支定界算法")

    greedy_time, result_greedy_min = execute_algorithm(new_tsp.greedy_solution, "贪心算法")

    GA_time, best_fitness = execute_algorithm(new_tsp.genetic_algorithm, "GA算法")

    PSO_time, best_fitness_pso = execute_algorithm(new_tsp.PSO, "PSO算法")

    x = ["回溯算法", "分支定界算法", "贪心算法", "GA算法", "PSO算法"]
    y = [backtrack_time, branch_bound_time, greedy_time, GA_time, PSO_time]
    plt.bar(x, y, color=['red', 'blue', 'green', 'yellow', 'purple'])
    plt.title("不同算法耗时对比")
    plt.xlabel("算法")
    plt.ylabel("耗时(秒)")
    plt.savefig(f"D:/study/HomeWork/TSP/当n={object.num_cities}时，不同算法耗时对比.png")
    plt.show()

    y_1 = [result_backtrack, result_branch_bound, result_greedy_min, best_fitness, best_fitness_pso]
    plt.bar(x, y_1, color=['red', 'blue', 'green', 'yellow', 'purple'])
    plt.title("不同算法最优值对比")
    plt.xlabel("算法")
    plt.ylabel("最优值")
    plt.savefig(f"D:/study/HomeWork/TSP/当n={object.num_cities}时，不同算法最优值对比.png")
    plt.show()













if __name__ == '__main__':

    mytest_1 = my_Karpsack(10)
    run_Knapsack(mytest_1)
    mytest_2 = my_Karpsack(12)
    run_Knapsack(mytest_2)
    mytest_3 = my_Karpsack(14)
    run_Knapsack(mytest_3)
    mytest_4 = my_Karpsack(16)
    run_Knapsack(mytest_4)
    mytest_5 = my_Karpsack(18)
    run_Knapsack(mytest_5)
    mytest_6 = my_Karpsack(20)
    run_Knapsack(mytest_6)
    mytest_7 = my_TSP(5)
    run_TSP(mytest_7)
    mytest_8 = my_TSP(8)
    run_TSP(mytest_8)
    mytest_9 = my_TSP(9)
    run_TSP(mytest_9)
    mytest_10 = my_TSP(10)
    run_TSP(mytest_10)
    mytest_12 = my_TSP(12)
    run_TSP(mytest_12)





