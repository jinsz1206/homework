import random
import numpy as np



class Knapsack:

    random.seed(0)  # 设置随机种子


    def __init__(self, max_weight, num_items, item_weights, item_values):
        """
        初始化背包问题参数
        :param max_weight:  最大承重
        :param num_items:   物品数量
        :param item_weights:  单个物品重量
        :param item_values:  单个物品价值
        """
        self.max_weight = max_weight
        self.num_items = num_items
        self.item_weights = item_weights
        self.item_values = item_values



    def backtrack_max(self, current_item = 0, remaining_weight = None, current_value = 0):
        """
        回溯法求解最大价值
        :param current_item:  当前考察的物品索引
        :param remaining_weight:  剩余承重
        :param current_value:  当前价值
        :return:    最大价值
        """
        if remaining_weight is None:
            remaining_weight = self.max_weight

        # 终止条件：已经考察完所有物品或剩余承重不足以放下剩余物品
        if current_item == self.num_items or remaining_weight <= 0:
            return current_value

        # 不选择当前物品的情况
        value_without_current = self.backtrack_max(current_item + 1, remaining_weight, current_value)

        # 选择当前物品的情况
        if self.item_weights[current_item] <= remaining_weight:
            value_with_current = self.backtrack_max(current_item + 1, remaining_weight - self.item_weights[current_item], current_value + self.item_values[current_item])
        else:
            value_with_current = 0  # 如果承重不足，当前物品不能选择

        return max(value_without_current, value_with_current)





    def branch_bound_max(self):
        """
        分支定界法求解最大价值
        :return:    最大价值
        """


        # 创建一个优先队列，初始状态为0，当前价值为0，当前重量为0
        from queue import PriorityQueue

        # 自定义节点类
        class Node:
            def __init__(self, level, value, weight):
                self.level = level  # 当前考察的物品索引
                self.value = value  # 当前价值
                self.weight = weight  # 当前重量

            # 为了实现优先队列的排序，定义一个比较函数
            def __lt__(self, other):
                return self.value > other.value  # 根据价值进行排序

        pq = PriorityQueue()
        pq.put(Node(0, 0, 0))  # 将初始节点放入优先队列
        max_value = 0

        while not pq.empty():
            node = pq.get()

            if node.level < self.num_items:
                # 计算当前物品的重量和价值
                current_weight = node.weight + self.item_weights[node.level]
                current_value = node.value + self.item_values[node.level]

                # 如果当前物品可以放入背包
                if current_weight <= self.max_weight:
                    max_value = max(max_value, current_value)  # 更新最大价值

                    # 将包含当前物品的节点放入队列
                    pq.put(Node(node.level + 1, current_value, current_weight))

                # 将不包含当前物品的节点放入队列
                pq.put(Node(node.level + 1, node.value, node.weight))

        return max_value



    def greedy_max(self):
        """
        贪心算法求解最大价值
        :return:    最大价值
        """
        # 计算价值/重量比并进行排序
        value_weight_ratio = [(self.item_values[i] / self.item_weights[i], self.item_weights[i], self.item_values[i]) for i in range(self.num_items)]
        value_weight_ratio.sort(reverse=True, key=lambda x: x[0])  # 按照价值/重量比从高到低排序

        total_value = 0
        remaining_weight = self.max_weight

        for ratio, weight, value in value_weight_ratio:
            if remaining_weight >= weight:
                remaining_weight -= weight
                total_value += value  # 完整放入物品
            else:
                continue  # 剩余承重不足，物品不能选择
        return total_value



    def greedy_min(self):
        # 计算价值/重量比并进行排序
        value_weight_ratio = [(self.item_values[i] / self.item_weights[i], self.item_weights[i], self.item_values[i]) for i in range(self.num_items)]
        value_weight_ratio.sort(reverse=False, key=lambda x: x[0])  # 按照价值/重量比从高到低排序

        total_value = 0
        remaining_weight = self.max_weight

        for ratio, weight, value in value_weight_ratio:
            if remaining_weight >= weight:
                remaining_weight -= weight
                total_value += value  # 完整放入物品
            else:
                continue  # 剩余承重不足，物品不能选择
        return total_value






    # 遗传算法
    def initialize_population(self, population_size):
        """
        初始化种群
        :param population_size:     种群大小
        :return:    种群
        """
        return [[random.randint(0, 1) for _ in range(self.num_items)] for _ in range(population_size)]

    def fitness(self, individual):
        """
        计算适应度
        :param individual:    个体
        :return:     适应度
        """
        total_weight = sum(self.item_weights[i] for i in range(self.num_items) if individual[i] == 1)
        total_value = sum(self.item_values[i] for i in range(self.num_items) if individual[i] == 1)

        if total_weight > self.max_weight:
            return 0  # 超重，适应度为0
        return total_value  # 适应度为总价值

    def select(self, population):
        """
        轮盘赌选择
        :param population:   种群
        :return:     选择的个体
        """
        # 轮盘赌选择
        total_fitness = sum(self.fitness(individual) for individual in population)
        pick = random.uniform(0, total_fitness)
        current = 0
        for individual in population:
            current += self.fitness(individual)
            if current > pick:
                return individual

    def crossover(self, parent1, parent2):
        """
        交叉
        :param parent1:  父1
        :param parent2:  父2
        :return:    交叉后的个体
        """
        point = random.randint(1, self.num_items - 1)
        return parent1[:point] + parent2[point:]

    def mutate(self, individual):
        """
        变异
        :param individual:       个体
        :return:         变异后的个体
        """
        mutation_rate = 0.01
        return [gene if random.random() > mutation_rate else 1 - gene for gene in individual]





    def run_genetic_algorithm(self, population_size, generations):
        """
             遗传算法
        :param population_size:     种群大小
        :param generations:         迭代次数
        :return:         最佳个体及其适应度
        """
        population = self.initialize_population(population_size)

        for generation in range(generations):
            new_population = []
            for _ in range(population_size):
                parent1 = self.select(population)
                parent2 = self.select(population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            population = new_population

        # 找到最佳个体
        best_individual = max(population, key=self.fitness)
        return best_individual, self.fitness(best_individual)






    def particle_swarm_optimizer(self, num_particles=10, max_iter=20):
        """
        粒子群算法
        :param num_particles:    粒子数量
        :param max_iter:         迭代次数
        :return:     最佳个体及其适应度
        """
        particles = np.random.randint(2, size=(num_particles, self.num_items)) # 初始化粒子位置
        velocities = np.random.randn(num_particles, self.num_items)  # 初始化粒子速度
        personal_best_positions = particles.copy()
        personal_best_fitness = np.array([self.fitness(p) for p in particles])
        global_best_position = personal_best_positions[np.argmax(personal_best_fitness, axis=0)]
        global_best_fitness = np.max(personal_best_fitness)

        for _ in range(max_iter):
            for i in range(num_particles):
                # 更新速度和位置
                r1, r2 = random.random(), random.random()
                velocities[i] = (velocities[i] +
                                 r1 * (personal_best_positions[i] - particles[i]) +
                                 r2 * (global_best_position - particles[i]))
                # 限制速度
                velocities[i] = np.clip(velocities[i], -1, 1)
                # 更新位置
                particles[i] = (particles[i] + velocities[i]).clip(0, 1).astype(int)

                # 更新个人最优解
                fitness = self.fitness(particles[i])
                if fitness > personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = particles[i].copy()

                # 更新全局最优解
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particles[i].copy()
        best_position_list = global_best_position.tolist()

        return best_position_list, global_best_fitness








if __name__ == '__main__':
    max_weight = 6
    num_items = 4
    item_weights = [1,2,3,4]
    item_values = [20,30,30,80]
    new_knapsack = Knapsack(max_weight, num_items, item_weights, item_values)
    result_backtrack = new_knapsack.backtrack_max()
    result_branch_bound = new_knapsack.branch_bound_max()
    result_greedy_max = new_knapsack.greedy_max()
    result_greedy_min = new_knapsack.greedy_min()
    best_individual, best_fitness = new_knapsack.run_genetic_algorithm(10, 5)
    print(f"GA最佳个体：{best_individual}")
    print(f"GA最佳个体适应度：{best_fitness}")
    print(f"回溯算法最优值：{result_backtrack}")
    print(f"分支定界算法最优值：{result_branch_bound}")
    print(f"贪心算法最优值：{result_greedy_max}")
    print(f"贪心算法最差值：{result_greedy_min}")
    best_solution_pso, best_fitness_pso = new_knapsack.particle_swarm_optimizer()
    print("PSO最优解:", best_solution_pso)
    print("PSO最佳适应度:", best_fitness_pso)
