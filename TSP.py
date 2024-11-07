
import random
import math
import numpy as np

random.seed(42)

class TSP:
    def __init__(self, citys):
        self.citys = citys # 城市坐标列表
        self.n = len(citys)    # 城市数量
        self.graph = self.distance_matrix() # 城市之间的距离矩阵



    def calculate_distance(self, city1, city2):
        """
        计算两个城市之间的距离
        :param city1:  城市1坐标
        :param city2:   city2坐标
        :return:     距离
        """

        return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)
    def distance_matrix(self):
        # 创建城市之间的距离矩阵
        graph = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    graph[i][j] = self.calculate_distance(self.citys[i], self.citys[j])
        return graph


    def backtrack_min(self):
        min_cost = float('inf')
        final_path = []

        def tsp_util(index, visited, count, cost, path):
            """
            :param index: 城市索引
            :param visited: 访问过的城市列表
            :param count: 访问的城市数量
            :param cost: 距离总和
            :param path: 城市路径
            :return: 退出
            """
            nonlocal min_cost, final_path
            if count == self.n and self.graph[index][0]:
                if cost + self.graph[index][0] < min_cost:
                    min_cost = cost + self.graph[index][0]
                    final_path = path + [0]
                return

            for i in range(self.n):
                if not visited[i] and self.graph[index][i]:
                    visited[i] = True
                    tsp_util(i, visited, count + 1, cost + self.graph[index][i], path + [i])
                    visited[i] = False

        visited = [False] * self.n
        visited[0] = True
        tsp_util(0, visited, 1, 0, [0])

        # return final_path, min_cost
        return  float(min_cost)









    def branch_and_bound_min(self):

        min_cost = float('inf')
        final_path = []


        def branch_and_bound(index, count, cost, visited,path):
            """

            :param index:   城市索引
            :param count:   访问的城市数量
            :param cost:    距离总和
            :param visited:     访问过的城市列表
            :param path:     城市路径
            :return:    退出
            """
            nonlocal min_cost, final_path
            if count == self.n and self.graph[index][0]:
                total_cost = cost + self.graph[index][0]
                if total_cost < min_cost:
                    min_cost = total_cost
                    final_path = path + [0]
                return

            for i in range(self.n):
                if not visited[i] and self.graph[index][i]:
                    new_cost = cost + self.graph[index][i]
                    if new_cost < min_cost:      # 剪枝
                        visited[i] = True
                        branch_and_bound(i, count + 1, new_cost, visited,path + [i])
                        visited[i] = False

        visited = [False] * self.n
        visited[0] = True
        branch_and_bound(0, 1, 0, visited,[0])

        # return final_path, min_cost
        return  float(min_cost)

    def greedy_solution(self):
        visited = [False] * self.n  # 记录城市是否被访问
        path = []  # 记录旅行路径
        total_distance = 0  # 记录总距离

        # 从第一个城市出发
        current_city = 0
        visited[current_city] = True
        path.append(current_city)

        for _ in range(1, self.n):
            next_city = -1
            min_distance = float('inf')
            # 寻找未访问的城市中，离当前城市最近的城市
            for j in range(self.n):
                if not visited[j] and self.graph[current_city][j] < min_distance:
                    min_distance = self.graph[current_city][j]
                    next_city = j

            visited[next_city] = True
            path.append(next_city)
            total_distance += min_distance
            current_city = next_city

        # 返回起始城市
        total_distance += self.graph[current_city][0]
        path.append(0)  # 回到起始城市

        # return path, total_distance
        return  float(total_distance)



    def generate_route(self):
        # 随机生成一个城市访问顺序
        route = list(range(1,self.n))
        random.shuffle(route)
        route.insert(0, 0)
        return route     # 返回一个随机的城市访问顺序



    def fitness(self, route):
        # 计算路径的适应度（路径长度）
        total_distance = sum(self.graph[route[i]][route[i + 1]] for i in range(len(route) - 1))
        total_distance += self.graph[route[-1]][route[0]]  # 回到起点
        return 1 / total_distance  # 适应度越高越好



    def breed(self, parent1, parent2):
        # 交叉生成新个体
        geneA = int(random.random() * len(parent1))
        geneB = int(random.random() * len(parent1))

        start, end = min(geneA, geneB), max(geneA, geneB)

        child = parent1[start:end] + [gene for gene in parent2 if gene not in parent1[start:end]]
        return child

    def mutate(self, route, mutation_rate=0.01):
        # 突变（交换两个城市的位置）
        for swapped in range(len(route)):
            if random.random() < mutation_rate:
                swap_with = int(random.random() * len(route))

                route[swapped], route[swap_with] = route[swap_with], route[swapped]

    def select_elites(self, population, elite_fraction=0.1):
        # 选择精英个体
        elite_count = int(elite_fraction * len(population))
        return population[:elite_count]

    def genetic_algorithm(self, population_size=10, generations=20):
        # 初始化种群
        population = [self.generate_route() for _ in range(population_size)]

        for generation in range(generations):
            population = sorted(population, key=self.fitness, reverse=True)
            new_population = []

            # 选择精英
            elites = self.select_elites(population)
            new_population.extend(elites)

            for i in range(len(population) - len(elites)):
                parent1 = random.choice(population[:50])
                parent2 = random.choice(population[:50])
                child = self.breed(parent1, parent2)
                self.mutate(child)
                new_population.append(child)

            population = new_population

        best_route = sorted(population, key=self.fitness, reverse=True)[0]
        best_route.append(0)
        best_cost = 1 / self.fitness(best_route)
        # return best_route, best_cost
        return  float(best_cost)


    def PSO(self, population_size=10, iterations=20):
        # 初始化粒子群
        particles = np.array([self.generate_route() for _ in range(population_size)])
        personal_best_positions = particles.copy()
        personal_best_fitness = np.array([self.fitness(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_fitness)]
        global_best_fitness = np.min(personal_best_fitness)

        for _ in range(iterations):
            for i in range(population_size):
                r1, r2 = random.random(), random.random()

                # 速度更新（使用简单随机交换方式保持路径有效性）
                velocities = np.zeros(len(particles[0]), dtype=int)
                if random.random() < 0.5:  # 50%概率随机交换两个城市
                    idx1, idx2 = random.sample(range(len(particles[i])), 2)
                    velocities[idx1], velocities[idx2] = velocities[idx2], velocities[idx1]

                # 更新位置
                particles[i] = (particles[i] + velocities) % self.n
                particles[i] = np.unique(particles[i], return_index=True)[1]  # 确保城市不重复
                while len(particles[i]) < self.n:  # 补充缺失的城市
                    missing_city = list(set(range(self.n)) - set(particles[i]))
                    particles[i] = np.concatenate((particles[i], np.random.choice(missing_city, 1)))

                # 更新个人最优解
                fitness = self.fitness(particles[i])
                if fitness > personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = particles[i].copy()

                # 更新全局最优解
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particles[i].copy()
        best_position = global_best_position.tolist()
        best_position.append(0)


        # return  best_position, 1/global_best_fitness
        return  float(1/global_best_fitness)














if __name__ == "__main__":
    citys = [
        (0, 0),   # 城市0坐标
        (10, 10),  # 城市1坐标
        (0, 10), # 城市2坐标
        (10, 0)   # 城市3坐标
    ]
    tsp = TSP(citys)
    path, cost = tsp.backtrack_min()
    print(f"回溯法得到的最优路径：{path}")
    print(f"回溯法得到的最优成本: {cost}")
    path_branch, cost_branch = tsp.branch_and_bound_min()
    print("分支限界法得到的最优路径:", path_branch)
    print("分支限界法得到的最优成本:", cost_branch)
    greedy_path, greedy_cost = tsp.greedy_solution()
    print("贪婪法得到的最优路径:", greedy_path)
    print("贪婪法得到的最优成本:", greedy_cost)
    best_route, best_cost = tsp.genetic_algorithm(population_size=10, generations=20)
    print("遗传算法得到的最优路径:", best_route)
    print("遗传算法得到的最优成本:", best_cost)
    best_route_pso, best_cost_pso = tsp.PSO(population_size=10, iterations=20)
    print("粒子群算法得到的最优路径:", best_route_pso)
    print("粒子群算法得到的最优成本:", best_cost_pso)



