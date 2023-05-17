from __future__ import annotations
import numpy as np
import random
import time
from typing import List
from statistics import mean
from dataclasses import dataclass
from copy import deepcopy

GRAPH_SIZE = 5
X_MIN = -600
X_MAX = 600
Y_MIN = -400
Y_MAX = 400
MIN_DISTANCE_BETWEEN_POINTS = 0.4
VIZU_SCALING = 0.5
K_TRUCKS = 1
K_MEANS_SLEEP = 0


def euclidian(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a-b)

random.seed(42)
def generate_coordinates(n_nodes: int, x_coords_range, y_coords_range, min_distance: float):
    generated_coords = [(0, 0)]
    for i in range(n_nodes - 1):
        is_ok = False
        while not is_ok:
            coord = (random.uniform(*x_coords_range), random.uniform(*y_coords_range))
            is_ok = True
            for node in generated_coords:
                if euclidian(coord, node) < min_distance:
                    is_ok = False
        generated_coords.append(coord)
    return generated_coords

points = generate_coordinates(GRAPH_SIZE, (X_MIN, X_MAX), (Y_MIN, Y_MAX), MIN_DISTANCE_BETWEEN_POINTS)
random.seed()

def select_initial_centroids(points: list, k: int, display) -> list:
    #Initialize list of centroids with a first random chosen one.
    first_centroid = list(random.choice(points))
    centroids = [first_centroid]
    for i_centroid in range(k - 1):
        print('GENERATED:', i_centroid)
        #Store minimum distances of each points to a centroid.
        all_distances = []
        for i_point, point in enumerate(points):
            min_distance = float('inf')
            best_i_centroid = 0
            #Compute the lowest distance to a previously selected
            #centroid from this point.
            for i_centroid in range(len(centroids)):
                distance = euclidian(point, centroids[i_centroid])
                if distance < min_distance:
                    min_distance = distance
                    best_i_centroid = i_centroid

            if display:
                time.sleep(K_MEANS_SLEEP)
            all_distances.append(min_distance)
        
        #Select the point with maximum distance as our new centroid
        next_centroid = list(points[np.argmax(np.array(all_distances))])
        centroids.append(next_centroid)
    return centroids

@dataclass
class SubPoint():
    global_index: int
    coordinates: tuple

def calc_centroid(points):
    points = np.array(points)
    length = points.shape[0]
    sum_x = np.sum(points[:, 0])
    sum_y = np.sum(points[:, 1])
    return [sum_x/length, sum_y/length]

def generate_clusters(points, centroids, display) -> List[List[SubPoint]]:
    centroids_childs = dict()
    centroid_changed = True

    while centroid_changed:
        for i in range(len(centroids)):
            centroids_childs[i] = []

        for i_point, point in enumerate(points):
            #Get the closest centroid
            min_distance = float('inf')
            best_i_centroid = 0
            for i_centroid in range(len(centroids)):
                distance = euclidian(point, centroids[i_centroid])
                if distance < min_distance:
                    best_i_centroid = i_centroid
                    min_distance = distance

            #Create subpoint to keep track of the global index of the node
            #Global index is i+1 because i = 0 = depot
            subpoint = SubPoint(i_point + 1, point)
            centroids_childs[best_i_centroid].append(subpoint)

        centroid_changed = False
        #Update centroids values
        for i_centroid, centroid in enumerate(centroids):
            childs_coordinates = [subpoint.coordinates for subpoint in centroids_childs[i_centroid]]
            new_centroid_value = calc_centroid(childs_coordinates)
            if new_centroid_value != centroid:
                centroid_changed = True
                centroid[0] = new_centroid_value[0]
                centroid[1] = new_centroid_value[1]
    clusters = [centroids_childs[centroid] for centroid in centroids_childs.keys()]
    return clusters

def generate_distance_matrix(points: list):
    length = len(points)
    matrix = np.empty((length, length))
    matrix[:] = np.nan
    for i in range(length):
        for j in range(length):
            if i != j:
                dist = euclidian(points[i], points[j])
                matrix[i][j] = dist
                matrix[j][i] = dist
    return matrix

def generate_graphs_from_clusters(clusters):
    graphs = []
    for cluster in clusters:
        #Start by adding the depot point
        points_coordinates = [points[0]]
        points_indexes = [0]
        for subpoint in cluster:
            points_coordinates.append(subpoint.coordinates)
            points_indexes.append(subpoint.global_index)
        distance_matrix = generate_distance_matrix(points_coordinates)
        graphs.append((points_indexes, distance_matrix))
    return graphs

points_without_0 = list(points)
points_without_0.pop(0)
centroids = select_initial_centroids(points_without_0, K_TRUCKS, False)
clusters = generate_clusters(points_without_0, centroids, False)
graphs = generate_graphs_from_clusters(clusters)
print(graphs[0])

class ALGO_neightbours():
    def __init__(self, graph, start: int) -> None:
        self.graph = deepcopy(graph)
        self.start = start
        self.best_length = 0
        self.current_node = 0
        self.path_history = [0]

    def run(self) -> list:   
        while len(self.path_history) < (len(self.graph)):
            actual_graph = self.graph[self.current_node]
            for i in self.path_history:
                actual_graph[i] = float("inf")
            min_distance = np.nanmin(actual_graph)
            next_node = np.where(actual_graph == min_distance)[0][0]
            self.path_history.append(next_node)
            self.current_node = next_node
            self.best_length += min_distance
        
        self.path_history.append(0)
        self.best_length += self.graph[self.current_node][0]
        return self.best_length,self.path_history    


#Evaporation factor for global update of pheromones
RHO = 0.1
#Evaporation factor for local update of pheromones
KAPPA = RHO
#Q
OMICRON = 1
#Impact of pheromones
ALPHA = 1
#Impact of weights
BETA = 2
#Initial pheromones
TAU = 1
#Exploration/exploitation trade off
EPSILON = 1

class Ant():
    def __init__(self, real_nodes_indexes, initial_tau, init_pos: int) -> None:
        self.real_nodes_indexes = real_nodes_indexes
        self.initial_tau = initial_tau
        self.init_pos = init_pos
        self.current_pos = init_pos
        self.path_history = [init_pos]
        self.can_continue = True

    def add_to_path_history(self, node) -> None:
        self.path_history.append(node)

    def move(self, dest: list, pheromones: list) -> None:
        choice = random.uniform(0, 1)

        if choice <= EPSILON:
            dest_picked = self.exploitation(dest, pheromones[self.current_pos])
        else:
            dest_picked = self.biased_exploration(dest, pheromones[self.current_pos])

        if dest_picked is None:
            self.add_to_path_history(self.init_pos)
            self.can_continue = False
            return

        self.update_pheromones_locally(pheromones, dest_picked)
        self.current_pos = dest_picked
        self.add_to_path_history(self.current_pos)

    def exploitation(self, dest: list, pheromones: list) -> int:
        current_best_viability = None
        current_best_dest = None
        for i in range(len(dest)):
            if not np.isnan(dest[i]) and i not in self.path_history:
                viability = pheromones[i] * (KAPPA / dest[i])**BETA
                if not current_best_viability or viability > current_best_viability:
                    current_best_viability = viability
                    current_best_dest = i
        return current_best_dest

    def biased_exploration(self, dest: list, pheromones: list) -> int:
        probabilities = []
        denominator = 0
        
        #Calculate the denominator first
        for i in range(len(dest)):
            if not np.isnan(dest[i]):
                denominator += pheromones[i]**ALPHA * (KAPPA/dest[i])**BETA

        #Calculate probabilities of picking one path
        for node, length in enumerate(dest):
            if node in self.path_history:
                probabilities.append(0)
            elif not np.isnan(dest[node]):
                nominator = pheromones[node]**ALPHA * (KAPPA/dest[node])**BETA
                probabilities.append(nominator / denominator)
            else:
                probabilities.append(np.nan)
        
        #If there is no path available return false
        if np.nansum(probabilities) == 0:
            return None

        #Roulette wheel
        cumulative_sum = []
        for i in range(len(probabilities)):
            if np.isnan(probabilities[i]):
                cumulative_sum.append(np.nan)
            elif np.nansum(cumulative_sum) == 0:
                cumulative_sum.append(1.0)
            else:
                cumulative_sum.append(np.nansum(probabilities[i:len(probabilities)]))
        
        #Pick a destination
        rand = random.uniform(0, 1)
        dest_picked = None
        cumulative_sum.append(0)

        for i in range(0, len(cumulative_sum)-1):
            p = cumulative_sum[i]
            nextp = cumulative_sum[i+1] if not np.isnan(cumulative_sum[i+1]) else cumulative_sum[i+2] if not np.isnan(cumulative_sum[i+2]) else cumulative_sum[i+3]
        
            if not np.isnan(p) and rand <= p and rand >= nextp:
                dest_picked = i
        return dest_picked

    def update_pheromones_locally(self, pheromones: list, dest: int) -> None:
        #Apply the ACS local updating rule
        #evaporate
        new_value = pheromones[self.current_pos][dest] * (1 - RHO)
        pheromones[self.current_pos][dest] = max(new_value, self.initial_tau)
        pheromones[dest][self.current_pos] = max(new_value, self.initial_tau)
        
        pheromones[self.current_pos][dest] += RHO * self.initial_tau
        pheromones[dest][self.current_pos] += RHO * self.initial_tau
    
    def get_real_node_index(self, index):
        return self.real_nodes_indexes[index]

    def get_real_path(self, path):
        real_path = []
        for n in path:
            real_path.append(self.get_real_node_index(n))
        return real_path

class ACO():
    def __init__(self, graph, real_nodes_indexes, start: int) -> None:
        self.graph = graph
        self.real_nodes_indexes = real_nodes_indexes
        self.start = start
        self.current_best_path = None
        self.current_best_length =  float('inf')
        algo_neighbOURS = ALGO_neightbours(graph, 0)
        length, path = algo_neighbOURS.run()
        self.initial_tau = (length*30)**-1
        #print('TAU:', self.initial_tau)
        self.pheromones = np.full(graph.shape, self.initial_tau)

    def run(self, iter: int) -> list:
        for i in range(iter):
            #print('Tour:', i)
            self.tour_construction(10)
            self.global_update_pheromones()
        real_best_path = self.get_real_path(self.current_best_path)
        #print('Best:', real_best_path)
        #print('Length:', self.current_best_length)
        return real_best_path, self.current_best_length

    def tour_construction(self, ant_amount: int) -> None:
        ants = [Ant(self.real_nodes_indexes, self.initial_tau, random.randint(0, len(self.real_nodes_indexes)-1)) for i in range(ant_amount)]
        while ants:
            for ant in ants:
                if ant.can_continue:
                    ant.move(self.graph[ant.current_pos], self.pheromones)
                else:
                    self.update_current_best(ant.path_history)
                    ants.remove(ant)

    def update_current_best(self, path):
        path_length = self.calc_total_distance(path)
        if path_length < self.current_best_length:
            #print('NEW BEST:', path)
            print('BEST LENGTH:', path_length)
            self.current_best_path = path
            self.current_best_length = path_length

    def global_update_pheromones(self) -> None:
        length = len(self.pheromones[0])
        #Evaporation
        for i in range(length):
            for j in range(length):
                new_value = self.pheromones[i][j] * (1 - RHO)
                self.pheromones[i][j] = max(new_value, self.initial_tau)

        #New pheromones
        for i in range(len(self.current_best_path)-1):
            current_node = self.current_best_path[i]
            next_node = self.current_best_path[i+1]
            self.pheromones[current_node][next_node] += RHO * (KAPPA / self.current_best_length)
            self.pheromones[next_node][current_node] += RHO * (KAPPA / self.current_best_length)

    def calc_total_distance(self, full_path: list) -> float:
        distance = 0
        for i in range(len(full_path)-1):
            cur_path = full_path[i]
            next_path = full_path[i+1]
            distance += self.graph[cur_path][next_path]
        return distance
    
    def get_real_node_index(self, index):
        return self.real_nodes_indexes[index]

    def get_real_path(self, path):
        real_path = []
        for n in path:
            real_path.append(self.get_real_node_index(n))
        return real_path

def get_dataset_graph(dataset):
    points = []
    points_indices = []
    with open(dataset+'.txt', 'r') as file:
        i = 0
        for line in file.readlines():
            x, y = line.split(',')
            x = float(x)
            y = float(y)
            points.append((float(x), float(y)))
            points_indices.append(i)
            i += 1
    return points_indices, generate_distance_matrix(points)

def oliver_30():
    oliver30 = get_dataset_graph('oliver30')
    alls_dists = []
    start = time.time()
    for i in range(1):
        print('iter:', i)
        aco = ACO(oliver30[1], oliver30[0], 0)
        path, distance = aco.run(900)
        alls_dists.append(distance)
    print('TOTAL TIME:', time.strftime('%H:%M:%S', time.gmtime(time.time()-start)))
    #vizu.add_path(path)
    print('MEAN:', mean(alls_dists))

oliver_30()