import pygame
import numpy as np
import random
from pygame import gfxdraw

SCREEN_WIDTH = 1300
SCREEN_HEIGHT = 900

CENTROIDS_COLOR = (215, 117, 250)
DEFAULT_NODES_COLOR = (160, 153, 255)
DEFAULT_PATHS_COLOR = (255, 15, 71)
DEPOT_COLOR = (54, 227, 106)
class VRPVisualizator():
    def __init__(self) -> None:
        self.init_pygame()
        self.nodes = list()
        self.centroids = list()
        self.clusters_colors = list()
        self.centroid_binded = dict()
        self.centroid_lines = list()
        self.pheromones = [[]]
        self.paths = []
        self.current_path = []
        self.max_pheromone = 1
        self.min_pheromone = 0
        self.scaling = 1
        self.font = pygame.font.SysFont(None, 24)

    def init_pygame(self):
        """
            Init pygame
        """
        pygame.init()

        pygame.display.set_caption("VRP Visualization")
        self.display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    def draw(self):
        """
            Draw
        """
        events = pygame.event.get()
        #Draw game
        self.display.fill((255, 255, 255))
         #self.draw_pheromones()
        self.draw_paths()
        self.draw_centroid_lines()
        self.draw_nodes()
        self.draw_centroids()
        pygame.display.flip()

        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()

    def draw_paths(self):
        for path in self.paths:
            for i in range(len(path) - 1):
                start_node = self.nodes[path[i]]
                end_node = self.nodes[path[i + 1]]
                start_node_index = self.nodes.index(start_node)
                node_index = start_node_index if start_node_index != 0 else self.nodes.index(end_node)
                if node_index - 1 in self.centroid_binded:
                    color = self.clusters_colors[self.centroid_binded[node_index - 1]]
                else:
                    color = DEFAULT_PATHS_COLOR
                start_coordinates = self.get_coordinates(start_node[0], start_node[1])
                end_coordinates = self.get_coordinates(end_node[0], end_node[1])
                pygame.draw.aaline(self.display, color, start_coordinates, end_coordinates, int(3*self.scaling))
    
    def draw_nodes(self):
        for i, node in enumerate(self.nodes):
            coordinates = self.get_coordinates(node[0], node[1])
            if i == 0:
                gfxdraw.aacircle(self.display, *coordinates, int(4*self.scaling), DEPOT_COLOR)
                gfxdraw.filled_circle(self.display, *coordinates, int(4*self.scaling), DEPOT_COLOR)
            else:
                if not i - 1 in self.centroid_binded:
                    color = DEFAULT_NODES_COLOR
                else:
                    color = self.clusters_colors[self.centroid_binded[i - 1]]
                gfxdraw.aacircle(self.display, *coordinates, int(4*self.scaling), color)
                gfxdraw.filled_circle(self.display, *coordinates, int(4*self.scaling), color)
            # img = self.font.render(str(i), True, (255, 0, 200))
            # coordinates = (coordinates[0] - self.scaling*5, coordinates[1] - self.scaling*5)
            # self.display.blit(img, coordinates)

    def draw_centroids(self):
        for centroid in self.centroids:
            coordinates = self.get_coordinates(centroid[0], centroid[1])
            gfxdraw.aacircle(self.display, *coordinates, int(5*self.scaling), CENTROIDS_COLOR)
            gfxdraw.filled_circle(self.display, *coordinates, int(5*self.scaling), CENTROIDS_COLOR)

    def draw_centroid_lines(self):
        for node, centroid in self.centroid_lines:
            centroid_index = self.centroids.index(centroid)
            node_index = self.nodes.index(node) - 1
            color = self.clusters_colors[centroid_index]
            start_coordinates = self.get_coordinates(node[0], node[1])
            end_coordinates = self.get_coordinates(centroid[0], centroid[1])
            pygame.draw.aaline(self.display, color, start_coordinates, end_coordinates, int(3*self.scaling))

    def add_centroid_line(self, node, centroid):
        self.centroid_lines.append((node, centroid))
        self.draw()

    def clear_centroid_lines(self):
        self.centroid_lines.clear()
        self.draw()

    def add_centroid(self, centroid):
        color = tuple(np.random.random(size=3) * 256)
        #color = cmapy.color('viridis', random.randrange(0, 256, 10), rgb_order=True)
        self.centroids.append(centroid)
        self.clusters_colors.append(color)

    def remove_centroid(self, centroid):
        index = self.centroids.index(centroid)
        self.centroids.pop(index)
        self.clusters_colors.pop(index)
        self.draw()

    def clear_centroids_and_color(self):
        self.centroids = list()
        self.clusters_colors = list()
        self.draw()

    def clear_centroids(self):
        self.centroids = list()
        self.draw()

    def clear_current_path(self):
        self.paths.remove(self.current_path)
        self.draw()

    def clear_paths(self):
        self.paths = []
        self.draw()

    def bind_to_centroid(self, node_i, centroid_i):
        self.centroid_binded[node_i] = centroid_i

    def set_path(self, path):
        if self.current_path in self.paths:
            self.paths.remove(self.current_path)
        self.paths.append(path)
        self.current_path = path
        self.draw()

    def add_path(self, path):
        self.paths.append(path)
        self.draw()

    def draw_pheromones(self):
        for i in range(len(self.pheromones[0])):
            for j in range(i, len(self.pheromones[0])):
                if i != j:
                    start_coordinates = self.get_coordinates(self.nodes[i][0], self.nodes[i][1])
                    end_coordinates = self.get_coordinates(self.nodes[j][0], self.nodes[j][1])
                    midpoint_x = (start_coordinates[0] / 2 + end_coordinates[0] / 2)
                    midpoint_y = (start_coordinates[1] / 2 + end_coordinates[1] / 2)
                    img = self.font.render(str(round(self.pheromones[i][j], 2)), True, (53, 217, 219))
                    self.display.blit(img, (midpoint_x, midpoint_y))
                # try:
                #     if self.max_pheromone - self.min_pheromone == 0:
                #         opacity = self.min_pheromone
                #     else:
                #         opacity = (self.pheromones[i][j] - self.min_pheromone) / (self.max_pheromone - self.min_pheromone)
                #     start_coordinates = self.get_coordinates(self.nodes[i][0], self.nodes[i][1])
                #     end_coordinates = self.get_coordinates(self.nodes[j][0], self.nodes[j][1])
                #     r = opacity * 120 + (1 - opacity) * 255
                #     g = opacity * 210 + (1 - opacity) * 255
                #     b = opacity * 240 + (1 - opacity) * 255
                #     if opacity > 0.6:
                #         pygame.draw.aaline(self.display, (r, g, b), start_coordinates, end_coordinates, int(3*self.scaling))
                # except Exception:
                #     print('CURRENT:', self.pheromones[i][j])
                #     print('MIN:', self.min_pheromone)
                #     print('MAX:', self.max_pheromone)
                #     print('OPACITY:', opacity)

    def get_coordinates(self, x, y):
        return (int(x + SCREEN_WIDTH/2), int(y + SCREEN_HEIGHT/2))

    def set_scaling(self, scaling):
        self.scaling = scaling
        self.font = pygame.font.SysFont(None, int(24*scaling))
        self.draw()

    def set_pheromones(self, pheromones):
        self.pheromones = pheromones
        self.max_pheromone = np.nanmax(self.pheromones)
        self.min_pheromone = np.nanmin(self.pheromones)
        self.draw()

    def add_node(self, x, y):
        self.nodes.append((x, y))
        self.draw()

    def clear_nodes(self):
        self.nodes = []
        self.draw()
