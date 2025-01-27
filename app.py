import os
import time
import psutil
from flask import Flask, render_template, request, jsonify, send_file
import folium
from folium.plugins import MarkerCluster
import requests
import polyline
import random
import itertools
import numpy as np
import math
import networkx as nx
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

class TSP:
    def __init__(self):
        self.distance_matrix = np.zeros((0, 0))
        self.routes_matrix = []
        self.map = folium.Map(location=[32.0853, 34.85], zoom_start=8)
        self.marker_cluster = MarkerCluster().add_to(self.map)
        self.nodes = []

    def geocode_address(self, address):
        nominatim_url = "https://nominatim.openstreetmap.org/search"
        headers = {
            'User-Agent': 'MyTSP/1.0 (davidor6@gmail.com)',
        }
        params = {
            'q': address,
            'format': 'json',
            'addressdetails': 1,
            'limit': 1
        }
        try:
            response = requests.get(nominatim_url, headers=headers, params=params)
            response.raise_for_status()
            results = response.json()
            if results:
                lat = float(results[0]['lat'])
                lon = float(results[0]['lon'])
                return lat, lon
        except (requests.RequestException, ValueError):
            return None

    def get_route_osrm(self, origin, destination):
        osrm_url = "http://router.project-osrm.org/route/v1/driving/"
        origin_str = f"{origin[1]},{origin[0]}"
        destination_str = f"{destination[1]},{destination[0]}"
        try:
            response = requests.get(f"{osrm_url}{origin_str};{destination_str}?overview=full")
            response.raise_for_status()
            data = response.json()
            if 'routes' in data and len(data['routes']) > 0:
                polyline_str = data['routes'][0]['geometry']
                route_coords = polyline.decode(polyline_str)
                distance = data['routes'][0]['distance'] / 1000  # Convert to kilometers
                return route_coords, distance
        except (requests.RequestException, ValueError):
            return [], 0

    def add_to_matrices(self):
        distances = []
        routes = []
        for i in range(len(self.nodes)):
            data = self.get_route_osrm(self.nodes[-1][:2], self.nodes[i][:2])
            distances.append(data[1])
            routes.append(data[0])
        distances = np.array(distances)

        self.routes_matrix.append(routes[:-1])
        for idx, route in enumerate(routes):
            if idx < len(self.routes_matrix):
                self.routes_matrix[idx].append(route)
            else:
                self.routes_matrix.append([route])
        self.distance_matrix = np.append(self.distance_matrix, [distances[:-1]], axis=0)
        self.distance_matrix = np.append(self.distance_matrix, distances.reshape(-1, 1), axis=1)

    def tsp_bruteforce(self):
        n = len(self.distance_matrix)
        permutations = itertools.permutations(range(n))
        min_cost = float('inf')
        min_path = None
        for perm in permutations:
            cost = sum(self.distance_matrix[perm[i]][perm[i + 1]] for i in range(n - 1))
            cost += self.distance_matrix[perm[-1]][perm[0]]
            if cost < min_cost:
                min_cost = cost
                min_path = perm

        return min_path, min_cost

    def nearest_neighbor(self):
        n = len(self.distance_matrix)
        visited = [False] * n
        path = []
        current_city = random.randint(0, n - 1)
        path.append(current_city)
        visited[current_city] = True
        total_cost = 0

        for _ in range(n - 1):
            last_node = path[-1]
            nearest_node = None
            min_distance = float('inf')

            for i in range(n):
                if not visited[i] and self.distance_matrix[last_node][i] < min_distance:
                    nearest_node = i
                    min_distance = self.distance_matrix[last_node][i]

            path.append(nearest_node)
            visited[nearest_node] = True
            total_cost += min_distance

        total_cost += self.distance_matrix[path[-1]][path[0]]
        return path, total_cost

    def tsp_genetic_algorithm(self, population_size=100, generations=500, mutation_rate=0.1):
        def fitness(path):
            return sum(self.distance_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1)) + self.distance_matrix[path[-1]][path[0]]

        def mutate(path):
            if random.random() < mutation_rate:
                i, j = random.sample(range(len(path)), 2)
                path[i], path[j] = path[j], path[i]

        def crossover(parent1, parent2):
            size = len(parent1)
            start, end = sorted(random.sample(range(size), 2))
            child = [-1] * size
            child[start:end] = parent1[start:end]

            pointer = 0
            for gene in parent2:
                if gene not in child:
                    while child[pointer] != -1:
                        pointer += 1
                    child[pointer] = gene
            return child

        n = len(self.distance_matrix)
        population = [random.sample(range(n), n) for _ in range(population_size)]

        for _ in range(generations):
            population = sorted(population, key=fitness)
            next_generation = population[:population_size // 2]

            while len(next_generation) < population_size:
                parents = random.sample(next_generation, 2)
                child = crossover(parents[0], parents[1])
                mutate(child)
                next_generation.append(child)

            population = next_generation

        best_path = min(population, key=fitness)
        best_cost = fitness(best_path)
        return best_path, best_cost

    def greedy_tsp(self):
        n = len(self.distance_matrix)
        visited = [False] * n
        tour = []
        total_distance = 0

        current_city = random.randint(0, n - 1)
        tour.append(current_city)
        visited[current_city] = True

        for _ in range(n - 1):
            nearest_city = None
            shortest_distance = float('inf')

            for next_city in range(n):
                if not visited[next_city] and 0 < self.distance_matrix[current_city][next_city] < shortest_distance:
                    shortest_distance = self.distance_matrix[current_city][next_city]
                    nearest_city = next_city

            if nearest_city is None:
                raise ValueError("Disconnected graph: No unvisited city is reachable.")

            visited[nearest_city] = True
            tour.append(nearest_city)
            total_distance += shortest_distance
            current_city = nearest_city

        total_distance += self.distance_matrix[current_city][tour[0]]
        return tour, total_distance


    def calculate_total_distance(self, solution):
        total_distance = 0
        for i in range(len(solution) - 1):
            total_distance += self.distance_matrix[solution[i]][solution[i + 1]]
        total_distance += self.distance_matrix[solution[-1]][solution[0]]  # Return to start
        return total_distance

    def swap_two_cities(self, solution):
        new_solution = solution[:]
        i, j = random.sample(range(len(solution)), 2)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        return new_solution

    def accept_new_solution(self, old_cost, new_cost, temp):
        if new_cost < old_cost:
            return True
        else:
            # Accept with a probability depending on the temperature
            probability = math.exp((old_cost - new_cost) / temp)
            return random.random() < probability

    def simulated_annealing(self, initial_temp=1000, cooling_rate=0.995, min_temp=1e-8, max_iterations=10):
        n = len(self.distance_matrix)
        current_solution = list(range(n))  # Initial solution: 0, 1, 2, ..., n-1
        random.shuffle(current_solution)  # Shuffle to get a random starting solution
        best_solution = current_solution[:]
        best_cost = self.calculate_total_distance(best_solution)

        temp = initial_temp
        while temp > min_temp:
            for _ in range(max_iterations):
                # Generate a neighboring solution by swapping two cities
                new_solution = self.swap_two_cities(current_solution)
                new_cost = self.calculate_total_distance(new_solution)

                # Decide whether to accept the new solution
                if self.accept_new_solution(self.calculate_total_distance(current_solution), new_cost, temp):
                    current_solution = new_solution

                # Update best solution found so far
                if new_cost < best_cost:
                    best_solution = new_solution
                    best_cost = new_cost

            # Lower the temperature
            temp *= cooling_rate

        return best_solution, best_cost

    def create_graph(self):
        # Create a graph from the distance matrix
        n = len(self.distance_matrix)
        G = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j, weight=self.distance_matrix[i][j])
        return G

    def find_odd_degree_vertices(self, MST):
        # Find the vertices with odd degree in the MST
        odd_vertices = [v for v in MST.nodes if MST.degree(v) % 2 == 1]
        return odd_vertices

    def minimum_weight_matching(self, odd_vertices):
        n = len(odd_vertices)
        cost_matrix = np.zeros((n, n))

        # Create the cost matrix for the odd-degree vertices
        for i in range(n):
            for j in range(i + 1, n):
                u = odd_vertices[i]
                v = odd_vertices[j]
                cost_matrix[i, j] = self.distance_matrix[u][v]
                cost_matrix[j, i] = self.distance_matrix[u][v]

        # Solve the assignment problem (minimum weight matching)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Return the matching edges
        matching_edges = [(odd_vertices[row], odd_vertices[col]) for row, col in zip(row_ind, col_ind)]
        return matching_edges

    def combine_mst_matching(self, MST, matching_edges):
        # Add the matching edges to the MST to make it a multigraph
        for u, v in matching_edges:
            MST.add_edge(u, v)
        return MST

    def find_eulerian_circuit(self, MST):
        # Find an Eulerian circuit using Hierholzer's algorithm
        circuit = []
        stack = [list(MST.nodes())[0]]
        current_path = []

        while stack:
            u = stack[-1]
            if MST.degree(u) > 0:
                v = list(MST.neighbors(u))[0]  # Take any neighbor
                stack.append(v)
                MST.remove_edge(u, v)
            else:
                circuit.append(u)
                stack.pop()

        return circuit

    def create_tsp_path(self, eulerian_circuit):
        # Remove repeated vertices to create a valid TSP path
        visited = set()
        tsp_path = []
        for v in eulerian_circuit:
            if v not in visited:
                tsp_path.append(v)
                visited.add(v)
        return tsp_path

    def christofides_algorithm(self):
        # Create the graph from the distance matrix
        G = self.create_graph()

        # Step 1: Find the minimum spanning tree
        mst = nx.minimum_spanning_tree(G)

        # Step 2: Find odd degree vertices in the MST
        odd_vertices = self.find_odd_degree_vertices(mst)

        # Step 3: Find the minimum weight perfect matching for the odd-degree vertices
        matching_edges = self.minimum_weight_matching(odd_vertices)

        # Step 4: Combine the MST and the matching to form a multigraph
        mst_with_matching = self.combine_mst_matching(mst, matching_edges)

        # Step 5: Find the Eulerian circuit
        eulerian_circuit = self.find_eulerian_circuit(mst_with_matching)

        # Step 6: Create the TSP path
        tsp_path = self.create_tsp_path(eulerian_circuit)

        # Calculate the total cost of the tour
        total_cost = sum(self.distance_matrix[tsp_path[i]][tsp_path[i + 1]] for i in range(len(tsp_path) - 1))
        total_cost += self.distance_matrix[tsp_path[-1]][tsp_path[0]]  # Return to the start

        return tsp_path, total_cost

    def two_opt_swap(self, tour, i, k):
        # Perform a 2-opt swap: reverse the segment between indices i and k
        new_tour = tour[:i] + tour[i:k + 1][::-1] + tour[k + 1:]
        return new_tour

    # Function to implement the Ant Colony Optimization (ACO) algorithm
    def ant_colony_optimization(self, num_ants=10, num_iterations=10, alpha=1, beta=2, rho=0.5, q0=0.9):
        n = len(self.distance_matrix)

        # Initialize pheromone matrix with small positive values
        pheromone_matrix = np.ones((n, n)) * 0.1
        best_tour = None
        best_cost = float('inf')

        # Main ACO loop
        for _ in range(num_iterations):
            all_tours = []
            all_costs = []

            # Each ant constructs a tour
            for ant in range(num_ants):
                tour = []
                visited = [False] * n
                current_city = random.randint(0, n - 1)
                tour.append(current_city)
                visited[current_city] = True

                for _ in range(n - 1):
                    # Select next city using probabilistic rule
                    probabilities = []
                    for next_city in range(n):
                        if not visited[next_city]:
                            pheromone = pheromone_matrix[current_city][next_city] ** alpha
                            distance = (1 / self.distance_matrix[current_city][next_city]) ** beta
                            probabilities.append(pheromone * distance)
                        else:
                            probabilities.append(0)

                    # Normalize probabilities
                    total_prob = sum(probabilities)
                    probabilities = [p / total_prob for p in probabilities]

                    # Choose next city based on probabilities
                    if random.random() < q0:
                        next_city = np.argmax(
                            probabilities)  # Exploitation: choose the city with the highest probability
                    else:
                        next_city = np.random.choice(range(n),
                                                     p=probabilities)  # Exploration: choose based on probability

                    # Update the tour
                    tour.append(next_city)
                    visited[next_city] = True
                    current_city = next_city

                # Calculate the cost of the tour
                cost = self.calculate_total_distance(tour)
                all_tours.append(tour)
                all_costs.append(cost)

                # Update the best solution
                if cost < best_cost:
                    best_tour = tour
                    best_cost = cost

            # Update pheromones: Evaporation + Deposition
            pheromone_matrix *= (1 - rho)  # Evaporation
            for ant in range(num_ants):
                for i in range(n - 1):
                    pheromone_matrix[all_tours[ant][i]][all_tours[ant][i + 1]] += 1 / all_costs[ant]
                pheromone_matrix[all_tours[ant][-1]][all_tours[ant][0]] += 1 / all_costs[ant]  # Return to the start

        # Convert the tour to a list of regular integers
        best_tour = [int(city) for city in best_tour]

        return best_tour, best_cost  # Return the best tour and its total cost

tsp = TSP()

@app.route("/")
def home():
    tsp.map.save("static/map.html")
    return render_template("index.html")


@app.route("/add_node", methods=["POST"])
def add_node():
    address = request.form.get("address")
    location = tsp.geocode_address(address)

    if location:
        tsp.nodes.append(location + (address,))
        tsp.add_to_matrices()
        folium.Marker(location, popup=f"Node {len(tsp.nodes)}: {address}").add_to(tsp.marker_cluster)
        tsp.map.save("static/map.html")
        return jsonify({"success": True, "message": f"Node added: {address}"})
    return jsonify({"success": False, "message": "Failed to geocode address."})


@app.route("/calculate_route", methods=["POST"])
def calculate_route():
    algorithm = request.form.get("algorithm")

    if len(tsp.nodes) < 2:
        return jsonify({"success": False, "message": "At least two nodes are required."})

    algorithm_map = {
        "nearest": tsp.nearest_neighbor,
        "bruteforce": tsp.tsp_bruteforce,
        "genetic": tsp.tsp_genetic_algorithm,
        "greedy": tsp.greedy_tsp,
        "heldkarp": tsp.held_karp,
        "annealing": tsp.simulated_annealing,
        "christofides": tsp.christofides_algorithm,
        "ant_colony": tsp.ant_colony_optimization
    }

    if algorithm not in algorithm_map:
        return jsonify({"success": False, "message": "Unknown algorithm."})

    path, total_cost = algorithm_map[algorithm]()

    route_description = []
    for i in range(len(path) - 1):
        origin = tsp.nodes[path[i]]
        destination = tsp.nodes[path[i + 1]]
        route_coords = tsp.routes_matrix[path[i]][path[i + 1]]
        folium.PolyLine(route_coords, color="red", weight=2.5, opacity=1).add_to(tsp.map)
        route_description.append(f"From {origin[2]} go to {destination[2]}")

    origin = tsp.nodes[path[-1]]
    destination = tsp.nodes[path[0]]
    route_coords = tsp.routes_matrix[path[-1]][path[0]]
    folium.PolyLine(route_coords, color="red", weight=2.5, opacity=1).add_to(tsp.map)
    route_description.append(f"From {origin[2]} go to {destination[2]}")

    tsp.map.save("static/map.html")

    route_description_str = " -> ".join(route_description)
    #run_all_algorithms()
    return jsonify({"success": True, "message": f"Route calculated: {route_description_str}", "total_cost": total_cost})


@app.route("/run_all_algorithms", methods=["POST"])
def run_all_algorithms():
    # List of algorithms to test
    algorithms = [
        "nearest",
        "genetic",
        "greedy",
        "heldkarp",
        "annealing",
        "christofides",
        "ant_colony",
        "bruteforce"
    ]

    # Dictionary to store results
    results = []
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss

    for algorithm in algorithms:
        # Measure execution time
        start_time = time.perf_counter()
        before_memory = get_memory_usage()
        if algorithm == "nearest":
            path, total_cost = tsp.nearest_neighbor()
        elif algorithm == "bruteforce" and len(tsp.nodes) <= 10:
            path, total_cost = tsp.tsp_bruteforce()
        elif algorithm == "genetic":
            path, total_cost = tsp.tsp_genetic_algorithm()
        elif algorithm == "greedy":
            path, total_cost = tsp.greedy_tsp()
        elif algorithm == "heldkarp" and len(tsp.nodes) <= 15:
            path, total_cost = tsp.held_karp()
        elif algorithm == "annealing":
            path, total_cost = tsp.simulated_annealing()
        elif algorithm == "christofides":
            path, total_cost = tsp.christofides_algorithm()
        elif algorithm == "ant_colony":
            path, total_cost = tsp.ant_colony_optimization()
        else:
            continue
        end_time = time.perf_counter()
        after_memory = get_memory_usage()
        memory_used = (after_memory - before_memory)
        # Calculate execution time
        execution_time = end_time - start_time
        print(f"Memory used by {algorithm}: {memory_used} bytes")

        # Append results
        results.append({
            "algorithm": algorithm,
            "path": path,
            "cost": float(total_cost),
            "execution_time": execution_time,
            "memory used": memory_used
        })

    execution_times = [item['execution_time'] for item in results]
    costs = [item['cost'] for item in results]
    algorithms = [item['algorithm'] for item in results]

    colors = plt.cm.tab10(range(len(results)))

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    for i, algorithm in enumerate(algorithms):
        plt.scatter(
            execution_times[i], costs[i], color=colors[i], label=algorithm, alpha=0.8
        )

    # Configure axes
    plt.xscale("log")  # Logarithmic scale for execution time
    plt.xlabel("Execution Time (seconds, log scale)")
    plt.ylabel("Cost")
    plt.title("Cost vs. Execution Time for TSP Algorithms")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Add legend
    plt.legend(title="Algorithms", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Show plot
    plt.tight_layout()
    plt.show()

    return results

if __name__ == "__main__":
    app.run(debug=True)

