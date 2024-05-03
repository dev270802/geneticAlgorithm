import streamlit as st
import csv
import math
import random
from heapq import heappop, heappush
import time
import os
import requests
import json


page_element = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://img.freepik.com/free-vector/winter-blue-pink-gradient-background-vector_53876-117276.jpg?size=626&ext=jpg&ga=GA1.1.1224184972.1714262400&semt=ais");
background-size: cover;
}
[data-testid="stHeader"]{
background-color: rgba(0,0,0,0);
}
[data-testid="stToolbar"]{
rat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = radius * c

    return distance

def genetic_algorithm(nodes, start_node):
    # Implement your genetic algorithm here
    # Placeholder implementation
    tour = random.sample(nodes, len(nodes))
    tour_length = sum([1 for _ in tour])
    return tour, tour_lengthight: 2rem;
background-image: url("");
background-size: cover;
}
[data-testid="stSidebar"]> div:first-child{
background-image: url("https://img.freepik.com/premium-vector/skyblue-gradient-background-advertisers-gradient-hq-wallpaper_189959-513.jpg");
background-size: cover;
}
</style>
"""

st.markdown(page_element, unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black';>Route Optimizaton App 🛣️</h1>", unsafe_allow_html=True)
st.markdown("---")




def main():
    def gaAlgorithm(visited_nodes,startNode,endNode,starter):
        class Node:
            def __init__(self, name, x, y,h,index,waste_level,traffic,road_quality):
                self.name = name
                self.x = x
                self.y = y
                self.h=h
                self.index=index
                self.neighbors = []
                self.waste_level=waste_level
                self.traffic=traffic
                self.road_quality=road_quality

            def add_neighbor(self, neighbor, cost):
                self.neighbors.append((neighbor, cost))

            def __lt__(self, other):
                return False  # Dummy implementation for heap ordering

        def read_graph_from_csv(filename):
            graph = {}
            base_url ='https://smartbinpredict-dot-theta-totem-413418.el.r.appspot.com/forecast?smartbinId={}'
            start_node=startNode
            end_node=endNode
            indexToName={}
            index=0
            graph_names=[]
            unvisited_graph_names=[]
            with open(filename, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header row
                for row in reader:
                    name = row[0]
                    x = float(row[1])
                    y = float(row[2])
                    h = float(row[3])
                    id=float(row[10])

                    api_url = base_url.format(id)
                    print(api_url)
                    response = requests.get(api_url)

                    # Check if the request was successful (status code 200)
                    if response.status_code == 200:
                        # Parse the JSON response into a Python dictionary
                        # Parse the JSON response into a Python dictionary
                        data = json.loads(response.json())
                        # Access the first element of the array
                        waste_level = data["latest_fullness_forecast"][1]
                        print("First element of the array:", waste_level)

                    else:
                        # If the request was not successful, print the error status code
                        waste_level=float(row[6])
                        print("Api not called ", waste_level)
                    index=index;
                    indexToName[index]=name
                    traffic=float(row[5])
                    road_quality=float(row[4])
                    if starter==1 and row[0] not in visited_nodes and row[0]!=startNode and row[0]!=endNode:
                        unvisited_graph_names.append(row[0])
                    if starter==0 and ( row[9]!='start'and row[9]!='end'):
                        unvisited_graph_names.append(row[0])
                    if starter==0 and row[9] =='start':
                        start_node=row[0]
                    elif starter==0 and row[9] =='end':
                        end_node=row[0]

                    graph_names.append(name)
                    #if row[9] =='start':
                    #start_node=row[0]
                    #elif row[9] =='end':
                    #end_node=row[0]
                    graph[name] = Node(name, x, y,h, index,waste_level,traffic,road_quality)
                    index=index+1
            return graph,graph_names,indexToName,index,start_node,end_node,unvisited_graph_names

        def heuristic(node1,node2):
            # Implement your heuristic function here, which estimates the cost from the current node to the goal node.
            # You can use Euclidean distance, for example.
            lat1 = math.radians(node1.x)
            lon1 = math.radians(node1.y)
            lat2 = math.radians(graph[node2].x)
            lon2 = math.radians(graph[node2].y)

            # Radius of the Earth in kilometers
            radius = 6371

            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = radius * c

            return distance


        def build_graph_with_routes(graph, routes):
            for route in routes:
                for i in range(len(route) - 2):
                    node1_name = route[i]
                    node2_name = route[i + 1]

                    #if node1_name not in graph or node2_name not in graph:
                        #continue  # Skip if the node is not in the graph

                    node1 = graph[node1_name]
                    node2 = graph[node2_name]
                    cost = calculate_cost(node1, node2)
                    node1.add_neighbor(node2, cost)
                    node2.add_neighbor(node1, cost)


        def calculate_time(node1,node2):
            distance=calculate_cost(node1,node2)
            speed=80/(node2.traffic+node2.road_quality)
            return distance/speed


        def calculate_function(distance,node,h):
            speed=80/(node.traffic+node.road_quality)
            return distance/speed+h/40

        def calculate_Normalized_cost(distance,node,h):

            normalised_distance= 0 + ((distance - min_value) * (5 - 0) / (max_value - min_value))
            normalized_h= 0 + ((h - min_heur) * (5 - 0) / (max_heur - min_heur))
            normalised_heuristic=1*normalized_h+1.5*node.road_quality+1.5*node.traffic+3*node.waste_level
            return 5*normalised_distance+normalised_heuristic

        def calculate_cost(node1, node2):
        # Convert latitude and longitude from degrees to radians
            lat1 = math.radians(node1.x)
            lon1 = math.radians(node1.y)
            lat2 = math.radians(node2.x)
            lon2 = math.radians(node2.y)

            # Radius of the Earth in kilometers
            radius = 6371

            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = radius * c

            return distance

        def calculate_cost_normal(node1, node2):
        # Convert latitude and longitude from degrees to radians
            lat1 = math.radians(node1.x)
            lon1 = math.radians(node1.y)
            lat2 = math.radians(node2.x)
            lon2 = math.radians(node2.y)


            # Radius of the Earth in kilometers
            radius = 6371

            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = radius * c

            return calculate_Normalized_cost(distance,node1,heur[node1.index][node2.index])

        def calculate_route_distance(route,start_node):
            #print(type(route[0]))
            total_distance = 0
            #print(route)
            for i in range(len(route)):
                if(i<len(route)-1):
                    total_distance += arr[graph[route[i]].index][graph[route[i+1]].index]
            return total_distance

        def calculate_route_time(route,start_node):
            #print(type(route[0]))
            total_time = 0
            total_function=0
            #print(route)
            for i in range(len(route)):
                if(i<len(route)-1):
                    total_time += time_taken[graph[route[i]].index][graph[route[i+1]].index]
                    total_function+=total_time*graph[route[i+1]].waste_level
                    total_time+=0.033
            return total_function



        def a_star(start, goal):
            open_set = [(start.h, start)]  # Priority queue of nodes to explore
            came_from = {}  # Dictionary to store the previous node in the optimal path
            g_score = {node: float('inf') for node in graph.values()}  # Cost from start node to each node
            g_score[start] = 0
            f_score = {node: float('inf') for node in graph.values()}  # Estimated total cost from start to goal through each node
            f_score[start] = start.h

            while open_set:
                _, current = heappop(open_set)  # Get the node with the lowest f_score
                #print("inside pq",current.name)
                if current.name == goal:
                    # Reconstruct the path
                    path = [current.name]
                    #print(cost)
                    cost=0;
                    while current in came_from:
                        prev = came_from[current]
                        #print('prev',prev.name)
                        cost += calculate_cost(prev, current)
                        current = prev
                        path.append(current.name)
                    return path[::-1],cost

                for neighbor, cost in current.neighbors:
                    tentative_g_score = g_score[current] + cost
                    if tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = g_score[neighbor] + heuristic(neighbor,goal)
                        heappush(open_set, (f_score[neighbor], neighbor))

            return None  # No path found

        def a_star_normalized(start, goal):
            open_set = [(start.h, start)]  # Priority queue of nodes to explore
            came_from = {}  # Dictionary to store the previous node in the optimal path
            g_score = {node: float('inf') for node in graph.values()}  # Cost from start node to each node
            g_score[start] = 0
            f_score = {node: float('inf') for node in graph.values()}  # Estimated total cost from start to goal through each node
            f_score[start] = start.h

            while open_set:
                _, current = heappop(open_set)  # Get the node with the lowest f_score
                #print("inside pq",current.name)
                if current.name == goal:
                    # Reconstruct the path
                    path = [current.name]
                    #print(cost)
                    cost=0
                    timer=0
                    while current in came_from:
                        prev = came_from[current]
                        #print('prev',prev.name)
                        cost += calculate_cost(prev, current)
                        timer+=calculate_time(prev,current)
                        current = prev
                        path.append(current.name)
                    return path[::-1],cost,timer

                for neighbor, cost in current.neighbors:
                    h=heur[neighbor.index][graph[goal].index]
                    tentative_g_score = g_score[current] + calculate_function(cost,neighbor,h);
                    if tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = g_score[neighbor]
                        heappush(open_set, (f_score[neighbor], neighbor))
            return None  # No path found

        import random

        def crossover(parent1, parent2):
            start_node = parent1[0]
            end_node = parent2[-1]

            child1 = [start_node]
            child2 = [start_node]  # Start with the last node of parent2

            visited_child1 = set([start_node])
            visited_child2 = set([start_node])

            subset_size = random.randint(1, len(parent1) - 1)
            subset_indices = random.sample(range(1, len(parent1) - 1), subset_size-1)
            subset_nodes = [parent1[i] for i in subset_indices]

            for node in subset_nodes:
                if node != end_node:  # Check if not end_node
                    child1.append(node)
                    visited_child1.add(node)

            for node in parent2[1:]:  # Start from the second element
                if node not in visited_child1 and node != end_node:  # Check if not end_node
                    child1.append(node)
                    visited_child1.add(node)

            child1.append(end_node)  # Ensure the last element is from parent2 (only once)

            for node in parent1[1:]:  # Start from the second element
                if node not in visited_child2 and node != end_node:  # Check if not end_node
                    child2.append(node)
                    visited_child2.add(node)

            for node in subset_nodes:
                if node not in visited_child2 and node != end_node:  # Check if not end_node
                    child2.insert(0, node)  # Insert at the beginning
                    visited_child2.add(node)

            child2.append(end_node)  # Ensure the first element is from parent1

            return child1, child2





        def mutate(route):
            for i in range(1, len(route) - 2):
                if random.random() < mutation_rate:
                    j = random.randint(1, len(route) - 2)
                    route[i], route[j] = route[j], route[i]

        def roulette_wheel_selection(population, start_node):
            fitness_list = [1 / (calculate_route_time(route, start_node)+1) for route in population]
            total_fitness = sum(fitness_list)
            cumulative_fitness = [0]  # Initialize with 0
            for i in range(len(fitness_list)):
                fitness_list[i]=fitness_list[i]/total_fitness

            def pick_one(population, fitness):
                index = 0
                r = random.random()

                while r > 0:
                    r -= fitness[index]
                    index += 1

                index -= 1
                return index
            selected_parents = []
            prevIndex=-1
            for _ in range(2):  # Select two parents
                currIndex=pick_one(population,fitness_list)
                while currIndex==prevIndex:
                    currIndex=pick_one(population,fitness_list)
                prevIndex=currIndex
                random_number = random.uniform(0, total_fitness)
                selected_parents.append(population[prevIndex - 1])

            return selected_parents

        def shuffleIt(lst):
            #first_element = lst[0]
            middle_part = lst[1:-1]  # Exclude the first and last elements
            random.shuffle(middle_part)  # Shuffle the middle part in place
            lst[1:-1] = middle_part  # Update the shuffled middle part in the original list
            return lst

        # Genetic Algorithm Parameters
        population_size = 100
        generations = 1000
        mutation_rate = 0.1

        def genetic_algorithm(nodes, start_node):
            population = []
            for i in range(population_size):
                populationNew=[]
                populationNew=list(shuffleIt(nodes))
                #populationNew.append(start_node)
                population.append(populationNew)
            for generation in range(generations):
                population.sort(key=lambda route: calculate_route_time(route, start_node))
                best_route = population[0]

            new_population = []

            for i in range(3, population_size, 2):
                parent1, parent2 = roulette_wheel_selection(population, start_node)
                child1, child2 = crossover(parent1, parent2)
                mutate(child1)
                mutate(child2)
                new_population.append(child1)
                new_population.append(child2)

            # Update the old population with the new one
            population[3:] = new_population
            #print(population)


            population.sort(key=lambda route: calculate_route_time(route, start_node))
            best_route = population[0]

            best_distance = calculate_route_distance(best_route,start_node)
            return best_route, best_distance

        # Read graph data from VRP_dataset1.csv
        graph,graph_names,indexToName,size,start_node,end_node,unvisited_nodes = read_graph_from_csv('apiDynamicV1.csv')

        # Read routes data from Routes1.csv
        routes = []
        with open('Routes1.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                route = [node for node in row if node]  # Remove empty values from the route
                routes.append(route)

        # Build the graph with routes
        build_graph_with_routes(graph, routes)


        rows, cols = (size, size)
        heur= [ [0]*rows for i in range(cols)]
        for i in graph.values():
            for j in graph.values():
                if i.index>=j.index:
                    cost=heuristic(i,j.name)
                    heur[i.index][j.index]=cost
                    heur[j.index][i.index]=cost

        max_heur = max(max(row) for row in heur)

        min_heur = min(min(row) for row in heur)

        dummy = [ [0]*rows for i in range(cols)]
        for i in graph.values():
            for j in graph.values():
                if i.index>=j.index:
                    path,cost=a_star(i,j.name)
                    dummy[i.index][j.index]=cost
                    dummy[j.index][i.index]=cost
            #if i.index==size-1 and j.index==0:
                #print(path[::-1])


        #print("distance matrix without heuristic")
        #print(dummy)
        max_value = max(max(row) for row in dummy)

        min_value = min(min(row) for row in dummy)

        arr = [ [0]*rows for i in range(cols)]
        time_taken = [ [0]*rows for i in range(cols)]
        subPath = [ [0]*rows for i in range(cols)]

        for i in graph.values():
            for j in graph.values():
                if i.index>=j.index:
                    path,cost,time=a_star_normalized(i,j.name)
                    arr[i.index][j.index]=cost
                    time_taken[i.index][j.index]=time
                    subPath[i.index][j.index]=path
                    arr[j.index][i.index]=cost
                    time_taken[j.index][i.index]=time
                    subPath[j.index][i.index]=path[::-1]

        #print(arr)
        unvisited_nodes.insert(0,start_node);
        unvisited_nodes.append(end_node)
        #print(unvisited_nodes)
        #t1=time.perf_counter()
        tour, tour_length = genetic_algorithm(unvisited_nodes,start_node)
        #t2 = time.perf_counter()
        #print("")
        print("Tour with heuristic cost:")
        #print(tour)

        print("")
        #for i in range(len(tour)-1):
        #print("->".join(i for i in subPath[graph[tour[i]].index][graph[tour[i+1]].index][:-1]),end="->")
        #print(tour[len(tour)-1])
        return tour,tour_length
    
    vrp_csv_path = st.sidebar.file_uploader("Upload VRP Dataset")
    routes_csv_path = st.sidebar.file_uploader("Upload Routes Dataset")
        


    if "tour" not in st.session_state:
        st.session_state.tour = []
        st.session_state.tour_length = 0

    if vrp_csv_path and routes_csv_path:
        if st.button("Run Optimization"):
            st.session_state.tour, st.session_state.tour_length = gaAlgorithm([], '', '', 0)
            st.subheader("Optimized Tour:")
            st.write(st.session_state.tour)
            st.subheader("Tour Length:")
            st.write(st.session_state.tour_length)

        option = st.selectbox("Select New Start Node", st.session_state.tour, index=None)

        if option is not None:
            visitedNode = []
            for item in st.session_state.tour:
                visitedNode.append(item)
                if item == option:
                    break

            # Call optimization function with the selected node as the start
            st.session_state.tour, st.session_state.tour_length = gaAlgorithm(visitedNode, option, st.session_state.tour[-1], 1)
            st.subheader("Optimized Tour:")
            st.write(st.session_state.tour)
            st.subheader("Tour Length:")
            st.write(st.session_state.tour_length)



        
        
        # Interface components for user interaction
        
        
if __name__ == "__main__":
    main()
