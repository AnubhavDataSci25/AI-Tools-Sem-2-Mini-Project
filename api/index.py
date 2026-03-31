import json
from flask import Flask, request, jsonify, render_template
from collections import deque
import heapq

# app = Flask(__name__)
app = Flask(__name__, template_folder="../templates")

# --- Graph Data Configuration ---
# Nodes with their heuristic values (estimated distance to goal 'G')
NODES = {
    'A': {'x': 50,  'y': 50,  'h': 10, 'label': 'Downtown'},
    'B': {'x': 250, 'y': 40,  'h': 8,  'label': 'North Suburb'},
    'C': {'x': 120, 'y': 180, 'h': 7,  'label': 'Central Park'},
    'D': {'x': 450, 'y': 60,  'h': 5,  'label': 'Tech Park'},
    'E': {'x': 300, 'y': 220, 'h': 4,  'label': 'Industrial East'},
    'F': {'x': 80,  'y': 350, 'h': 6,  'label': 'West Gate'},
    'G': {'x': 480, 'y': 340, 'h': 0,  'label': 'City Hospital'},  # Goal
    'H': {'x': 280, 'y': 360, 'h': 3,  'label': 'South Side'},
    'I': {'x': 100, 'y': 20,  'h': 12, 'label': 'North High'},
    'J': {'x': 400, 'y': 180, 'h': 4,  'label': 'East Market'},
    'K': {'x': 30,  'y': 220, 'h': 9,  'label': 'West Port'},
    'L': {'x': 200, 'y': 320, 'h': 5,  'label': 'University'},
    'M': {'x': 540, 'y': 200, 'h': 3,  'label': 'Airport Road'},
    'N': {'x': 380, 'y': 310, 'h': 2,  'label': 'Residential South'}
}

# Edges representing roads and their costs
EDGES = [
    ('A', 'B', 4), ('A', 'C', 2), ('B', 'D', 5), ('B', 'C', 1),
    ('C', 'E', 8), ('C', 'F', 10), ('D', 'G', 6), ('E', 'G', 2),
    ('F', 'H', 3), ('H', 'G', 5), ('D', 'E', 2),
    ('I', 'A', 3), ('I', 'B', 7), ('A', 'K', 5), ('K', 'F', 6),
    ('C', 'L', 6), ('L', 'H', 4), ('J', 'D', 4), ('J', 'E', 3),
    ('E', 'M', 6), ('M', 'G', 4), ('H', 'N', 4), ('N', 'G', 3)
]

# Build adjacency list for bidirectional travel
def get_adj_list():
    adj = {node: [] for node in NODES}
    for start, end, cost in EDGES:
        adj[start].append({'node': end, 'cost': cost})
        adj[end].append({'node': start, 'cost': cost})
    return adj

ADJ = get_adj_list()

# --- Search Algorithms ---

def get_path_cost(path):
    cost = 0
    for i in range(len(path) - 1):
        for neighbor in ADJ[path[i]]:
            if neighbor['node'] == path[i+1]:
                cost += neighbor['cost']
                break
    return cost

def run_bfs(start, goal):
    queue = deque([[start]])
    visited = {start}
    traversal_order = []

    while queue:
        path = queue.popleft()
        node = path[-1]
        traversal_order.append(node)

        if node == goal:
            return path, traversal_order, get_path_cost(path)

        for neighbor in ADJ[node]:
            if neighbor['node'] not in visited:
                visited.add(neighbor['node'])
                queue.append(path + [neighbor['node']])
    return None

def run_dfs(start, goal):
    stack = [[start]]
    visited = set()
    traversal_order = []

    while stack:
        path = stack.pop()
        node = path[-1]

        if node not in visited:
            visited.add(node)
            traversal_order.append(node)

            if node == goal:
                return path, traversal_order, get_path_cost(path)

            for neighbor in reversed(ADJ[node]): # Reversed for consistent stack behavior
                if neighbor['node'] not in visited:
                    stack.append(path + [neighbor['node']])
    return None

def run_best_fs(start, goal):
    # Priority Queue stores (heuristic, path)
    pq = [(NODES[start]['h'], [start])]
    visited = set()
    traversal_order = []

    while pq:
        _, path = heapq.heappop(pq)
        node = path[-1]

        if node in visited: continue
        visited.add(node)
        traversal_order.append(node)

        if node == goal:
            return path, traversal_order, get_path_cost(path)

        for neighbor in ADJ[node]:
            if neighbor['node'] not in visited:
                h = NODES[neighbor['node']]['h']
                heapq.heappush(pq, (h, path + [neighbor['node']]))
    return None

def run_a_star(start, goal):
    # Priority Queue stores (f_score, g_score, path)
    # f(n) = g(n) + h(n)
    start_h = NODES[start]['h']
    pq = [(start_h, 0, [start])]
    visited = {} # Stores best g_score for each node
    traversal_order = []

    while pq:
        f, g, path = heapq.heappop(pq)
        node = path[-1]

        if node in visited and visited[node] <= g: continue
        visited[node] = g
        traversal_order.append(node)

        if node == goal:
            return path, traversal_order, g

        for neighbor in ADJ[node]:
            new_g = g + neighbor['cost']
            if neighbor['node'] not in visited or new_g < visited[neighbor['node']]:
                new_f = new_g + NODES[neighbor['node']]['h']
                heapq.heappush(pq, (new_f, new_g, path + [neighbor['node']]))
    return None

# --- Flask Endpoints ---

@app.route('/')
def index():
    return render_template(
        'index.html',
        nodes_json=json.dumps(NODES),
        edges_json=json.dumps(EDGES)
    )

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    start = data.get('start')
    goal = data.get('goal')
    algo = data.get('algorithm')

    if start not in NODES or goal not in NODES:
        return jsonify({'error': 'Invalid nodes'}), 400

    result = None
    if algo == 'BFS':
        result = run_bfs(start, goal)
    elif algo == 'DFS':
        result = run_dfs(start, goal)
    elif algo == 'BestFS':
        result = run_best_fs(start, goal)
    elif algo == 'AStar':
        result = run_a_star(start, goal)

    if result:
        path, traversal, cost = result
        return jsonify({
            'path': path,
            'traversal': traversal,
            'cost': cost
        })
    
    return jsonify({'error': 'Path not found'}), 404

# if __name__ == '__main__':
#     # Default Flask behavior: listens on http://127.0.0.1:5000
#     print("🚑 Ambulance AI Router starting...")
#     app.run(debug=True, port=5000)
app = app