import pickle
import sys
from collections import deque
import heapq
import time

class PriorityQueue:
    """A Queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first.
    If order is 'min', the item with minimum f(x) is
    returned first; if order is 'max', then it is the item with maximum f(x).
    Also supports dict-like lookup."""

    def __init__(self, order='min', f=lambda x: x):
        self.heap = []
        if order == 'min':
            self.f = f
        elif order == 'max':  # now item with max f(x)
            self.f = lambda x: -f(x)  # will be popped first
        else:
            raise ValueError("Order must be either 'min' or 'max'.")

    def append(self, item):
        """Insert item at its correct position."""
        heapq.heappush(self.heap, (self.f(item), item))

    def extend(self, items):
        """Insert each item in items at its correct position."""
        for item in items:
            self.append(item)

    def pop(self):
        """Pop and return the item (with min or max f(x) value)
        depending on the order."""
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception('Trying to pop from empty PriorityQueue.')

    def __len__(self):
        """Return current capacity of PriorityQueue."""
        return len(self.heap)

    def __contains__(self, key):
        """Return True if the key is in PriorityQueue."""
        return any([item == key for _, item in self.heap])

    def __getitem__(self, key):
        """Returns the first value associated with key in PriorityQueue.
        Raises KeyError if key is not present."""
        for value, item in self.heap:
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the priority queue")

    def __delitem__(self, key):
        """Delete the first occurrence of key."""
        try:
            del self.heap[[item == key for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
        heapq.heapify(self.heap)



class Node:

    """A node in a search tree. """

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1


    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def __lt__(self, node):
        return self.state < node.state




    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __repr__(self):
        return "<Node {}>".format(self.state)

class GridWorldState:

    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = GridWorldState.state_id
        GridWorldState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    def __eq__(self, other):
        return self.id == other.id or self.board == other.board

    def copy(self):
        return GridWorldState(self.board.copy(), self.id)

    def __repr__(self):
        return str(self.id) + " " + str(self.board)

    def __hash__(self):
        return hash(self.id)



def buildBoard(size, blocked, car, goal):

    grid = {}
    goals = {}
    cars = {}

    car_count = 1
    # check for input errors (valid board)
    for r in range(size):
        for c in range(size):
            if (r,c) in blocked:
                grid[(r,c)] = -1
            elif (r,c) in car:
                cars[car_count] = car[car_count-1]
                goals[car_count] = goal[car_count-1]
                grid[(r,c)] = car_count
                car_count += 1
            else:
                grid[(r,c)] = 0

    return Board(size, grid, goals, cars)

# auxiliary function to get ajacents of a position (valid or not)
def adjacents(position):
    return [(position[0]-1, position[1]), (position[0], position[1]-1), (position[0]+1, position[1]),
            (position[0], position[1]+1)]



class Board:

    def __init__(self, size, grid, goals, cars):
        self.size = size
        self.grid = grid
        self.goals = goals
        self.cars = cars


    def copy(self):
        return Board(self.size, pickle.loads(pickle.dumps(self.grid, -1)), self.goals, pickle.loads(pickle.dumps(self.cars, -1)))

    def __eq__(self, other):

        for car in self.cars:
            if self.cars[car] != other.cars[car]:
                return False
        return True

    def validPos(self, position):
        """ Verifies if position is with in the board and is occupied or blocked. """
        return position[0] >= 0 and position[1] >= 0 and position[0] < self.size and position[1] < self.size and self.grid[position] == 0

    def validAdj(self, position):
        valid = []
        for pos in adjacents(position):
            if self.validPos(pos):
                valid += [pos]
        return valid

    def changePosition(self, car, position):

        if self.validPos(position):
            self.grid[self.cars[car]] = 0
            self.cars[car] = position
            self.grid[position] = car
        else:
            print("Invalid Position")


    def isCompleted(self):
        return self.cars == self.goals

    def __repr__(self):
        return str(self.grid)


class GridWorldProblem:
    """
    A class that wraps around the GridWorld object. This version of the problem
    starts with an inital grid and then progressively changes the position of one car.
    """

    def __init__(self, initial):
        self.initial = GridWorldState(initial)
        self.expandedNodes = self.frontierNodes = 0

    def actions(self, state):
        """
        Returns a list to the legal actions that can be executed in the
        current state.
        """
        actionslst = []
        board = state.board
        for car in board.cars:
            valids = board.validAdj(board.cars[car])
            for pos in valids:
                yield (car, pos)

        return actionslst

    def result(self, state: GridWorldState, action):
        newBoard = state.board.copy()
        newBoard.changePosition(action[0], action[1])
        return GridWorldState(newBoard)

    def goal_test(self, state: GridWorldState):
        """Return True if the state is a goal."""
        return state.board.isCompleted()

    def path_cost(self, cost, state1, action, state2):
        return cost + 1




def breadth_first_search(problem):
    """ BFS """
    frontier = deque([Node(problem.initial)])  # FIFO queue

    while frontier:
        node = frontier.popleft()
        if problem.goal_test(node.state):
            problem.frontierNodes = len(frontier)
            return node
        expanded = node.expand(problem)
        problem.expandedNodes += len(expanded)
        frontier.extend(expanded)

    return None



def depth_first_search(problem):
    """ DFS """
    frontier = [(Node(problem.initial))]  # Stack

    explored =[]
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            problem.frontierNodes = len(frontier)
            return node

        explored += [node.state]
        expanded = node.expand(problem)
        problem.expandedNodes += len(expanded)
        frontier.extend(child for child in expanded
                        if child.state not in explored and child not in frontier)
    return None




def best_first_search(problem, f):

    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            problem.frontierNodes = len(frontier)
            return node
        explored.add(node.state)
        expanded = node.expand(problem)
        problem.expandedNodes += len(expanded)
        for child in expanded:
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None


def sumAllDist(node:Node):
    """ Heurist has the value of the sum of all Manhattan distances between a car and its goal """
    cars = node.state.board.cars
    goals = node.state.board.goals
    dist = 0
    for car in cars:
        dist += sum(abs(x - y) for x, y in zip(cars[car], goals[car]))
    return dist

def closestOneFirst(node:Node):
    """ Heurist has the value of the smallest Manhattan distances between a car and its goal """
    cars = node.state.board.cars
    goals = node.state.board.goals
    minDist = node.state.board.size**2
    for car in cars:
        dist = sum(abs(x - y) for x, y in zip(cars[car], goals[car]))
        if minDist > dist:
            minDist = dist
    return minDist

# Greedy best-first search is accomplished by specifying f(n) = h(n).
def greedy_search(problem, h):
    """Greedy"""
    return best_first_search(problem, h)

def astar_search(problem, h):
    """A*"""
    return best_first_search(problem, lambda n: n.path_cost + h(n))



def displayResults(solNode, start_time, end_time):
    if solNode != None:
        solution = [solNode]
        solCost = solNode.path_cost
        current = solNode
        while current != None:
            solution += [current]
            solCost += current.path_cost
            current = current.parent
        print("\n\nSuccess !")
        print("Solution cost: ", solCost)
        print("Expanded Nodes: ", problem.expandedNodes)
        print("Nodes in frontier: ", problem.frontierNodes)
        print("Solution: ")
        for sol in solution[1:][::-1]:
            print(sol)
    print("Processing time: ", end_time - start_time)

if __name__ == "__main__":

    size = eval(input("Board Size: "))
    blocked = []
    cars = []
    goals = []
    numberOfBlocks = eval(input("Number of Blocks: "))
    while numberOfBlocks > 0:
        blocked += [eval(input("Position: "))]
        numberOfBlocks -= 1
    numberOfCars = eval(input("Number of Cars: "))
    while numberOfCars > 0:
        cars += [eval(input("Car Position: "))]
        goals += [eval(input("Goal Position: "))]
        numberOfCars -= 1


    board = buildBoard(size, blocked, cars, goals)
    print("Board:")
    print(board)
    print("Goals:")
    print(board.goals)
    problem = GridWorldProblem(board)

    search = input("1. bfs\n2. dfs\n3. greedy\n4. astar\nEnter search method: ")
    while search not in ["1", "2", "3", "4"]:
        print("Invalid search enter 1/2/3/4")
        search = input("1. bfs\n2. dfs\n3. greedy\n4. astar\nEnter search method: ")

    if search == "1":
        print("\nstarting...")

        start_time = time.time()
        solNode = breadth_first_search(problem)
        end_time = time.time()
        displayResults(solNode, start_time, end_time)

    elif search == "2":
        print("\nstarting...")

        start_time = time.time()
        solNode = depth_first_search(problem)
        end_time = time.time()
        displayResults(solNode, start_time, end_time)

    elif search == "3":
        heuristic = input("\n1. sumAllDist\n2. closestOneFirst\nEnter heuristic: ")
        while heuristic not in ["1", "2"]:
            print("Invalid search enter 1/2")
            heuristic = input("\n1. sumAllDist\n2. closestOneFirst\nEnter heuristic: ")
        if heuristic == "1":
            print("\nstarting...")

            start_time = time.time()
            solNode = greedy_search(problem, sumAllDist)
            end_time = time.time()
            displayResults(solNode, start_time, end_time)
        if heuristic == "2":
            print("\nstarting...")

            start_time = time.time()
            solNode = greedy_search(problem, closestOneFirst)
            end_time = time.time()
            displayResults(solNode, start_time, end_time)

    elif search == "4":
        heuristic = input("\n1. sumAllDist\n2. closestOneFirst\nEnter heuristic: ")
        while heuristic not in ["1", "2"]:
            print("\nInvalid search enter 1/2")
            heuristic = input("\n1. sumAllDist\n2. closestOneFirst\nEnter heuristic: ")
        if heuristic == "1":
            print("\nstarting...")

            start_time = time.time()
            solNode = astar_search(problem, sumAllDist)
            end_time = time.time()
            displayResults(solNode, start_time, end_time)
        if heuristic == "2":
            print("\nstarting...")

            start_time = time.time()
            solNode = astar_search(problem, closestOneFirst)
            end_time = time.time()
            displayResults(solNode, start_time, end_time)
