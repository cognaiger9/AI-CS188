# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def genericSearch(problem: SearchProblem, fringe: any):
    # fringe contain tuple (state, prev state, action prev state -> state)
    path = []

    start_state = problem.getStartState()
    fringe.push((start_state, None, None))
    parent_map = dict() # use to backtrack move (associate parent state with its child state as direction)
    cur_node = None
    cur_state = None
    while not fringe.isEmpty():
        cur_node = fringe.pop()
        cur_state = cur_node[0]
        if cur_state in parent_map:
            continue
        parent_map[cur_state] = (cur_node[1], cur_node[2])
        if problem.isGoalState(cur_state):
            break
        else:
            successors = problem.getSuccessors(cur_state)
            for successor in successors:
                if successor[0] not in parent_map:
                    fringe.push((successor[0], cur_state, successor[1]))

    # From goal state construct move backward
    while cur_state != start_state:
        item = parent_map[cur_state]
        path.insert(0, item[1])
        cur_state = item[0]

    return path

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    fringe = util.Stack()
    return genericSearch(problem, fringe)

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    
    fringe = util.Queue()
    return genericSearch(problem, fringe)

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    
    path = []

    fringe = util.PriorityQueue()                       # fringe contain tuple (state, prev state, action prev state -> state, cost from start)
    start_state = problem.getStartState()
    fringe.push((start_state, None, None, 0), 0)
    parent_map = dict() # use to backtrack move (associate parent state with its child state as direction)
    cur_node = None
    cur_state = None

    while not fringe.isEmpty():
        cur_node = fringe.pop()
        cur_state = cur_node[0]
        if cur_state in parent_map:
            continue
        parent_map[cur_state] = (cur_node[1], cur_node[2])
        if problem.isGoalState(cur_state):
            break
        else:
            successors = problem.getSuccessors(cur_state)
            for successor in successors:
                if successor[0] not in parent_map:
                    fringe.update((successor[0], cur_state, successor[1], successor[2] + cur_node[3]), successor[2] + cur_node[3])

    # From goal state construct move backward
    while cur_state != start_state:
        item = parent_map[cur_state]
        path.insert(0, item[1])
        cur_state = item[0]

    return path

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    path = []

    fringe = util.PriorityQueue()                       # fringe contain tuple (state, prev state, action prev state -> state, cost from start)
    start_state = problem.getStartState()
    fringe.push((start_state, None, None, 0), 0)
    parent_map = dict() # use to backtrack move (associate parent state with its child state as direction)
    cur_node = None
    cur_state = None

    while not fringe.isEmpty():
        cur_node = fringe.pop()
        cur_state = cur_node[0]
        if cur_state in parent_map:
            continue
        parent_map[cur_state] = (cur_node[1], cur_node[2])
        if problem.isGoalState(cur_state):
            break
        else:
            successors = problem.getSuccessors(cur_state)
            for successor in successors:
                if successor[0] not in parent_map:
                    fringe.update((successor[0], cur_state, successor[1], successor[2] + cur_node[3]), successor[2] + cur_node[3] + heuristic(successor[0], problem))

    # From goal state construct move backward
    while cur_state != start_state:
        item = parent_map[cur_state]
        path.insert(0, item[1])
        cur_state = item[0]

    return path

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
