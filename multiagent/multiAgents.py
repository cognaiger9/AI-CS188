# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        currentFood = currentGameState.getFood()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        foodList = currentFood.asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        for agent in range(len(newGhostStates)):
            ghostPos = newGhostStates[agent].getPosition()
            if manhattanDistance(newPos, ghostPos) <= 1:
                return 0
                        
        min_distance = -1
        for food in foodList:
            distance = manhattanDistance(food, newPos)
            if min_distance == -1 or distance < min_distance:
                min_distance = distance

        return 1 / (min_distance + 1) # avoid case min_distance = 0

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getValueOfState(self, state: GameState, agentIdx, depth):
        numAgent = state.getNumAgents()
        possibleActions = state.getLegalActions(agentIdx)

        if len(possibleActions) == 0:                   # Base case: leaf
            return self.evaluationFunction(state)
    
        if agentIdx == 0 and depth == 0:                # Base case: leaf
            return self.evaluationFunction(state)
        
        state_values = []
        for action in possibleActions:
            nextState = state.generateSuccessor(agentIdx, action)
            val = None
            if agentIdx == numAgent - 1:
                val = self.getValueOfState(nextState, 0, depth - 1)
            else:
                val = self.getValueOfState(nextState, agentIdx + 1, depth)
            state_values.append(val)

        if agentIdx == 0:
            return max(state_values)
        
        return min(state_values)

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        
        possibleActions = gameState.getLegalActions()
        nextAction = None
        max_val = float('-inf')
        for action in possibleActions:
            nextState = gameState.generateSuccessor(0, action)
            val = self.getValueOfState(nextState, 1, self.depth)
            if val > max_val:
                max_val = val
                nextAction = action

        #print(max_val)
        return nextAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getValueOfState(self, state: GameState, alpha, beta, agentIdx, depth):
        numAgent = state.getNumAgents()
        possibleActions = state.getLegalActions(agentIdx)

        if len(possibleActions) == 0:                   # Base case: leaf
            return self.evaluationFunction(state)
        
        if agentIdx == 0 and depth == 0:                # Base case: leaf
            return self.evaluationFunction(state)
        
        if agentIdx == 0: # Pacman
            v = float('-inf')
            for action in possibleActions:
                nextState = state.generateSuccessor(agentIdx, action)
                v = max(v, self.getValueOfState(nextState, alpha, beta, agentIdx + 1, depth))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            
            return v
        
        if agentIdx == numAgent - 1:
            v = float('inf')
            for action in possibleActions:
                nextState = state.generateSuccessor(agentIdx, action)
                v = min(v, self.getValueOfState(nextState, alpha, beta, 0, depth - 1))
                if v < alpha:
                    return v
                beta = min(beta, v)

            return v
        
        v = float('inf')
        for action in possibleActions:
            nextState = state.generateSuccessor(agentIdx, action)
            v = min(v, self.getValueOfState(nextState, alpha, beta, agentIdx + 1, depth))
            if v < alpha:
                return v
            beta = min(beta, v)

        return v

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        
        possibleActions = gameState.getLegalActions()
        nextAction = None
        max_val = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in possibleActions:
            nextState = gameState.generateSuccessor(0, action)
            val = self.getValueOfState(nextState, alpha, beta, 1, self.depth)
            if val > max_val:
                max_val = val
                nextAction = action
            alpha = max(alpha, max_val)

        return nextAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getValueOfState(self, state: GameState, agentIdx, depth):
        numAgent = state.getNumAgents()
        possibleActions = state.getLegalActions(agentIdx)

        if len(possibleActions) == 0:                   # Base case: leaf
            return self.evaluationFunction(state)
        
        if agentIdx == 0 and depth == 0:                # Base case: leaf
            return self.evaluationFunction(state)

        state_values = []
        if agentIdx == 0: # Pacman
            for action in possibleActions:
                nextState = state.generateSuccessor(agentIdx, action)
                val = self.getValueOfState(nextState, agentIdx + 1, depth)
                state_values.append(val)
            
            return max(state_values)
        
        numAction = len(possibleActions)
        val = 0
        for action in possibleActions:
            nextState = state.generateSuccessor(agentIdx, action)
            if agentIdx == numAgent - 1:
                val += self.getValueOfState(nextState, 0, depth - 1) / numAction
            else:
                val += self.getValueOfState(nextState, agentIdx + 1, depth)

        return val

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        
        possibleActions = gameState.getLegalActions()
        nextAction = None
        max_val = float('-inf')
        for action in possibleActions:
            nextState = gameState.generateSuccessor(0, action)
            val = self.getValueOfState(nextState, 1, self.depth)
            if val > max_val:
                max_val = val
                nextAction = action

            #print(f"val = {val}, action = {action}")
        return nextAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    
    # Avoid ghost
    numAgents = currentGameState.getNumAgents()
    ghostPos = currentGameState.getGhostPositions()
    pacmanPos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore() # state more sooner -> higher score (avoid thrashing)
    val1 = 0
    for i in range(0, numAgents - 1):
        #if manhattanDistance(ghostPos[i], pacmanPos) <= 1:
        #    return 0
        val1 += manhattanDistance(ghostPos[i], pacmanPos) / (numAgents - 1)

    # Nearest food
    foods = currentGameState.getFood().asList()
    numFoodLeft = len(foods)
    min_distance = -1
    for food in foods:
        distance = manhattanDistance(food, pacmanPos)
        if min_distance == -1 or distance < min_distance:
            min_distance = distance

    val = 1 / min_distance + 1 / (numFoodLeft + 1) + score + val1 / 100 # practical show ghost contribute no effect
    return val

# Abbreviation
better = betterEvaluationFunction
