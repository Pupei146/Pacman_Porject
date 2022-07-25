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

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        """
        print('successorGameState: ', successorGameState) 
        print('newPos: ', newPos) # get the location of pacman
        print('newFood:', newFood[1][2]) # reamining food
        print('newFood:', type(newFood))
        print('newGhostStates: ', newGhostStates)
        print('newScaredTimes: ', newScaredTimes)
        """
        
        G_Pos = successorGameState.getGhostPositions() # store the position of ghost
        P_Pos = newPos # store the position of pacman
        
        distances = []
        for G in G_Pos:
            distances.append(abs(P_Pos[0] - G[0]) + abs(P_Pos[1] - G[1])) # manhattan distance between G and P
        
        # exam the distance to each of the ghost, the closer, the lower score
        G_score = 0
        for d in distances:
            if d < 5 and d != 0: # only consider the ghost that is 5 steps close
                G_score -= 21 / d
            
        foods = newFood.asList()
        food_distances = []
        for F in foods:
            food_distances.append(abs(P_Pos[0] - F[0]) + abs(P_Pos[1] - F[1])) # manhattan distance between food and P        
        
        F_score = 0
        
        if min(distances) > 2 and len(foods) != 0: # if no ghost is around the pacman
            F_score += 10 / min(food_distances) 
        
        if len(foods) == 0:
            F_score += 20
        
        # if the dot is eaten, the score will increase 
        Eating_score = 0
        if len(foods) > 0:
            Eating_score += 20 / len(foods) 
        
        initial_score = successorGameState.getScore()
        
        total_score = G_score + F_score + initial_score + Eating_score
        
        
        return total_score


def scoreEvaluationFunction(currentGameState):
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
            
    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        

        
        def minMax(agentIndex, gameState, Depth):
            # end criteria
            if Depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            # during pacman's turn, maximize the score
            if agentIndex == 0:
                v = float("-inf")
                actions = gameState.getLegalActions(agentIndex)
                for action in actions:
                    v = max(v, minMax(1, gameState.generateSuccessor(0, action), Depth))
                return v
            
            # during ghosts' turn, minimize the score
            else:
                v = float("inf")
                actions = gameState.getLegalActions(agentIndex)
                if (agentIndex == gameState.getNumAgents() -1):
                    Depth += 1
                    for action in actions:
                        v = min(v, minMax(0, gameState.generateSuccessor(agentIndex, action), Depth))
                else:
                    for action in actions:
                        v = min(v, minMax(agentIndex + 1, gameState.generateSuccessor(agentIndex, action), Depth))
                return v

                
        Depth = 0
        actions = gameState.getLegalActions(0)
        P_moves = util.PriorityQueue()

        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            v = minMax(1, successor, Depth) # after Pacman's turn, there should be Ghost's turn
            P_moves.push(action, v)
        
        
        while not P_moves.isEmpty():
            move = P_moves.pop() # return the action with highest score
            #print(move)
        return move


        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        
       

        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        def ab_minMax(agentIndex, gameState, Depth, alpha, beta):
            # end criteria
            if Depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            # during pacman's turn, maximize the score
            if agentIndex == 0:
                v = float("-inf")
                actions = gameState.getLegalActions(agentIndex)
                for action in actions:
                    v = max(v, ab_minMax(1, gameState.generateSuccessor(0, action), Depth, alpha, beta))
                    if v > beta:
                        return v
                    alpha = max(alpha, v)
                return v
            
            # during ghosts' turn, minimize the score
            else:
                v = float("inf")
                actions = gameState.getLegalActions(agentIndex)
                if (agentIndex == gameState.getNumAgents() -1):
                    Depth += 1
                    for action in actions:
                        v = min(v, ab_minMax(0, gameState.generateSuccessor(agentIndex, action), Depth, alpha, beta))
                        if v < alpha:
                            return v
                        beta = min(beta, v)
                else:
                    for action in actions:
                        v = min(v, ab_minMax(agentIndex + 1, gameState.generateSuccessor(agentIndex, action), Depth, alpha, beta))
                        if v < alpha:
                            return v
                        beta = min(beta, v)
                return v
        
                v = float("-inf")
                
        Depth = 0
        actions = gameState.getLegalActions(0)
        P_moves = util.PriorityQueue()
        alpha = float("-inf")
        beta = float("inf")
        
        
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            #score_to_beat = score
            v = ab_minMax(1, successor, Depth, alpha, beta) # after Pacman's turn, there should be Ghost's turn
            if v > beta:
                return action
            alpha = max(alpha, v)
            P_moves.push(action, v)  
        
        while not P_moves.isEmpty():
            move = P_moves.pop() # return the action with highest score
            #print(move)
        return move      

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        #my code

        def expMax(agentIndex, gameState, Depth):
            # end criteria
            if Depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            # during pacman's turn, maximize the score
            if agentIndex == 0:
                v = float("-inf")
                actions = gameState.getLegalActions(agentIndex)
                for action in actions:
                    v = max(v, expMax(1, gameState.generateSuccessor(0, action), Depth))
                return v

            # during ghosts' turn, minimize the score
            else:
                v = 0
                actions = gameState.getLegalActions(agentIndex)
                if (agentIndex == gameState.getNumAgents() - 1):
                    Depth += 1
                    for action in actions:
                        v += expMax(0, gameState.generateSuccessor(agentIndex, action), Depth)
                else:
                    for action in actions:
                        v += expMax(agentIndex + 1, gameState.generateSuccessor(agentIndex, action), Depth)
                return v/len(actions)

        Depth = 0
        actions = gameState.getLegalActions(0)
        best_action = Directions.STOP
        V = float("-inf")

        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            v = expMax(1, successor, Depth)  # after Pacman's turn, there should be Ghost's turn
            if v > V:
                V = v
                best_action = action

        return best_action


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
      In this function, three scores are considered as state evaluation. Firstly, the pacman's distance to the nearest
      food. Secondly, pacman's distance to the ghost when the ghost is scared. Finally, pacman's distance to the ghost
      when the ghost is not scared. For food distances, the closer's the higher, for ghost distance when scared, the
      closer the higher, and when not scared, the closer the lower.
    """
    "*** YOUR CODE HERE ***"

    #my code

    newFood = currentGameState.getFood()
    score = currentGameState.getScore()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


    G_Pos = currentGameState.getGhostPositions()  # store the position of ghost
    P_Pos = currentGameState.getPacmanPosition()  # store the position of pacman
    foods = newFood.asList()

    F_score = 0
    food_distances = []
    for F in foods:
        food_distances.append(abs(P_Pos[0] - F[0]) + abs(P_Pos[1] - F[1]))  # manhattan distance
    if len(food_distances) > 0:
        F_score += 10 / min(food_distances) #only consider the nearest food
    else:
        F_score += 10

    G_score = 0

    ghost_distance = (abs(P_Pos[0] - G_Pos[0][0]) + abs(P_Pos[1] - G_Pos[0][1]))
    if ghost_distance > 0:
        if newScaredTimes[0] > 0:
            G_score += 200 / ghost_distance
        else:
            G_score -= 21 / ghost_distance
    else:
        return float("-inf")

    return score + F_score + G_score


















# Abbreviation
better = betterEvaluationFunction

