
from util import manhattanDistance
from game import Directions
import random
import util
from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    def getAction(self, gameState: GameState):
        legalMoves = gameState.getLegalActions()
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()
        if action == 'Stop':
            return float('-inf')
        for ghoststate in newGhostStates:
            if ghoststate.getPosition() == newPos and ghoststate.scaredTimer == 0:
                return float('-inf')
        foodDis = []
        for food in currentGameState.getFood().asList():
            dis = abs(food[0] - newPos[0]) + abs(food[1] - newPos[1])
            foodDis.append(dis)
        return -1 * min(foodDis)


def scoreEvaluationFunction(currentGameState: GameState):
    ...


class MultiAgentSearchAgent(Agent):
    ...


class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState):
        def minimax(state, depth, agent):
            if (agent == 0 and depth == 0) or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            action = None
            nextAgent = (agent + 1) % state.getNumAgents()
            if agent == 0:
                maxEval = float('-inf')
                for legalAction in state.getLegalActions(agent):
                    eval, act = minimax(state.generateSuccessor(
                        agent, legalAction), depth - 1, nextAgent)
                    if max(maxEval, eval) == eval:
                        action = legalAction
                        maxEval = max(maxEval, eval)
                return maxEval, action
            else:
                minEval = float('inf')
                for legalAction in state.getLegalActions(agent):
                    eval, act = minimax(state.generateSuccessor(
                        agent, legalAction), depth, nextAgent)
                    if min(minEval, eval) == eval:
                        action = legalAction
                        minEval = min(minEval, eval)
                return minEval, action

        val, action = minimax(gameState, self.depth, self.index)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    # Minimax agent with alpha-beta pruning
    def getAction(self, gameState: GameState):

        def AlphaBetaAgent(state, depth, alpha, beta, agent):
            if (agent == 0 and depth == 0) or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            action = None
            nextAgent = (agent + 1) % state.getNumAgents()
            if agent == 0:
                maxEval = float('-inf')
                for legalAction in state.getLegalActions(agent):
                    eval, act = AlphaBetaAgent(state.generateSuccessor(
                        agent, legalAction), depth - 1, alpha, beta, nextAgent)
                    if max(maxEval, eval) == eval:
                        action = legalAction
                        maxEval = max(maxEval, eval)
                    alpha = max(alpha, maxEval)
                    if maxEval > beta:
                        return maxEval, action
                return maxEval, action
            else:
                minEval = float('inf')
                for legalAction in state.getLegalActions(agent):
                    eval, act = AlphaBetaAgent(state.generateSuccessor(
                        agent, legalAction), depth, alpha, beta, nextAgent)
                    if min(minEval, eval) == eval:
                        action = legalAction
                        minEval = min(minEval, eval)
                    beta = min(beta, minEval)
                    if minEval < alpha:
                        return minEval, action
                return minEval, action

        val, action = AlphaBetaAgent(
            gameState, self.depth, float('-inf'), float('inf'), self.index)
        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    # Expectimax agent
    def getAction(self, gameState: GameState):

        def expectimax(state, depth, agent):
            if (agent == 0 and depth == 0) or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            action = None
            nextAgent = (agent + 1) % state.getNumAgents()
            if agent == 0:
                maxEval = float('-inf')
                for legalAction in state.getLegalActions(agent):
                    eval, act = expectimax(state.generateSuccessor(
                        agent, legalAction), depth - 1, nextAgent)
                    if max(maxEval, eval) == eval:
                        action = legalAction
                        maxEval = max(maxEval, eval)
                return maxEval, action
            else:
                legalAction = state.getLegalActions(agent)
                expectVal = 0
                prob = 1 / float(len(legalAction))
                for a in legalAction:
                    eval, act = expectimax(
                        state.generateSuccessor(agent, a), depth, nextAgent)
                    expectVal += eval * prob
                return expectVal, action

        val, action = expectimax(gameState, self.depth, self.index)
        return action


def betterEvaluationFunction(currentGameState: GameState):
    # Extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function
    if currentGameState.isWin():
        return float("inf")
    elif currentGameState.isLose():
        return float("-inf")

    score = 0
    ghosts = currentGameState.getGhostStates()
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    walls = currentGameState.getWalls()

    for g in ghosts:
        PacGhostDistance = manhattanDistance(pos, g.getPosition())
        if g.scaredTimer != 0:
            score += 2
        else:
            score -= 1 / PacGhostDistance

    visitedlist = []
    currentDis = walls.copy()
    queue = util.Queue()
    queue.push(pos)
    currentDis[pos[0]][pos[1]] = 0
    dis = 0
    while queue:
        (x, y) = queue.pop()
        if food[x][y]:
            break
        dis = currentDis[x][y] + 1
        if (x, y) not in visitedlist:
            visitedlist.append((x, y))
            if currentDis[x + 1][y] == 0:
                currentDis[x + 1][y] = dis
                queue.push((x + 1, y))
            if currentDis[x - 1][y] == 0:
                currentDis[x - 1][y] = dis
                queue.push((x - 1, y))
            if currentDis[x][y + 1] == 0:
                currentDis[x][y + 1] = dis
                queue.push((x, y + 1))
            if currentDis[x][y - 1] == 0:
                currentDis[x][y - 1] = dis
                queue.push((x, y - 1))
    score += 1 / dis
    score -= 2 * food.count()
    return score


# Abbreviation
better = betterEvaluationFunction
