from typing import List, Tuple, Any
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import pacman


class GoWestAgent(Agent):
    ...


class SearchAgent(Agent):
    ...


class PositionSearchProblem(search.SearchProblem):
    ...


class StayEastSearchAgent(SearchAgent):
    ...


class StayWestSearchAgent(SearchAgent):
    ...


def manhattanHeuristic(position, problem, info={}):
    ...


def euclideanHeuristic(position, problem, info={}):
    ...


class CornersProblem(search.SearchProblem):
    def __init__(self, startingGameState: pacman.GameState):
        ...

    def getStartState(self):
        cornerVisited = [False, False, False, False]
        return (self.startingPosition, cornerVisited)

    def isGoalState(self, state: Any):
        return state[1] == [True, True, True, True]

    def getSuccessors(self, state: Any):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state[0]
            stateVisited = state[1]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]
            if not hitsWall:
                nextState = (nextx, nexty)
                nextVisited = list(stateVisited)
                for i in range(4):
                    if self.corners[i] == nextState:
                        nextVisited[i] = True
                successors.append(((nextState, nextVisited), action, 1))
        self._expanded += 1  # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        ...


def cornersHeuristic(state: Any, problem: CornersProblem):
    corners = problem.corners  # These are the corner coordinates
    # These are the walls of the maze, as a Grid (game.py)
    walls = problem.walls

    x1, y1 = state[0]
    visited = state[1]
    if visited == [True, True, True, True]:
        return 0

    min_so_far = 0
    for i in range(4):
        if visited[i] == False:
            x2, y2 = corners[i]
            distance = util.manhattanDistance((x1, y1), (x2, y2))
            if distance > min_so_far:
                min_so_far = distance
    return min_so_far


class AStarCornersAgent(SearchAgent):
    ...


class FoodSearchProblem:
    ...


class AStarFoodSearchAgent(SearchAgent):
    ...


def foodHeuristic(state: Tuple[Tuple, List[List]], problem: FoodSearchProblem):
    position, foodGrid = state
    x1, y1 = state[0]
    foodGrid = state[1]
    if foodGrid.count == 0:
        return 0

    min_so_far = 0
    for x2, y2 in foodGrid.asList():
        if foodGrid[x2][y2] == True:
            distance = mazeDistance(
                (x1, y1), (x2, y2), problem.startingGameState)
            if distance > min_so_far:
                min_so_far = distance
    return min_so_far


class ClosestDotSearchAgent(SearchAgent):
    ...

    def findPathToClosestDot(self, gameState: pacman.GameState):
        ...
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)
        return search.breadthFirstSearch(problem)


class AnyFoodSearchProblem(PositionSearchProblem):
    def __init__(self, gameState):
        ...

    def isGoalState(self, state: Tuple[int, int]):
        x, y = state
        return self.food[x][y]


def mazeDistance(point1: Tuple[int, int], point2: Tuple[int, int], gameState: pacman.GameState) -> int:
    ...
