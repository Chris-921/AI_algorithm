from distanceCalculator import Distancer
from game import Actions
import util
from util import raiseNotDefined
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters


class NullGraphics:
    ...


class KeyboardInference(inference.InferenceModule):
    ...


class BustersAgent:
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__(self, index=0, inference="ExactInference", ghostAgents=None, observeEnable=True, elapseTimeEnable=True):
        ...

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        ...

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        ...

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        ...

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        ...


class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."
    ...


class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState: busters.GameState):
        ...

    def chooseAction(self, gameState: busters.GameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closest to the closest ghost (according to mazeDistance!).
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]

        lowestDis = float('inf')
        action = None
        for position in livingGhostPositionDistributions:
            maxProbGhostPos = position.argMax()

        for act in legal:
            distance = self.distancer.getDistance(
                Actions.getSuccessor(pacmanPosition, act), maxProbGhostPos)
            if lowestDis > distance:
                action = act
                lowestDis = distance
        return action
