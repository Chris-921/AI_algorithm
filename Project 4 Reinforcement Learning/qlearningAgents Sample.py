from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import gridworld
import random
import util
import math
import copy


class QLearningAgent(ReinforcementAgent):

    def __init__(self, **args):
        # Initialize Q-values
        ReinforcementAgent.__init__(self, **args)
        self.values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Return 0.0 if we have never seen a state 
          or the Q node value otherwise
        """
        return self.values[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0
        return max([self.getQValue(state, action) for action in legalActions])

    def computeActionFromQValues(self, state):
        # Compute the best action to take in a state.
        legalActions = self.getLegalActions(state)
        maxValue = float('-inf')
        bestAction = None
        for action in legalActions:
            value = self.getQValue(state, action)
            if value > maxValue:
                maxValue = value
                bestAction = action
        return bestAction

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.
        """
        legalActions = self.getLegalActions(state)
        action = None
        if legalActions:
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            else:
                action = self.getPolicy(state)
        return action

    def update(self, state, action, nextState, reward: float):
        self.values[(state, action)] = (1 - self.alpha)*self.getQValue(state, action) + self.alpha * \
            (reward + self.discount *
             self.getValue(nextState))

    ...


class PacmanQAgent(QLearningAgent):
    ...


class ApproximateQAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        ...

    def getWeights(self):
        ...

    def getQValue(self, state, action):
        """
          Return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        total = 0
        for i in features:
            total += features[i] * self.getWeights()[i]
        return total

    def update(self, state, action, nextState, reward: float):
        """
           Update weights based on transition
        """
        features = self.featExtractor.getFeatures(state, action)
        difference = (reward + self.discount *
                      self.getValue(nextState)) - self.getQValue(state, action)
        for i in self.weights:
            self.weights[i] += self.alpha * difference * features[i]

    ...
