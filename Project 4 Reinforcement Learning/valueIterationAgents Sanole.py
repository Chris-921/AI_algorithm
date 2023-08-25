from learningAgents
import ValueEstimationAgent
import collections
import mdp
import util


class ValueIterationAgent(ValueEstimationAgent):

    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount=0.9, iterations=100):
        ...

    def runValueIteration(self):
        """
          Run the value iteration algorithm.In standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        for i in range(self.iterations):
          valueOnGrid = util.Counter()
          for state in self.mdp.getStates():
            maxValue = float("-inf")
            for action in self.mdp.getPossibleActions(state):
              QValue = self.computeQValueFromValues(state, action)
              if QValue > maxValue:
                maxValue = QValue
              valueOnGrid[state] = maxValue
          self.values = valueOnGrid

    def getValue(self, state):
        ...

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        QValue = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            QValue += prob * (self.mdp.getReward(state, action,
                              nextState) + self.discount * self.getValue(nextState))
        return QValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
        """
        actionList = self.mdp.getPossibleActions(state)
        maxValue = float('-inf')
        bestAction = None
        for action in actionList:
            value = self.computeQValueFromValues(state, action)
            if value > maxValue:
                maxValue = value
                bestAction = action
        return bestAction

  ...