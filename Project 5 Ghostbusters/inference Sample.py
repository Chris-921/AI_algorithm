import random
import itertools
from typing import List, Dict, Tuple
import busters
import game
import bayesNet as bn
from bayesNet import normalize
import hunters
from util import manhattanDistance, raiseNotDefined
from factorOperations import joinFactorsByVariableWithCallTracking, joinFactors
from factorOperations import eliminateWithCallTracking


def constructBayesNet(gameState: hunters.GameState):
    """
    Construct an empty Bayes net according to the structure given in Figure 1
    of the project description.

    - populate `variables` with the Bayes Net nodes
    - populate `edges` with every edge in the Bayes Net. we will represent each
      edge as a tuple `(from, to)`.
    - set each `variableDomainsDict[var] = values`, where `values` is a list
      of the possible assignments to `var`.
        - each agent position is a tuple (x, y) where x and y are 0-indexed
        - each observed distance is a noisy Manhattan distance:
          it's non-negative and |obs - true| <= MAX_NOISE
    """
    # constants to use
    PAC = "Pacman"
    GHOST0 = "Ghost0"
    GHOST1 = "Ghost1"
    OBS0 = "Observation0"
    OBS1 = "Observation1"
    X_RANGE = gameState.getWalls().width
    Y_RANGE = gameState.getWalls().height
    MAX_NOISE = 7
    variables = []
    edges = []
    variableDomainsDict = {}

    variables = [PAC, GHOST0, GHOST1, OBS0, OBS1]
    edges = [(GHOST0, OBS0), (PAC, OBS0), (PAC, OBS1), (GHOST1, OBS1)]

    PacGhostPossiablePos = []
    for i in range(X_RANGE):
        for j in range(Y_RANGE):
            PacGhostPossiablePos.append((i, j))

    for i in range(3):
        variableDomainsDict[variables[i]] = PacGhostPossiablePos

    variableDomainsDict[OBS0] = range(X_RANGE + Y_RANGE + MAX_NOISE - 1)
    variableDomainsDict[OBS1] = range(X_RANGE + Y_RANGE + MAX_NOISE - 1)

    net = bn.constructEmptyBayesNet(variables, edges, variableDomainsDict)
    return net


def inferenceByEnumeration(bayesNet: bn, queryVariables: List[str], evidenceDict: Dict):
    ...


def inferenceByVariableEliminationWithCallTracking(callTrackingList=None):

    def inferenceByVariableElimination(bayesNet: bn, queryVariables: List[str], evidenceDict: Dict, eliminationOrder: List[str]):
        """
        This function perform a probabilistic inference query that
        returns the factor:

        P(queryVariables | evidenceDict)

        It perform inference by interleaving joining on a variable
        and eliminating that variable, in the order of variables according
        to eliminationOrder.

        Use joinFactorsByVariable to join all of the factors 
        that contain a variable in order for the autograder to 
        recognize that you performed the correct interleaving of 
        joins and eliminates.

        If a factor that are about to eliminate a variable from has 
        only one unconditioned variable, we should not eliminate it 
        and instead just discard the factor.  This is since the 
        result of the eliminate would be 1, but it is not a 
        valid factor. So this simplifies using the result of eliminate.

        The sum of the probabilities should sum to one (so that it is a true 
        conditional probability, conditioned on the evidence).

        bayesNet:         The Bayes Net on which we are making a query.
        queryVariables:   A list of the variables which are unconditioned
                          in the inference query.
        evidenceDict:     An assignment dict {variable : value} for the
                          variables which are presented as evidence
                          (conditioned) in the inference query. 
        eliminationOrder: The order to eliminate the variables in.
        """
        joinFactorsByVariable = joinFactorsByVariableWithCallTracking(
            callTrackingList)
        eliminate = eliminateWithCallTracking(callTrackingList)
        if eliminationOrder is None:  # set an arbitrary elimination order if None given
            eliminationVariables = bayesNet.variablesSet() - set(queryVariables) -\
                set(evidenceDict.keys())
            eliminationOrder = sorted(list(eliminationVariables))

        factors = bayesNet.getAllCPTsWithEvidence(evidenceDict)
        for eliminateVariable in eliminationOrder:
            factors, table = joinFactorsByVariable(factors, eliminateVariable)
            if len(table.unconditionedVariables()) == 1:
                continue
            newTable = eliminate(table, eliminateVariable)
            factors.append(newTable)
        return normalize(joinFactors(factors))

    return inferenceByVariableElimination


inferenceByVariableElimination = inferenceByVariableEliminationWithCallTracking()


def sampleFromFactorRandomSource(randomSource=None):
    ...


class DiscreteDistribution(dict):
    ...

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.
        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """

        if not self:
            return

        total = self.total()
        if total != 0:
            for i in self.keys():
                self[i] = self[i] / total

    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.
        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        randomNum = random.random()
        keyrange = 0
        self.normalize()

        for key, value in self.items():
            keyrange += value
            if randomNum < keyrange:
                return key


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ...

    def getObservationProb(self, noisyDistance: int, pacmanPosition: Tuple, ghostPosition: Tuple, jailPosition: Tuple):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        if jailPosition == ghostPosition:
            if noisyDistance == None:
                return 1
            else:
                return 0

        if noisyDistance == None:
            return 0

        trueDistance = manhattanDistance(ghostPosition, pacmanPosition)
        return busters.getObservationProbability(noisyDistance, trueDistance)
    ...

    ...


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """

    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation: int, gameState: busters.GameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.
        """
        for pos in self.allPositions:
            probability = self.getObservationProb(
                observation, gameState.getPacmanPosition(), pos, self.getJailPosition())
            self.beliefs[pos] *= probability
        self.beliefs.normalize()

    def elapseTime(self, gameState: busters.GameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.
        """
        belief = DiscreteDistribution()
        for oldPos in self.allPositions:
            newPosDist = self.getPositionDistribution(gameState, oldPos)
            for newPos, probability in newPosDist.items():
                belief[newPos] += self.beliefs[oldPos] * \
                    probability
        belief.normalize()
        self.beliefs = belief

    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """

    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState: busters.GameState):
        """
        Initialize a list of particles. 
        """
        self.particles = []
        numParticleOnPos = self.numParticles // len(self.legalPositions)

        for pos in self.legalPositions:
            for _ in range(numParticleOnPos):
                self.particles.append(pos)

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        This function should return a normalized distribution.
        """
        probOfSingleParticle = 1 / self.numParticles
        belief = DiscreteDistribution()
        for pos in self.particles:
            belief[pos] += probOfSingleParticle
        belief.normalize()
        return belief

    def observeUpdate(self, observation: int, gameState: busters.GameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.
        """
        belief = DiscreteDistribution()
        for pos in self.particles:
            probOfParticle = self.getObservationProb(
                observation, gameState.getPacmanPosition(), pos, self.getJailPosition())
            belief[pos] += probOfParticle

        if belief.total() == 0:
            self.initializeUniformly(gameState)
            return

        belief.normalize()
        self.particles = [belief.sample() for _ in self.particles]

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        self.particles = [self.getPositionDistribution(
            gameState, pos).sample() for pos in self.particles]
