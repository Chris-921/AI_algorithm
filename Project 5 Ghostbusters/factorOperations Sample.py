from typing import List
from bayesNet import Factor
import functools
from util import raiseNotDefined


def joinFactorsByVariableWithCallTracking(callTrackingList=None):
    def joinFactorsByVariable(factors: List[Factor], joinVariable: str):
        ...


joinFactorsByVariable = joinFactorsByVariableWithCallTracking()


def joinFactors(factors: List[Factor]):
    """
    Input factors is a list of factors.  

    Calculate the set of unconditioned variables and conditioned 
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries 
    are product of the corresponding rows of the input factors.

    May assume that the variableDomainsDict for all the input 
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in 
    one input factor (so their join is well defined).
    """
    setsOfUnconditioned = [set(factor.unconditionedVariables())
                           for factor in factors]
    if len(factors) > 1:
        intersect = functools.reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print("Factor failed joinFactors typecheck: ", factor)
            raise ValueError("unconditionedVariables can only appear in one factor. \n"
                             + "unconditionedVariables: " + str(intersect) +
                             "\nappear in more than one input factor.\n" +
                             "Input factors: \n" +
                             "\n".join(map(str, factors)))

    variableDomainsDict = {}
    condition = set()
    uncondition = set()
    for factor in factors:
        variableDomainsDict = factor.variableDomainsDict()
        condition.update(factor.conditionedVariables())
        uncondition.update(factor.unconditionedVariables())
    for x in uncondition:
        if x in condition:
            condition.remove(x)

    result = Factor(uncondition, condition, variableDomainsDict)
    allPossibleAssignmentDicts = result.getAllPossibleAssignmentDicts()
    for possibleCombination in allPossibleAssignmentDicts:
        probability = 1
        for factor in factors:
            probability *= factor.getProbability(possibleCombination)
        result.setProbability(possibleCombination, probability)
    return result


def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor: Factor, eliminationVariable: str):
        """
        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.

        Calculate the set of unconditioned variables and conditioned 
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.
        """
        ...

        variableDomainsDict = {}
        condition = set()
        uncondition = set()

        variableDomainsDict = factor.variableDomainsDict()
        condition = factor.conditionedVariables()
        uncondition = factor.unconditionedVariables()
        uncondition.remove(eliminationVariable)

        result = Factor(uncondition, condition, variableDomainsDict)
        allPossibleAssignmentDicts = result.getAllPossibleAssignmentDicts()
        eliminationDict = variableDomainsDict[eliminationVariable]
        for possibleCombination in allPossibleAssignmentDicts:
            probability = 0
            for eliminationVal in eliminationDict:
                possibleCombination[eliminationVariable] = eliminationVal
                probability += factor.getProbability(possibleCombination)
            result.setProbability(possibleCombination, probability)
        return result
    return eliminate


eliminate = eliminateWithCallTracking()
