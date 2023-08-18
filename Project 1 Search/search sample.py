import util
...


class SearchProblem:
    ...


def depthFirstSearch(problem: SearchProblem):
    stack = util.Stack()
    stack.push((problem.getStartState(), [], 0))
    visitedlist = []
    while stack:
        state, action, cost = stack.pop()
        if state not in visitedlist:
            if problem.isGoalState(state):
                return action
            visitedlist.append(state)
            for nextstate, nextaction, nextcost in problem.getSuccessors(state):
                stack.push((nextstate, action + [nextaction], cost+nextcost))


def breadthFirstSearch(problem: SearchProblem):
    queue = util.Queue()
    queue.push((problem.getStartState(), [], 0))
    visitedlist = []
    while queue:
        state, action, cost = queue.pop()
        if problem.isGoalState(state):
            return action
        if state not in visitedlist:
            visitedlist.append(state)
            for nextstate, nextaction, nextcost in problem.getSuccessors(state):
                queue.push((nextstate, action + [nextaction], cost+nextcost))


def uniformCostSearch(problem: SearchProblem):
    priorityQueue = util.PriorityQueue()
    priorityQueue.push((problem.getStartState(), [], 0), 0)
    visitedlist = []
    while priorityQueue:
        state, action, cost = priorityQueue.pop()
        if problem.isGoalState(state):
            return action
        if state not in visitedlist:
            visitedlist.append(state)
            for nextstate, nextaction, nextcost in problem.getSuccessors(state):
                priorityQueue.push(
                    (nextstate, action + [nextaction], cost+nextcost), cost+nextcost)


def nullHeuristic(state, problem=None):
    ...


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    priorityQueue = util.PriorityQueue()
    priorityQueue.push((problem.getStartState(), [], 0),
                       heuristic(problem.getStartState(), problem))
    visitedlist = []
    while priorityQueue:
        state, action, cost = priorityQueue.pop()
        if problem.isGoalState(state):
            return action
        if state not in visitedlist:
            visitedlist.append(state)
            for nextstate, nextaction, nextcost in problem.getSuccessors(state):
                if nextstate not in visitedlist:
                    priorityQueue.push((nextstate, action + [nextaction], 0), (problem.getCostOfActions(
                        action + [nextaction]) + heuristic(nextstate, problem)))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
