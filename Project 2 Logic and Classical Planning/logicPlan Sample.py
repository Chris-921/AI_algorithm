import copy
import itertools
from logic import PropSymbolExpr, Expr, to_cnf, pycoSAT, parseExpr, pl_true
from logic import conjoin, disjoin
import game
import logic
import sys
import util
from typing import Dict, List, Tuple, Callable, Generator, Any
...


pacman_str = 'P'
food_str = 'FOOD'
wall_str = 'WALL'
pacman_wall_str = pacman_str + wall_str
DIRECTIONS = ['North', 'South', 'East', 'West']
blocked_str_map = dict(
    [(direction, (direction + "_blocked").upper()) for direction in DIRECTIONS])
geq_num_adj_wall_str_map = dict(
    [(num, "GEQ_{}_adj_walls".format(num)) for num in range(1, 4)])
DIR_TO_DXDY_MAP = {'North': (0, 1), 'South': (
    0, -1), 'East': (1, 0), 'West': (-1, 0)}

...


def findModel(sentence: Expr) -> Dict[Expr, bool]:
    ...


def findModelUnderstandingCheck() -> Dict[Expr, bool]:
    a = Expr('A')
    print("a.__dict__ is:", a.__dict__)  # might be helpful for getting ideas
    a.op = 'a'
    return {a: True}


def entails(premise: Expr, conclusion: Expr) -> bool:
    P = premise
    C = conclusion
    return not findModel(P & ~C)


def plTrueInverse(assignments: Dict[Expr, bool], inverse_statement: Expr) -> bool:
    return pl_true(~inverse_statement, assignments)


def atLeastOne(literals: List[Expr]) -> Expr:
    return disjoin(literals)


def atMostOne(literals: List[Expr]) -> Expr:
    print(literals[0])
    Pairinlist = list(itertools.combinations(literals, 2))
    print(Pairinlist[0])
    i = 0
    while i < len(Pairinlist):
        Pairinlist[i] = disjoin(~Pairinlist[i][0], ~Pairinlist[i][1])
        i = i+1
    print(Pairinlist[0])
    return conjoin(Pairinlist)


def exactlyOne(literals: List[Expr]) -> Expr:
    return atMostOne(literals) & atLeastOne(literals)


def pacmanSuccessorAxiomSingle(x: int, y: int, time: int, walls_grid: List[List[bool]] = None) -> Expr:
    now, last = time, time - 1
    possible_causes: List[Expr] = []
    if walls_grid[x][y+1] != 1:
        possible_causes.append(PropSymbolExpr(pacman_str, x, y+1, time=last)
                               & PropSymbolExpr('South', time=last))
    if walls_grid[x][y-1] != 1:
        possible_causes.append(PropSymbolExpr(pacman_str, x, y-1, time=last)
                               & PropSymbolExpr('North', time=last))
    if walls_grid[x+1][y] != 1:
        possible_causes.append(PropSymbolExpr(pacman_str, x+1, y, time=last)
                               & PropSymbolExpr('West', time=last))
    if walls_grid[x-1][y] != 1:
        possible_causes.append(PropSymbolExpr(pacman_str, x-1, y, time=last)
                               & PropSymbolExpr('East', time=last))
    if not possible_causes:
        return None
    return PropSymbolExpr(pacman_str, x, y, time=now) % disjoin(possible_causes)


def SLAMSuccessorAxiomSingle(x: int, y: int, time: int, walls_grid: List[List[bool]]) -> Expr:
    now, last = time, time - 1
    # enumerate all possible causes for P[x,y]_t, assuming moved to having moved
    moved_causes: List[Expr] = []
    if walls_grid[x][y+1] != 1:
        moved_causes.append(PropSymbolExpr(pacman_str, x, y+1, time=last)
                            & PropSymbolExpr('South', time=last))
    if walls_grid[x][y-1] != 1:
        moved_causes.append(PropSymbolExpr(pacman_str, x, y-1, time=last)
                            & PropSymbolExpr('North', time=last))
    if walls_grid[x+1][y] != 1:
        moved_causes.append(PropSymbolExpr(pacman_str, x+1, y, time=last)
                            & PropSymbolExpr('West', time=last))
    if walls_grid[x-1][y] != 1:
        moved_causes.append(PropSymbolExpr(pacman_str, x-1, y, time=last)
                            & PropSymbolExpr('East', time=last))
    if not moved_causes:
        return None

    moved_causes_sent: Expr = conjoin([~PropSymbolExpr(
        pacman_str, x, y, time=last), ~PropSymbolExpr(wall_str, x, y), disjoin(moved_causes)])

    # using merged variables, improves speed significantly
    failed_move_causes: List[Expr] = []
    auxilary_expression_definitions: List[Expr] = []
    for direction in DIRECTIONS:
        dx, dy = DIR_TO_DXDY_MAP[direction]
        wall_dir_clause = PropSymbolExpr(
            wall_str, x + dx, y + dy) & PropSymbolExpr(direction, time=last)
        wall_dir_combined_literal = PropSymbolExpr(
            wall_str + direction, x + dx, y + dy, time=last)
        failed_move_causes.append(wall_dir_combined_literal)
        auxilary_expression_definitions.append(
            wall_dir_combined_literal % wall_dir_clause)

    failed_move_causes_sent: Expr = conjoin([
        PropSymbolExpr(pacman_str, x, y, time=last),
        disjoin(failed_move_causes)])

    return conjoin([PropSymbolExpr(pacman_str, x, y, time=now) % disjoin([moved_causes_sent, failed_move_causes_sent])] + auxilary_expression_definitions)


def pacphysicsAxioms(t: int, all_coords: List[Tuple], non_outer_wall_coords: List[Tuple], walls_grid: List[List] = None, sensorModel: Callable = None, successorAxioms: Callable = None) -> Expr:
    pacphysics_sentences = []
    legalLocation = []
    actions = []
    # if a wall is at (x,y), then Pacman is not at (x,y) at t
    for (x, y) in all_coords:
        pacphysics_sentences.append(PropSymbolExpr(
            wall_str, x, y) >> ~PropSymbolExpr(pacman_str, x, y, time=t))
    # Pacman is exactly one of the non_outer_wall_coords at t
    for (x, y) in non_outer_wall_coords:
        legalLocation.append(PropSymbolExpr(pacman_str, x, y, time=t))
    pacphysics_sentences.append(exactlyOne(legalLocation))
    # Pacman take exactly one of the four actions in DIRECTIONS at t
    for dir in DIRECTIONS:
        actions.append(PropSymbolExpr(dir, time=t))
    pacphysics_sentences.append(exactlyOne(actions))

    if sensorModel is not None:
        pacphysics_sentences.append(sensorModel(t, non_outer_wall_coords))

    if successorAxioms is not None and t > 0:
        pacphysics_sentences.append(successorAxioms(
            t, walls_grid, non_outer_wall_coords))

    return conjoin(pacphysics_sentences)


def checkLocationSatisfiability(x1_y1: Tuple[int, int], x0_y0: Tuple[int, int], action0, action1, problem):
    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(
        range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(
        range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))
    KB = []
    x0, y0 = x0_y0
    x1, y1 = x1_y1
    # We know which coords are walls:
    map_sent = [PropSymbolExpr(wall_str, x, y) for x, y in walls_list]
    KB.append(conjoin(map_sent))
    KB.append(pacphysicsAxioms(0, all_coords,
              non_outer_wall_coords, walls_grid))
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time=0))
    KB.append(PropSymbolExpr(action0, time=0))
    KB.append(pacphysicsAxioms(1, all_coords,
              non_outer_wall_coords, walls_grid))
    KB.append(allLegalSuccessorAxioms(1, walls_grid, non_outer_wall_coords))
    KB.append(PropSymbolExpr(action1, time=1))
    KB = conjoin(KB)
    model1 = findModel(KB & PropSymbolExpr(pacman_str, x1, y1, time=1))
    model2 = findModel(KB & ~PropSymbolExpr(pacman_str, x1, y1, time=1))
    return (model1, model2)


def positionLogicPlan(problem) -> List:
    walls_grid = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls_grid.asList()
    x0, y0 = problem.startState
    xg, yg = problem.goal
    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2),
                                        range(height + 2)))
    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = ['North', 'South', 'East', 'West']
    KB = []

    KB.append(PropSymbolExpr(pacman_str, x0, y0, time=0))
    for t in range(50):
        paclocation = []
        for (x, y) in non_wall_coords:
            paclocation.append(PropSymbolExpr(pacman_str, x, y, time=t))
        KB.append(exactlyOne(paclocation))
        model = findModel(conjoin(KB) & PropSymbolExpr(
            pacman_str, xg, yg, time=t))
        if model:
            return extractActionSequence(model, actions)
        exactOneAction = []
        for action in DIRECTIONS:
            exactOneAction.append(PropSymbolExpr(action, time=t))
        KB.append(exactlyOne(exactOneAction))
        for (x, y) in non_wall_coords:
            KB.append(conjoin(pacmanSuccessorAxiomSingle(x, y, t+1, walls_grid)))


def foodLogicPlan(problem) -> List:
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls.asList()
    (x0, y0), food = problem.start
    food = food.asList()
    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), range(height + 2)))
    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = ['North', 'South', 'East', 'West']
    KB = []

    # Initilalize Food[x, y]_t
    for (x, y) in food:
        KB.append(PropSymbolExpr(food_str, x, y, time=0))
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time=0))

    for t in range(50):
        paclocation = []
        for x, y in non_wall_coords:
            paclocation.append(PropSymbolExpr(pacman_str, x, y, time=t))
        KB.append(exactlyOne(paclocation))

        # Foods state at t
        foods_t = []
        for (x, y) in non_wall_coords:
            foods_t.append(PropSymbolExpr(food_str, x, y, time=t))

        model = findModel(conjoin(KB) & ~disjoin(foods_t))
        if (model):
            return extractActionSequence(model, actions)
        exactOneAction = []
        for action in DIRECTIONS:
            exactOneAction.append(PropSymbolExpr(action, time=t))
        KB.append(exactlyOne(exactOneAction))
        for (x, y) in non_wall_coords:
            KB.append(pacmanSuccessorAxiomSingle(x, y, t + 1, walls))
        # Add a food successor axiom
        for (x, y) in non_wall_coords:
            expr1 = ~PropSymbolExpr(pacman_str, x, y, time=t) & PropSymbolExpr(
                food_str, x, y, time=t)
            expr2 = PropSymbolExpr(food_str, x, y, time=t+1)
            KB.append(expr1 % expr2)

# Add pacphysics, action, and percept information to KB


def helper1(agent, KB, t: int, percept_rules: List[Tuple], all_coords: List[Tuple], non_outer_wall_coords: List[Tuple], walls_grid: List[List] = None, sensorModel: Any = None, successorAxioms: Any = None):
    KB.append(pacphysicsAxioms(t, all_coords, non_outer_wall_coords,
              walls_grid, sensorModel, successorAxioms))
    action = agent.actions[t]
    KB.append(PropSymbolExpr(action, time=t))
    KB.append(percept_rules)

# Find possible pacman locations with updated KB


def helper2(KB, t, non_outer_wall_coords: List[Tuple], possible_locations: List[Tuple]):
    for (x, y) in non_outer_wall_coords:
        pac_loc = PropSymbolExpr(pacman_str, x, y, time=t)
        model = findModel(conjoin(conjoin(KB), pac_loc))
        if (model):
            possible_locations.append((x, y))
        if (entails(conjoin(KB), pac_loc)):
            KB.append(pac_loc)
        elif (entails(conjoin(KB), ~pac_loc)):
            KB.append(~pac_loc)

# Find provable wall locations with updated KB


def helper3(KB, non_outer_wall_coords: List[Tuple], known_map: List[Tuple]):
    for (x, y) in non_outer_wall_coords:
        wall_loc = PropSymbolExpr(wall_str, x, y)
        if (entails(conjoin(KB), wall_loc)):
            KB.extend([wall_loc])
            known_map[x][y] = 1
        if (entails(conjoin(KB), ~wall_loc)):
            KB.extend([~wall_loc])
            known_map[x][y] = 0


def localization(problem, agent) -> Generator:
    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(
        range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(
        range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    KB = []
    possible_locations = []

    for (x, y) in all_coords:
        if (x, y) in walls_list:
            KB.append(PropSymbolExpr(wall_str, x, y))
        else:
            KB.append(~PropSymbolExpr(wall_str, x, y))

    for t in range(agent.num_timesteps):
        percepts = agent.getPercepts()
        percept_rules = fourBitPerceptRules(t, percepts)
        helper1(agent, KB, t, percept_rules, all_coords, non_outer_wall_coords,
                walls_grid, sensorAxioms, allLegalSuccessorAxioms)
        possible_locations = []
        helper2(KB, t, non_outer_wall_coords, possible_locations)
        agent.moveToNextState(agent.actions[t])
        yield possible_locations


def mapping(problem, agent) -> Generator:
    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(
        range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(
        range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight()+2)]
                 for x in range(problem.getWidth()+2)]

    # Pacman knows that the outer border of squares are all walls
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))
    KB.append(PropSymbolExpr(pacman_str, pac_x_0, pac_y_0, time=0))
    KB.append(~PropSymbolExpr(wall_str, pac_x_0, pac_y_0))
    known_map[pac_x_0][pac_y_0] = 0
    for t in range(agent.num_timesteps):
        percepts = agent.getPercepts()
        percept_rules = fourBitPerceptRules(t, percepts)
        helper1(agent, KB, t, percept_rules, all_coords, non_outer_wall_coords,
                known_map, sensorAxioms, allLegalSuccessorAxioms)
        helper3(KB, non_outer_wall_coords, known_map)
        agent.moveToNextState(agent.actions[t])
        yield known_map


def slam(problem, agent) -> Generator:
    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(
        range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(
        range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight()+2)]
                 for x in range(problem.getWidth()+2)]

    # We know that the outer_coords are all walls.
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    KB.append(PropSymbolExpr(pacman_str, pac_x_0, pac_y_0, time=0))
    KB.append(~PropSymbolExpr(wall_str, pac_x_0, pac_y_0))
    known_map[pac_x_0][pac_y_0] = 0

    for t in range(agent.num_timesteps):
        possible_locations = []
        percepts = agent.getPercepts()
        percept_rules = numAdjWallsPerceptRules(t, percepts)
        helper1(agent, KB, t, percept_rules, all_coords, non_outer_wall_coords,
                known_map, SLAMSensorAxioms, SLAMSuccessorAxioms)
        helper2(KB, t, non_outer_wall_coords, possible_locations)
        helper3(KB, non_outer_wall_coords, known_map)

        agent.moveToNextState(agent.actions[t])
        yield (known_map, possible_locations)


# Abbreviations
plp = positionLogicPlan
loc = localization
mp = mapping
flp = foodLogicPlan
...


def sensorAxioms(t: int, non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    ...


def fourBitPerceptRules(t: int, percepts: List) -> Expr:
    ...


def numAdjWallsPerceptRules(t: int, percepts: List) -> Expr:
    ...


def SLAMSensorAxioms(t: int, non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    ...


def allLegalSuccessorAxioms(t: int, walls_grid: List[List], non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    ...


def SLAMSuccessorAxioms(t: int, walls_grid: List[List], non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    ...


def modelToString(model: Dict[Expr, bool]) -> str:
    ...


def extractActionSequence(model: Dict[Expr, bool], actions: List) -> List:
    ...


def visualizeCoords(coords_list, problem) -> None:
    ...


def visualizeBoolArray(bool_arr, problem) -> None:
    ...


class PlanningProblem:
    ...
