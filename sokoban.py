import util
import os, sys
import datetime, time
import argparse
import signal, gc
import collections


class SokobanState:
    # player: 2-tuple representing player location (coordinates)
    # boxes: list of 2-tuples indicating box locations
    def __init__(self, player, boxes):
        # self.data stores the state
        self.data = tuple([player] + sorted(boxes))
        # below are cache variables to avoid duplicated computation
        self.all_adj_cache = None
        self.adj = {}
        self.dead = None
        self.solved = None
        self.visited = []

    def __str__(self):
        return 'player: ' + str(self.player()) + ' boxes: ' + str(self.boxes())

    def __eq__(self, other):
        return type(self) == type(other) and self.data == other.data

    def __lt__(self, other):
        return self.data < other.data

    def __hash__(self):
        return hash(self.data)

    # return player location
    def player(self):
        return self.data[0]

    # return boxes locations
    def boxes(self):
        return self.data[1:]

    def is_goal(self, problem):
        if self.solved is None:
            self.solved = all(problem.map[b[0]][b[1]].target for b in self.boxes())
        return self.solved

    def act(self, problem, act):
        if act in self.adj: return self.adj[act]
        else:
            val = problem.valid_move(self,act)
            self.adj[act] = val
            return val

    def deadp(self, problem):
        if self.dead is None:
            self.dead = False
            boxes = self.boxes()
            for box in boxes:
                if box not in problem.valid_box_pos:
                    self.dead = True
                    break

        return self.dead

    def all_adj(self, problem):
        if self.all_adj_cache is None:
            succ = []
            for move in 'udlr':
                valid, box_moved, nextS = self.act(problem, move)
                if valid:
                    succ.append((move, nextS, 1))
            self.all_adj_cache = succ
        return self.all_adj_cache


class MapTile:
    def __init__(self, wall=False, floor=False, target=False):
        self.wall = wall
        self.floor = floor
        self.target = target


def parse_move(move):
    if move == 'u': return (-1,0)
    elif move == 'd': return (1,0)
    elif move == 'l': return (0,-1)
    elif move == 'r': return (0,1)
    raise Exception('Invalid move character.')


class DrawObj:
    WALL = '\033[37;47m \033[0m'
    PLAYER = '\033[97;40m@\033[0m'
    BOX_OFF = '\033[30;101mX\033[0m'
    BOX_ON = '\033[30;102mX\033[0m'
    TARGET = '\033[97;40m*\033[0m'
    FLOOR = '\033[30;40m \033[0m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class SokobanProblem(util.SearchProblem):
    # valid sokoban characters
    valid_chars = 'T#@+$*. '

    def __init__(self, map, dead_detection=False, a2=False):
        self.map = [[]]
        self.dead_detection = dead_detection
        self.init_player = (0,0)
        self.init_boxes = []
        self.numboxes = 0
        self.targets = []
        self.visited = []
        self.valid_box_pos = []
        self.valid_player_pos = []
        self.valid_player_pos_in_current_state = []
        self.valid_box_pos_in_current_state = []
        self.valid_moves_per_location = {}
        self.parse_map(map)

        if self.dead_detection:
            # calculate all simple "not-dead" states
            for target in self.targets:
                self.simple_deadlock(target)
                self.visited.clear()
        self.valid_box_pos = sorted(self.valid_box_pos)
        self.find_valid_moves_per_location()

        # Need to find the valid positions the player can move
        self.valid_player_pos = self.valid_box_pos.copy()
        for pos in self.valid_box_pos:
            self.find_valid_player_pos(pos)
        self.valid_player_pos = sorted(self.valid_player_pos)

        # Valid moves a box can make at a certain location without going to a dead end
        # for key, value in self.valid_moves_per_location.items():
        #     print(f"For {key} the box can move: {value}")

    def find_valid_player_pos(self, pos):
        if pos not in self.valid_player_pos:
            self.valid_player_pos.append(pos)
        if (pos[0], pos[1]+1) not in self.valid_player_pos and self.map[pos[0]][pos[1]+1].floor:
            self.find_valid_player_pos((pos[0], pos[1]+1))
        if (pos[0], pos[1]-1) not in self.valid_player_pos and self.map[pos[0]][pos[1]-1].floor:
            self.find_valid_player_pos((pos[0], pos[1]-1))
        if (pos[0]+1, pos[1]) not in self.valid_player_pos and self.map[pos[0]+1][pos[1]].floor:
            self.find_valid_player_pos((pos[0]+1, pos[1]))
        if (pos[0]-1, pos[1]) not in self.valid_player_pos and self.map[pos[0]-1][pos[1]].floor:
            self.find_valid_player_pos((pos[0]-1, pos[1]))
        else:
            return
        return

    def find_valid_moves_per_location(self):
        for pos in self.valid_box_pos:
            self.valid_moves_per_location[str(pos)] = ""
            self.find_valid_moves_helper(pos[0], pos[1], pos)
        return

    def find_valid_moves_helper(self, curr_row, curr_col, pos):
        # check if position to the right is available
        if (curr_row+1, curr_col) in self.valid_box_pos and self.map[curr_row - 1][curr_col].floor:
            self.valid_moves_per_location[str(pos)] += 'd'
        # check if position to the left is available
        if (curr_row - 1, curr_col) in self.valid_box_pos and self.map[curr_row + 1][curr_col].floor:
            self.valid_moves_per_location[str(pos)] += 'u'
        # check if position up is available
        if (curr_row, curr_col - 1) in self.valid_box_pos and self.map[curr_row][curr_col + 1].floor:
            self.valid_moves_per_location[str(pos)] += 'l'
        # check if position down is available
        if (curr_row, curr_col + 1) in self.valid_box_pos and self.map[curr_row][curr_col - 1].floor:
            self.valid_moves_per_location[str(pos)] += 'r'
        return

    def can_pull_to(self, candidate, pos):
        if self.map[candidate[0]][candidate[1]].wall:
            return False

        if candidate[0] == pos[0] + 1:          # candidate is below the box
            if self.map[candidate[0]+1][candidate[1]].floor:
                return True
            else:
                return False
        elif candidate[0] == pos[0] - 1:        # candidate is above of box
            if self.map[candidate[0]-1][candidate[1]].floor:
                return True
            else:
                return False
        elif candidate[1] == pos[1] + 1:        # candidate is to the right of box
            if self.map[candidate[0]][candidate[1]+1].floor:
                return True
            else:
                return False
        elif candidate[1] == pos[1] - 1:        # candidate is to the left of box
            if self.map[candidate[0]][candidate[1]-1].floor:
                return True
            else:
                return False
    
    def simple_deadlock(self, pos):
        candidates = [(pos[0]+1, pos[1]), (pos[0]-1, pos[1]), (pos[0], pos[1]+1), (pos[0], pos[1]-1)]       # down, up, right, left
        
        for candidate in candidates:
            if candidate in self.visited:
                continue
            self.visited.append(candidate)
            if self.can_pull_to(candidate, pos):
                if candidate not in self.valid_box_pos:
                    self.valid_box_pos.append(candidate)
                self.simple_deadlock(candidate)

    # parse the input string into game map
    # Wall              #
    # Player            @
    # Player on target  +
    # Box               $
    # Box on target     *
    # Target            .
    # Floor             (space)
    def parse_map(self, input_str):
        coordinates = lambda: (len(self.map)-1, len(self.map[-1])-1)
        for c in input_str:
            if c == '#':
                self.map[-1].append(MapTile(wall=True))
            elif c == ' ':
                self.map[-1].append(MapTile(floor=True))
            elif c == '@':
                self.map[-1].append(MapTile(floor=True))
                self.init_player = coordinates()
            elif c == '+':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.init_player = coordinates()
                self.targets.append(coordinates())
                self.valid_box_pos.append(coordinates())
            elif c == '$':
                self.map[-1].append(MapTile(floor=True))
                self.init_boxes.append(coordinates())
            elif c == '*':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.init_boxes.append(coordinates())
                self.targets.append(coordinates())
                self.valid_box_pos.append(coordinates())
            elif c == '.':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.targets.append(coordinates())
                self.valid_box_pos.append(coordinates())
            elif c == '\n':
                self.map.append([])
        assert len(self.init_boxes) == len(self.targets), 'Number of boxes must match number of targets.'
        self.numboxes = len(self.init_boxes)

    def print_state(self, s):
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                target = self.map[row][col].target
                box = (row, col) in s.boxes()
                player = (row, col) == s.player()
                if box and target: print(DrawObj.BOX_ON, end='')
                elif player and target: print(DrawObj.PLAYER, end='')
                elif target: print(DrawObj.TARGET, end='')
                elif box: print(DrawObj.BOX_OFF, end='')
                elif player: print(DrawObj.PLAYER, end='')
                elif self.map[row][col].wall: print(DrawObj.WALL, end='')
                else: print(DrawObj.FLOOR, end='')
            print()

    # decide if a move is valid
    # return: (whether a move is valid, whether a box is moved, the next state)
    def valid_move(self, s, move, p=None):
        if p is None:
            p = s.player()
        dx,dy = parse_move(move)
        x1 = p[0] + dx
        y1 = p[1] + dy
        x2 = x1 + dx
        y2 = y1 + dy
        if self.map[x1][y1].wall:
            return False, False, None
        elif (x1,y1) in s.boxes():
            if self.map[x2][y2].floor and (x2,y2) not in s.boxes():
                return True, True, SokobanState((x1,y1),
                    [b if b != (x1,y1) else (x2,y2) for b in s.boxes()])
            else:
                return False, False, None
        else:
            return True, False, SokobanState((x1,y1), s.boxes())

    ##############################################################################
    # Problem 1: Dead end detection                                              #
    # Modify the function below. We are calling the deadp function for the state #
    # so the result can be cached in that state. Feel free to modify any part of #
    # the code or do something different from us.                                #
    # Our solution to this problem affects or adds approximately 50 lines of     #
    # code in the file in total. Your can vary substantially from this.          #
    ##############################################################################
    # detect dead end
    def dead_end(self, s):
        if not self.dead_detection:
            return False
        return s.deadp(self)

    def start(self):
        return SokobanState(self.init_player, self.init_boxes)

    def goalp(self, s):
        return s.is_goal(self)

    def expand(self, s):
        if self.dead_end(s):
            return []
        return s.all_adj(self)


class SokobanProblemFaster(SokobanProblem):
    ##############################################################################
    # Problem 2: Action compression                                              #
    # Redefine the expand function in the derived class so that it overrides the #
    # previous one. You may need to modify the solve_sokoban function as well to #
    # account for the change in the action sequence returned by the search       #
    # algorithm. Feel free to make any changes anywhere in the code.             #
    # Our solution to this problem affects or adds approximately 80 lines of     #
    # code in the file in total. Your can vary substantially from this.          #
    ##############################################################################
    def expand(self, s):
        list_new_states = []

        # Update the valid positions a player and box can move to
        self.update_valid_loc_player(s)
        self.update_valid_loc_box(s)

        for i, box_loc in enumerate(s.boxes()):
            # print(f"For {box_loc}")
            valid_moves_box_loc = list(self.valid_moves_per_location[str(box_loc)])
            for move in valid_moves_box_loc:
                player_dest_to_move_box, moved_box_pos, is_pos_valid = self.move_box(box_coordinates=box_loc, direction=move)
                if is_pos_valid:
                    action_by_player = self.find_shortest_path_to_location_move_player(
                        desired_location_player=player_dest_to_move_box,
                        start_location_player=s.player()
                    )
                    if action_by_player is not None:
                        action_by_player.append(box_loc)
                        action_by_player_directions = directions_of_path(action_by_player)
                        boxes_new_state = list(s.boxes()).copy()
                        boxes_new_state[i] = moved_box_pos
                        # box_loc ends up being final player location
                        new_state = (box_loc, tuple(boxes_new_state))
                        cost = 1                            # Can be changed later
                        list_new_states.append((action_by_player_directions, new_state, cost))

        # print(list_new_states)
        # for new_state in list_new_states:
        #     print(f"Action: {new_state[0]}")
        #     print(f"New State: {new_state[1]}")
        #     print(f"Cost: {new_state[2]}")

        return list_new_states

    def update_valid_loc_box(self, s):
        self.valid_box_pos_in_current_state = self.valid_box_pos.copy()
        # print(f"PRE {self.valid_box_pos_in_current_state}")
        for box_loc in s.boxes():
            if box_loc in self.valid_box_pos_in_current_state:
                self.valid_box_pos_in_current_state.remove(box_loc)
        # print(f"UPDATE {self.valid_box_pos_in_current_state}")
        return

    def update_valid_loc_player(self, s):
        self.valid_player_pos_in_current_state = self.valid_player_pos.copy()
        for box_loc in s.boxes():
            if box_loc in self.valid_player_pos_in_current_state:
                self.valid_player_pos_in_current_state.remove(box_loc)
        return

    def move_box(self, box_coordinates, direction):
        player_destination_coordinates_to_move_box = None
        desired_location_to_move = None
        is_position_available = False
        if direction in self.valid_moves_per_location[str(box_coordinates)]:
            if direction == 'r':
                player_destination_coordinates_to_move_box = (box_coordinates[0], box_coordinates[1] - 1)
                desired_location_to_move = (box_coordinates[0], box_coordinates[1] + 1)
                if desired_location_to_move in self.valid_box_pos_in_current_state:
                    is_position_available = True
            elif direction == 'l':
                player_destination_coordinates_to_move_box = (box_coordinates[0], box_coordinates[1] + 1)
                desired_location_to_move = (box_coordinates[0], box_coordinates[1] - 1)
                if desired_location_to_move in self.valid_box_pos_in_current_state:
                    is_position_available = True
            elif direction == 'd':
                player_destination_coordinates_to_move_box = (box_coordinates[0] - 1, box_coordinates[1])
                desired_location_to_move = (box_coordinates[0]+1, box_coordinates[1])
                if desired_location_to_move in self.valid_box_pos_in_current_state:
                    is_position_available = True
            elif direction == 'u':
                player_destination_coordinates_to_move_box = (box_coordinates[0] + 1, box_coordinates[1])
                desired_location_to_move = (box_coordinates[0]-1, box_coordinates[1])
                if desired_location_to_move in self.valid_box_pos_in_current_state:
                    is_position_available = True
        else:
            print("Can't move the box to this location")
            return None, None, False
        return player_destination_coordinates_to_move_box, desired_location_to_move, is_position_available

    def find_shortest_path_to_location_move_player(self, desired_location_player, start_location_player):
        # Use BFS to find the shortest path to a given position. The problem is that to move the player we need all
        # the positions the player can move to, basically all the ones inside the box
        queue = collections.deque([[start_location_player]])
        seen = {start_location_player}
        while queue:
            path = queue.popleft()
            x, y = path[-1]
            if (x, y) == desired_location_player:
                # print(f"Path: {path}")
                return path

            adj_loc = self.find_adj_loc((x, y))

            for x2, y2 in adj_loc:
                if (x2, y2) not in seen:
                    queue.append(path + [(x2, y2)])
                    seen.add((x2, y2))
        return None

    def find_adj_loc(self, loc):
        adj_loc = []
        if (loc[0]+1, loc[1]) in self.valid_player_pos_in_current_state:
            adj_loc.append((loc[0]+1, loc[1]))
        if (loc[0]-1, loc[1]) in self.valid_player_pos_in_current_state:
            adj_loc.append((loc[0]-1, loc[1]))
        if (loc[0], loc[1]+1) in self.valid_player_pos_in_current_state:
            adj_loc.append((loc[0], loc[1]+1))
        if (loc[0], loc[1]-1) in self.valid_player_pos_in_current_state:
            adj_loc.append((loc[0], loc[1]-1))
        return adj_loc


# Given a certain path of coordinates, returns the "rldu" directions
def directions_of_path(path):
    directions = []
    prev_position = None
    for pos in path:
        if prev_position is not None:
            diff_row, diff_col = pos[0] - prev_position[0], pos[1] - prev_position[1]
            if diff_row == -1:
                directions.append('u')
            elif diff_row == 1:
                directions.append('d')
            elif diff_col == -1:
                directions.append('l')
            elif diff_col == 1:
                directions.append('r')
        prev_position = pos
    return directions


class Heuristic:
    def __init__(self, problem):
        self.problem = problem

    ##############################################################################
    # Problem 3: Simple admissible heuristic                                     #
    # Implement a simple admissible heuristic function that can be computed      #
    # quickly based on Manhattan distance. Feel free to make any changes         #
    # anywhere in the code.                                                      #
    # Our solution to this problem affects or adds approximately 10 lines of     #
    # code in the file in total. Your can vary substantially from this.          #
    ##############################################################################
    def heuristic(self, s):
        raise NotImplementedError('Override me')

    ##############################################################################
    # Problem 4: Better heuristic.                                               #
    # Implement a better and possibly more complicated heuristic that need not   #
    # always be admissible, but improves the search on more complicated Sokoban  #
    # levels most of the time. Feel free to make any changes anywhere in the     #
    # code. Our heuristic does some significant work at problem initialization   #
    # and caches it.                                                             #
    # Our solution to this problem affects or adds approximately 40 lines of     #
    # code in the file in total. Your can vary substantially from this.          #
    ##############################################################################
    def heuristic2(self, s):
        raise NotImplementedError('Override me')


# solve sokoban map using specified algorithm
#  algorithm can be ucs a a2 fa fa2
def solve_sokoban(map, algorithm='ucs', dead_detection=False):
    # problem algorithm
    if 'f' in algorithm:
        problem = SokobanProblemFaster(map, dead_detection, '2' in algorithm)
    else:
        problem = SokobanProblem(map, dead_detection, '2' in algorithm)

    # search algorithm
    h = Heuristic(problem).heuristic2 if ('2' in algorithm) else Heuristic(problem).heuristic
    if 'a' in algorithm:
        search = util.AStarSearch(heuristic=h)
    else:
        search = util.UniformCostSearch()

    # solve problem
    search.solve(problem)
    if search.actions is not None:
        print('length {} soln is {}'.format(len(search.actions), search.actions))
    if 'f' in algorithm:
        raise NotImplementedError('Override me')    # TODO: Need to change
    else:
        return search.totalCost, search.actions, search.numStatesExplored


# let the user play the map
def play_map_interactively(map, dt=0.2):

    problem = SokobanProblem(map)
    state = problem.start()
    clear = 'cls' if os.name == 'nt' else 'clear'

    seq = ""
    i = 0
    visited=[state]

    os.system(clear)
    print()
    problem.print_state(state)

    while True:
        while i > len(seq)-1:
            try:
                seq += input('enter some actions (q to quit, digit d to undo d steps ): ')
            except EOFError:
                print()
                return

        os.system(clear)
        if seq != "":
            print(seq[:i] + DrawObj.UNDERLINE + seq[i] + DrawObj.END + seq[i+1:])
        problem.print_state(state)

        if seq[i] == 'q':
            return
        elif seq[i] in ['u','d','l','r']:
            time.sleep(dt)
            valid, _, new_state = problem.valid_move(state, seq[i])
            state = new_state if valid else state
            visited.append(state)
            os.system(clear)
            print(seq)
            problem.print_state(state)
            if not valid:
                print('Cannot move ' + seq[i] + ' in this state')
        elif seq[i].isdigit():
            i = max(-1, i - 1 - int(seq[i]))
            seq = seq[:i+1]
            visited = visited[:i+2]
            state = visited[i+1]
            os.system(clear)
            print(seq)
            problem.print_state(state)
            
        if state.is_goal(problem): 
            for _ in range(10): print('\033[30;101mWIN!!!!!\033[0m')
            time.sleep(5)
            return
        i = i + 1


# animate the sequence of actions in sokoban map
def animate_sokoban_solution(map, seq, dt=0.2):
    problem = SokobanProblem(map)
    state = problem.start()
    clear = 'cls' if os.name == 'nt' else 'clear'
    for i in range(len(seq)):
        os.system(clear)
        print(seq[:i] + DrawObj.UNDERLINE + seq[i] + DrawObj.END + seq[i+1:])
        problem.print_state(state)
        time.sleep(dt)
        valid, _, state = problem.valid_move(state, seq[i])
        if not valid:
            raise Exception('Cannot move ' + seq[i] + ' in state ' + str(state))
    os.system(clear)
    print(seq)
    problem.print_state(state)


# read level map from file, returns map represented as string
def read_map_from_file(file, level):
    map = ''
    start = False
    found = False
    with open(file, 'r') as f:
        for line in f:
            if line[0] == "'": continue
            if line.strip().lower()[:5] == 'level':
                if start: break
                if line.strip().lower() == 'level ' + level:
                    found = True
                    start = True
                    continue
            if start:
                if line[0] in SokobanProblem.valid_chars:
                    map += line
                else: break
    if not found:
        raise Exception('Level ' + level + ' not found')
    return map.strip('\n')


# extract all levels from file
def extract_levels(file):
    levels = []
    with open(file, 'r') as f:
        for line in f:
            if line.strip().lower()[:5] == 'level':
                levels += [line.strip().lower()[6:]]
    return levels


def extract_timeout(file, level):
    start = False
    found = False
    with open(file, 'r') as f:
        for line in f:
            if line[0] == "'": continue
            if line.strip().lower()[:5] == 'level':
                if start: break
                if line.strip().lower() == 'level ' + level:
                    found = True
                    continue
            if found:
                if line.strip().lower()[:7] == 'timeout':
                    return(int(line.strip().lower()[8:]))
                else: break
    if not found:
        raise Exception('Level ' + level + ' not found')
    return None


def solve_map(file, level, algorithm, dead, simulate):
    map = read_map_from_file(file, level)
    print(map)
    if dead: print('Dead end detection on for solution of level {level}'.format(**locals()))
    if algorithm == "me": 
        play_map_interactively(map)
    else:
        tic = datetime.datetime.now()
        cost, sol, nstates = solve_sokoban(map, algorithm, dead)
        toc = datetime.datetime.now()
        print('Time consumed: {:.3f} seconds using {} and exploring {} states'.format(
            (toc - tic).seconds + (toc - tic).microseconds/1e6, algorithm, nstates))
        seq = ''.join(sol)
        print(len(seq), 'moves')
        print(' '.join(seq[i:i+5] for i in range(0, len(seq), 5)))
        if simulate:
            animate_sokoban_solution(map, seq)
        return (toc - tic).seconds + (toc - tic).microseconds/1e6


def main():
    parser = argparse.ArgumentParser(description="Solve Sokoban map")
    parser.add_argument("level", help="Level name or 'all'")
    parser.add_argument("algorithm", help="me | ucs | [f][a[2]] | all")
    parser.add_argument("-d", "--dead", help="Turn on dead state detection (default off)", action="store_true")
    parser.add_argument("-s", "--simulate", help="Simulate the solution (default off)", action="store_true")
    parser.add_argument("-f", "--file", help="File name storing the levels (levels.txt default)", default='levels.txt')
    parser.add_argument("-t", "--timeout", help="Seconds to allow (default 300) (ignored if level specifies)", type=int, default=300)

    args = parser.parse_args()
    level = args.level
    algorithm = args.algorithm
    dead = args.dead
    simulate = args.simulate
    file = args.file
    maxSeconds = args.timeout

    if algorithm == 'all' and level == 'all':
        raise Exception('Cannot do all levels with all algorithms')

    def solve_now(): return solve_map(file, level, algorithm, dead, simulate)

    def solve_with_timeout(timeout):
        level_timeout = extract_timeout(file, level)
        if level_timeout is not None: timeout = level_timeout

        try:
            return util.TimeoutFunction(solve_now, timeout)()
        except KeyboardInterrupt:
            raise
        except MemoryError as e:
            signal.alarm(0)
            gc.collect()
            print('Memory limit exceeded.')
            return None
        except util.TimeoutFunctionException as e:
            signal.alarm(0)
            print('Time limit (%s seconds) exceeded.' % timeout)
            return None

    if level == 'all':
        levels = extract_levels(file)
        solved = 0
        time_used = 0
        for level in levels:
            print('Starting level {}'.format(level), file=sys.stderr)
            sys.stdout.flush()
            result = solve_with_timeout(maxSeconds)
            if result is not None:
                solved += 1
                time_used += result
        print(f'\n\nOVERALL RESULT: {solved} levels solved out of {len(levels)} ({100.0*solved/len(levels)})% using {time_used:.3f} seconds')
    elif algorithm == 'all':
        for algorithm in ['ucs', 'a', 'a2', 'f', 'fa', 'fa2']:
            print('Starting algorithm {}'.format(algorithm), file=sys.stderr)
            sys.stdout.flush()
            solve_with_timeout(maxSeconds)
    elif algorithm == 'me':
        solve_now()
    else:
        solve_with_timeout(maxSeconds)


if __name__ == '__main__':
    main()
