# Sokoban Maze Solver
Solve sokoban maze by implementing state-space search according to Uniform Cost Search and A* Search algorithms with increasingly refined heuristics and action compression methods.

The logic for solver is found in 'sokoban.py' and 'util.py' while 'levels.txt' and 'design.txt' contain sample mazes that can be inputted into the program for solving.

Map Legend:
'@' = player
'$' = box
'#' = wall
'.' = target
'+' = player standing on target
