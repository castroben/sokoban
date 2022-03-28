# check a sokoban problem solution against a named level in the levels file
# used by the min std grader
import sokoban
import argparse

def checker(map, seq):
    problem = sokoban.SokobanProblem(map, False)
    state = problem.start()
    for move in seq:
        move = move.lower()
        if move not in 'udlr':
            continue
        valid, _, state = problem.valid_move(state, move)
        if not valid:
            return False
    return problem.goalp(state)

def check_solution(level, filename, seq):
    map = sokoban.read_map_from_file(filename, str(level))
    return checker(map, seq)

def main():
    parser = argparse.ArgumentParser(description="Check Sokoban solution")
    parser.add_argument("level", help="Level name")
    parser.add_argument("solution", help="Solution sequence (string of u/d/l/r)")
    parser.add_argument("-f", "--file", help="File name storing the levels (levels.txt default)", default='levels.txt')

    args = parser.parse_args()
    level = args.level
    seq = args.solution
    file = args.file

    map = sokoban.read_map_from_file(file, level)
    print(checker(map, seq))

if __name__ == '__main__':
    main()
