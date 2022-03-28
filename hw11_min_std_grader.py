import datetime
import os, re, csv
import argparse
import subprocess
from check_solution import check_solution
from sokoban import extract_levels

temp_file_name = "000min_grader_temp.txt"

def retrieve_sol(filename):
    sol = ''
    with open(filename, 'r') as f:
        for line in f:
            line = line.lower()
            if re.match(r'^[udlr ]+$', line):
                sol = [c for c in line if c in 'udlr']
                break
    return sol

# Assume test.txt is provided with testing levels
def grader_helper(level, mode_1, mode_2, solution_code, level_file):
    failure = False
    time1 = None
    time2 = None
    f = open(temp_file_name, "w")     # redirect output into a file to validate output
    command = "python3 %s %s %s -f %s -t 180" % (solution_code, level, mode_1, level_file)
    # test time for t1
    t1_start = datetime.datetime.now()
    print(f'executing {command}')
    subprocess.call(command, stdout=f, shell=True)
    t1_end = datetime.datetime.now()
    f.close()
    # ensure the output sequence is valid with check_solution.py
    sol = retrieve_sol(temp_file_name)

    # mode 1 is slow mode. it's ok to timeout in mode 1
    if sol != '' and not check_solution(level, level_file, sol):
        failure = True
        
    if not failure:
        f = open(temp_file_name, "w")
        command = "python3 %s %s %s -f %s -t 60" % (solution_code, level, mode_2, level_file)
        # test time for t2
        t2_start = datetime.datetime.now()
        print(f'executing {command}')
        subprocess.call(command, stdout=f, shell=True)
        t2_end = datetime.datetime.now()
        f.close()

        # ensure the output sequence is valid
        sol = retrieve_sol(temp_file_name)
        if not check_solution(level, level_file, sol):
            failure = True
        else:
            time1 = t1_end - t1_start
            time2 = t2_end - t2_start

    if time1 != None: time1 = time1.seconds
    if time2 != None: time2 = time2.seconds
    return (not failure, time1, time2)

# for levels with names beginning with "p1", (or "p5" and ending in "1")
#
# compare performance with "ucs" and "ucs -d", require 0.7 time reduction factor 
# (or timeout on first) 
# Likewise for other problems
config = {"p1": (0.7, ("ucs","ucs -d")), "p2": (0.7, ("ucs -d","f -d")), "p3" : (0.7, ("f -d","fa -d"))}
# no longer included: "p4": (0.7, ("fa -d", "fa2 -d"))

def main():
    solution_code_default = 'sokoban.py'
    level_file_default = 'design.txt'

    parser = argparse.ArgumentParser(description="Solve Sokoban map")
    parser.add_argument("-l", "--levels", help="File name storing the levels", default=level_file_default)
    parser.add_argument("-s", "--source", help="Source code to run against the levels (default sokoban.py)", default=solution_code_default)

    args = parser.parse_args()
    solution_code = args.source
    level_file = args.levels

    levels = extract_levels(level_file)

    score = 0
    results = {}
    for level in levels:
        type = level[0:2]
        if type=="p5": type = "p"+level[-1]
        if type not in config: break
        (ratio, (mode1, mode2)) = config[type]
        
        print(f'Comparing level {level} for {mode1} vs {mode2}')
        (success, time1, time2) = grader_helper(level, mode1, mode2, solution_code, level_file)
        results[level] = (success, time1, time2)
        print(f'value was {results[level]}')
        if success and ((time1==0 and time2>5) or (time1>0 and ((time2 / time1) < ratio))): score += 1
        print(f'-->Result was {(success, time1, time2)}\n')
        
    print(f'passed {score} levels of {len(levels)} with details {results}')


if __name__ == "__main__":
    main()

    
