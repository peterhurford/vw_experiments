from vowpal_platypus import run
from vowpal_platypus.models import als
from vowpal_platypus.utils import safe_remove
from vowpal_platypus.evaluation import rmse
import argparse
import os
from datetime import datetime

start = datetime.now()
print('...Starting at ' + str(start))

print("Setting up...")
start = datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument('--cores')
parser.add_argument('--num_ratings')
parser.add_argument('--hypersearch', action='store_true', default=False)
cores = int(parser.parse_args().cores)
num_ratings = parser.parse_args().num_ratings
hypersearch = parser.parse_args().hypersearch
if num_ratings is None:
    num_ratings = 1000000
else:
    num_ratings = int(num_ratings)

def compile_rating(item):
    item = item.split('::')
    rating = item[2]
    user_id = item[0]
    movie_id = item[1]
    return {'label': rating, 'c': user_id, 'p': movie_id}

if num_ratings < 1000000:
    os.system('head -n ' + str(num_ratings) + ' als/1m/data/ratings.dat > als/1m/data/ratings_.dat')
else:
    os.system('cp als/1m/data/ratings.dat als/1m/data/ratings_.dat')

if hypersearch:
    model = als(name='ALS', passes=[5, 10], cores=cores,
                quadratic='cp',
                rank=[5, 10],
                nn=5,
                learning_rate=0.015,
                decay_learning_rate=0.97,
                power_t=0)
    evaluate_function = rmse
else:
    model = als(name='ALS', passes=10, cores=cores,
                quadratic='cp', rank=10, nn=5,
                l2=0.01, learning_rate=0.015, decay_learning_rate=0.97, power_t=0)
    evaluate_function = None

results = run(model,
              'als/1m/data/ratings_.dat',
              line_function=compile_rating,
              evaluate_function=rmse,
              header=False)
safe_remove('als/1m/data/ratings_.dat')

import pdb
pdb.set_trace()
rmse = 'RMSE: ' + str(rmse(results))
end = datetime.now()
time = 'Time: ' + str((end - start).total_seconds()) + ' sec'
speed = 'Speed: ' + str((end - start).total_seconds() * 1000000 / float(num_ratings)) + ' mcs/row'
title = 'MOVILELENS IN VP (HYPERSEARCH)' if hypersearch else 'MOVIELENS IN VP'
with open('test_results.txt', 'a') as test_file:
    for line in ['\n', title + '\n', str(datetime.now()) + '\n', rmse + '\n', time + '\n', speed + '\n']:
        test_file.write(line)
print(rmse)
print(time)
print(speed)
