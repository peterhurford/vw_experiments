#!/usr/bin/env python

# Example usage: python runner.py --train_cores 40 --predict_cores 20 --num_ratings 20000000

from vowpal_platypus import run, als, safe_remove, split_file, daemon, daemon_predict
from datetime import datetime
from multiprocessing import Pool
import os
import argparse
import random
import socket
from retrying import retry

def train_on_core(model):
    core = 0 if model.node is None else model.node
    user_id_pool = filter(lambda x: int(x) % train_cores == core, user_ids)
    with model.training():
        for user_id in user_id_pool:
            for movie_id, rating in ratings[user_id].iteritems():
                vw_item = rating + ' |u ' + user_id + ' |i ' + movie_id
                vw.push_instance(vw_item)
    return None


def compile_ratings(ratings_file):
    ratings = {}
    while True:
        item = ratings_file.readline()
        if not item:
            break
        item = item.split(',')
        rating = item[2]
        user_id = item[0]
        movie_id = item[1]
        if ratings.get(user_id) is None:
            ratings[user_id] = {} 
        ratings[user_id][movie_id] = rating
    return ratings

def rec_for_user(daemon):
    core = 0 if daemon.node is None else daemon.node
    rfile = rec_files[core]
    user_id_pool = filter(lambda x: int(x) % predict_cores == core, user_ids)
    for user_id in user_id_pool:
        unseen_movie_ids = list(set(movie_ids) - set(ratings[user_id].values()))
        vw_items = ''.join(map(lambda m: '|u ' + user_id + ' |i ' + m + '\n', unseen_movie_ids))
        print('Connecting to port %i...' % port)
        preds = daemon_predict(daemon, vw_items, quiet=True)
        user_recs = [list(a) for a in zip(preds, unseen_movie_ids)]
        user_recs.sort(reverse=True)
        rfile.write(str({'user': user_id,
                        'products': map(lambda x: x[1], user_recs[:10])}) + '\n')
    rfile.flush()
    return None

def evaluate_on_core(core):
    port = 4040 + core
    user_id_pool = filter(lambda x: int(x) % predict_cores == core, user_ids)
    all_preds = []
    for user_id in user_id_pool:
        vw_items = ''
        user_ratings = []
        if ratings.get(user_id) is not None:
            for movie_id, rating in ratings[user_id].iteritems():
                vw_items += '|u ' + user_id + ' |i ' + movie_id + '\n'
                user_ratings.append(float(rating))
            print('Connecting to port %i...' % port)
            preds = netcat('localhost', port, vw_items)
            all_preds.append(zip(preds, user_ratings))
    all_preds = sum(all_preds, [])
    return sum(map(lambda x: (float(x[0]) - float(x[1])) ** 2, all_preds)) / len(all_preds)


print("Setting up...")
start = datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument('--volume')
parser.add_argument('--op_sys')
parser.add_argument('--train_cores')
parser.add_argument('--predict_cores')
parser.add_argument('--num_ratings')
parser.add_argument('--evaluate')
parser.add_argument('--evaluate_only', action='store_true', default=False)
volume = parser.parse_args().volume
op_sys = parser.parse_args().op_sys
if volume is None:
    volume = os.getcwd()
volume = volume + '/'
if op_sys is None:
    op_sys = 'ubuntu'
train_cores = int(parser.parse_args().train_cores)
predict_cores = int(parser.parse_args().predict_cores)
num_ratings = int(parser.parse_args().num_ratings)
evaluate = parser.parse_args().evaluate
evaluate_only = parser.parse_args().evaluate_only
if evaluate_only and (evaluate is None or evaluate is False):
    evaluate = "ib"

print("Cleaning up...")
targets = ['ALS*', 'ratings_*', '*recs*', 'users.csv']
[safe_remove('rm ' + volume + target) for target in targets]

print("Formating data...")
os.system("head -n {} {}ratings.csv | tail -n +2 > {}ratings_.csv".format(num_ratings + 1, volume, volume)) # +1 to not trim header
os.system("tail -n +2 " + volume + "ratings_.csv | awk -F\",\" '{print $1}' | uniq > " + volume + "users.csv")

def process_csv(item):
   return item.split(',')[0] 
movie_ids = load_file('{}movies.csv'.format(volume), process_csv)
user_ids = load_file('{}user.csv'.format(volume), process_csv)
movie_ids.pop(0) # Throw out headers
user_ids.pop(0)

ratings_file = open('{}ratings_.csv'.format(volume), 'r')
ratings = compile_ratings(ratings_file)
setup_done = datetime.now()

if not evaluate_only:
    print("Jamming some train on {} cores...".format(train_cores))
    vw_models = als(name='{}ALS'.format(volume),
                    cores=train_cores,
                    passes=40,
                    quadratic='ui',
                    rank=25,
                    l2=0.001,
                    learning_rate=0.015,
                    decay_learning_rate=0.97,
                    power_t=0)
    run(vw_models, train_on_core)

    print("Spooling predictions on {} cores...".format(predict_cores))
    daemons = [daemon(model) for model in vw_models]
    rec_files = [open('{}/py_recs'.format(volume) + str(i) + '.dat', 'w') for i in range(predict_cores)]
    run(daemons, rec_for_user)

    for f in rec_files:
        f.close()
    os.system('cat {}/py_recs* > {}/all_py_recs.csv'.format(volume, volume))
    recs_done = datetime.now()

if evaluate:
    print('Evaluating...')
    if evaluate == 'ib':
        print('Shuffling for ib evaluate...')
        if op_sys == 'mac':
            shuf = 'gshuf'
            split = 'gsplit'
        else:
            shuf = 'shuf'
            split = 'split'
        os.system("{} ratings_.csv > ratings__.csv; mv ratings__.csv ratings_.csv".format(shuf))
    os.system("{} -d -l {} ratings_.csv".format(split, int(num_ratings * 0.9)))
    os.system("mv x00 ratings_train.csv; mv x01 ratings_test.csv")
    ratings_file = open('ratings_train.csv', 'r')
    ratings = compile_ratings(ratings_file)
    if train_cores > 1:
        pool = Pool(train_cores)
        pool.map(train_on_core, range(train_cores))
    else:
        train_on_core(0)

    ratings = {}
    ratings_file = open('ratings_test.csv', 'r')
    ratings = compile_ratings(ratings_file)

    if predict_cores > 1:
        daemons = [daemon(core) for core in random.sample(range(predict_cores), predict_cores)]
        pool = Pool(predict_cores)
        rmses = pool.map(evaluate_on_core, range(predict_cores))
        rmse = (sum(rmses) / predict_cores) ** 0.5
    else:
        daemons = [daemon(0)]
        rmse = evaluate_on_core(0)
    print("RMSE: " + str(rmse))
    evaluate_done = datetime.now()

print("Spinning down server...")
if train_cores > 1:
    os.system("killall spanning_tree")
for port in range(4040, 4040 + predict_cores):
    print("Spinning down port %i" % port)
    os.system("pkill -9 -f 'vw.*--port %i'" % port)

print("Timing...")
print("Set up in " + str(setup_done - start))
if not evaluate_only:
    print("Training in " + str(training_done - setup_done))
    print("Reccing in " + str(recs_done - training_done))
    if evaluate:
        print("Evaluating in: " + str(evaluate_done - recs_done))
        print("Total (without evaluate): " + str(recs_done - start))
        print("Total: " + str(evaluate_done - start))
    else:
        print("Total: " + str(recs_done - start))
else:
    print("Evaluating in: " + str(evaluate_done - setup_done))
    print("Total: " + str(evaluate_done - start))
