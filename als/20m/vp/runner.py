from vowpal_platypus import run_parallel
from vowpal_platypus.models import als
from vowpal_platypus.daemon import daemon, daemon_predict

import argparse
import os
import time
from datetime import datetime

start = datetime.now()
print('...Starting at ' + str(start))

start = datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument('--cores')
parser.add_argument('--machines')
parser.add_argument('--machine_number')
parser.add_argument('--master_ip')
cores = int(parser.parse_args().cores)
machines = int(parser.parse_args().machines)
machine_number = int(parser.parse_args().machine_number)
master_ip = parser.parse_args().master_ip

print("Formating data...")
os.system("awk -F\",\" '{print $1}' ../data/ratings.csv | uniq > ../data/users.csv")

ratingsfile = open('../data/ratings.csv', 'r')
movie_file = open('../data/movies.csv', 'r')
user_file = open('../data/users.csv', 'r')
movie_ids = [movie.split(',')[0] for movie in list(movie_file.read().splitlines())]
user_ids = [user.split(',')[0] for user in list(user_file.read().splitlines())]
movie_ids.pop(0) # Throw out headers
user_ids.pop(0)

def compile_ratings(ratingsfile):
    ratings = {}
    while True:
        item = ratingsfile.readline()
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

ratings = compile_ratings(ratingsfile)

movie_file.close()
user_file.close()
ratingsfile.close()

model = als(name='ALS', passes=10,
            cores=cores, machines=machines, machine_number=machine_number, master_ip=master_ip,
            quadratic='ui', rank=10,
            l2=0.01, learning_rate=0.015, decay_learning_rate=0.97, power_t=0)

def rec_for_model(model):
    core = model.params.get('node', 0)
    user_id_pool = filter(lambda x: int(x) % (cores * machines) == core, user_ids)
    num_lines = len(user_id_pool)
    with model.training():
        i = 0
        curr_done = 0
        for user_id in user_id_pool:
            i += 1
            done = int(i / float(num_lines) * 100)
            if done - curr_done > 4:
                print 'Train - Core {} @ {}: done {}%'.format(core, str(datetime.now()), done)
                curr_done = done
            for movie_id, rating in ratings[user_id].iteritems():
                model.push_instance({'label': float(rating), 'u': user_id, 'i': movie_id})
    model = daemon(model)
    print 'Waiting for available ports...'
    time.sleep(20)
    with open('recs' + str(core) + '.txt', 'w') as rfile:
        i = 0
        curr_done = 0
        for user_id in user_id_pool:
            i += 1
            done = int(i / float(num_lines) * 100)
            if done - curr_done >= 1:
                print 'Predict - Core {} @ {}: done {}%'.format(core, str(datetime.now()), done)
                curr_done = done
            unseen_movie_ids = list(set(movie_ids) - set(ratings[user_id].values()))
            vw_items = map(lambda m: {'u': user_id, 'i': m}, unseen_movie_ids)
            preds = daemon_predict(model, vw_items, quiet=True)
            user_recs = [list(a) for a in zip(preds, unseen_movie_ids)]
            user_recs.sort(reverse=True)
            rfile.write(str({'user': user_id,
                            'products': map(lambda x: x[1], user_recs[:10])}) + '\n')
    return None

run_parallel(model, rec_for_model)
end = datetime.now()
print('Time: ' + str((end - start).total_seconds()) + ' sec')
