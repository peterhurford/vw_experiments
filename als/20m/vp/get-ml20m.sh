#!/bin/bash
cd ..
wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
sudo apt-get install unzip
unzip ml-20m.zip
mv ml-20m data
rm ml-20m.zip
cd vp
