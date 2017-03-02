#!/bin/bash
cd ~/vp_examples/criteo
mkdir data; cd data
wget https://s3-eu-west-1.amazonaws.com/criteo-labs/dac.tar.gz  # Accept terms at http://labs.criteo.com/downloads/2014-kaggle-display-advertising-challenge-dataset/
tar -xvzf dac.tar.gz
