#!/bin/bash

./run-docker.sh --run 04_ranknet+scalar_500_encoder --model ranknet_aux --station-id 29 --aux-model concat --aux-file aux.csv

./run-docker.sh --run 04_ranknet+scalar_500_none --model ranknet_aux --station-id 29

