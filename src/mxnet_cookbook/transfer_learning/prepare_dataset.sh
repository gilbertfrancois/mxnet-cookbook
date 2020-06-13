#!/usr/bin/env bash

DATA_FOLDER=../../../data/minc-2500
INTERPRETER=`which python`
IM2REC=${INTERPRETER}/../lib/python3.7/site-packages/mxnet/tools/im2rec.py

python im2rec.py ${DATA_FOLDER}/train_rec ${DATA_FOLDER}/train/ --recursive --list --num-thread 8
python im2rec.py ${DATA_FOLDER}/train_rec ${DATA_FOLDER}/train/ --recursive --pass-through --pack-label --num-thread 8

python im2rec.py ${DATA_FOLDER}/val_rec ${DATA_FOLDER}/val/ --recursive --list --num-thread 8
python im2rec.py ${DATA_FOLDER}/val_rec ${DATA_FOLDER}/val/ --recursive --pass-through --pack-label --num-thread 8

python im2rec.py ${DATA_FOLDER}/test_rec ${DATA_FOLDER}/test/ --recursive --list --num-thread 8
python im2rec.py ${DATA_FOLDER}/test_rec ${DATA_FOLDER}/test/ --recursive --pass-through --pack-label --num-thread 8
