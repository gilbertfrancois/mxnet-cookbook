#!/usr/bin/env bash

DATA_FOLDER=../data/minc-2500
IM2REC=im2rec.py

python {IM2REC} ${DATA_FOLDER}/train_rec ${DATA_FOLDER}/train/ --recursive --list --num-thread 8
python {IM2REC} ${DATA_FOLDER}/train_rec ${DATA_FOLDER}/train/ --recursive --pass-through --pack-label --num-thread 8

python {IM2REC} ${DATA_FOLDER}/val_rec ${DATA_FOLDER}/val/ --recursive --list --num-thread 8
python {IM2REC} ${DATA_FOLDER}/val_rec ${DATA_FOLDER}/val/ --recursive --pass-through --pack-label --num-thread 8

python {IM2REC} ${DATA_FOLDER}/test_rec ${DATA_FOLDER}/test/ --recursive --list --num-thread 8
python {IM2REC} ${DATA_FOLDER}/test_rec ${DATA_FOLDER}/test/ --recursive --pass-through --pack-label --num-thread 8
