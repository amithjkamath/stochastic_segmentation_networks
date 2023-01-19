#!/bin/bash

#python ./custom_preprocessing-MV-brainstem.py --input-dir "/home/akamath/data/insel/ground_truth_oar/" --output-dir "/home/akamath/Desktop/ssn_data"

JOB_DIR="/home/akamath/Desktop/ssn_job"
TEST_CSV_PATH="/home/akamath/Desktop/ssn_data/assets/data_index_test.csv"
CONFIG_FILE="./etc/stochastic-deepmedic.json"
SAVED_MODEL_PATHS="./assets/saved_models/brainstem_MV.torch_model"

python ./ssn/inference.py --job-dir $JOB_DIR --test-csv-path $TEST_CSV_PATH --config-file $CONFIG_FILE --device "0" --saved-model-paths $SAVED_MODEL_PATHS --overwrite "True"