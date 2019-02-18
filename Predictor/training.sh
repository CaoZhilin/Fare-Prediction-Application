#!/usr/bin/env bash

PROJECT_ID="linen-option-222503"
BUCKET_ID="ml-fare-prediction-gs"
JOB_NAME="test_job_$(date +"%m%d_%H%M")"

JOB_DIR="gs://${BUCKET_ID}/"
TRAINING_PACKAGE_PATH="$(pwd)/mle_trainer"
MAIN_TRAINER_MODULE=mle_trainer.train
REGION=us-east1
RUNTIME_VERSION=1.10
PYTHON_VERSION=3.5
CONFIG_YAML=config.yaml

gcloud ml-engine jobs submit training "${JOB_NAME}" \
 --job-dir "${JOB_DIR}" \
 --package-path "${TRAINING_PACKAGE_PATH}" \
 --module-name "${MAIN_TRAINER_MODULE}" \
 --region "${REGION}" \
 --runtime-version="${RUNTIME_VERSION}" \
 --python-version="${PYTHON_VERSION}" \
 --config "${CONFIG_YAML}"