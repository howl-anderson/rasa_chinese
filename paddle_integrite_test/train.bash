#!/usr/bin/env bash

python -m rasa_nlu.train \
    --config config.yaml \
    --data data.json \
    --path projects