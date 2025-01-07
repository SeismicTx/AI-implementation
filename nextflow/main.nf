#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Import modules
include { PREPROCESS } from './modules/preprocess'
include { TRAIN } from './modules/train'
include { EVALUATE } from './modules/evaluate'
include { ACTIVE_LEARNING } from './modules/active_learning'

// Pipeline parameters
params.data_path = "${params.input_bucket}/*.csv"
params.model_version = '1.0.0'

// Main workflow
workflow {
    // Channel for input data
    input_channel = Channel
        .fromPath(params.data_path)
        .map { file -> tuple(file.baseName, file) }

    // Execute pipeline
    PREPROCESS(input_channel)
    TRAIN(PREPROCESS.out)
    EVALUATE(TRAIN.out)
    ACTIVE_LEARNING(EVALUATE.out)
}

// modules/preprocess.nf
process PREPROCESS {
    container 'preprocessing:latest'

    input:
        tuple val(id), path(input_file)

    output:
        tuple val(id), path("${id}_processed.csv")

    script:
    """
    python /app/preprocess.py \
        --input $input_file \
        --output ${id}_processed.csv
    """
}

// modules/train.nf
process TRAIN {
    container 'training:latest'

    input:
        tuple val(id), path(processed_data)

    output:
        tuple val(id), path("${id}_model.pkl"), path("${id}_metrics.json")

    script:
    """
    python /app/train.py \
        --input $processed_data \
        --output ${id}_model.pkl \
        --metrics ${id}_metrics.json
    """
}