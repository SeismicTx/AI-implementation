#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Import modules
include { PREPROCESS } from './modules/preprocess'
include { TRAIN } from './modules/train'

// Main workflow
workflow {
    input_dir = Channel.fromPath(params.input_dir, type: 'dir')

    // Execute pipeline
    PREPROCESS(input_dir)
    TRAIN(PREPROCESS.out)
}