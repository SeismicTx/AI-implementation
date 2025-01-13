process TRAIN {
    container 'training:latest'

    publishDir "${params.output_dir}/models", mode: 'copy'

    input:
        tuple val(id), path(merged_data)

    output:
        tuple val(id), path("best_model.pth")

    script:
    """
    python /app/train.py --input $ml_data
    """
}