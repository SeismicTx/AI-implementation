process PREPROCESS {
    container 'preprocessing:latest'

    publishDir "${params.output_dir}/processed", mode: 'copy'

    input:
        path input_dir

    output:
        tuple val('merged'), path('merged_antibody_data.csv')

    script:
    """
    python /app/preprocess.py --data_dir ${params.input_dir}
    """
}