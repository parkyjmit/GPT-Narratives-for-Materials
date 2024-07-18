#!/bin/sh


device='cuda:1'


for llm in 'allenai/scibert_scivocab_cased' 'm3rg-iitd/matscibert'; do

    label='text'
    evaluation='zero-shot-retrieval'
    for batch_size in 128; do
        python -u baseline_compute_metrics_text_encoder.py  --llm $llm --label $label --device $device --batch-size $batch_size --evaluation-method $evaluation
    done
done