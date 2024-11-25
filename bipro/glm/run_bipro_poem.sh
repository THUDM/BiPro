#!/bin/bash
source $1
MPSIZE=1
MAXSEQLEN=512
MASTER_PORT=0

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=40
TOPP=0


python bipro_poem.py \
       --mode inference \
       --model-parallel-size $MPSIZE \
       $MODEL_ARGS \
       --num-beams 6 \
       --no-repeat-ngram-size 0 \
       --length-penalty 0.7 \
       --fp16 \
       --out-seq-length $MAXSEQLEN \
       --temperature $TEMP \
       --top_k $TOPK \
       --output-path samples_glm \
       --batch-size 6 \
       --out-seq-length 800 \
       --input-source interactive \
       --sampling-strategy iPromptSearchStrategy \
       --device $2 \
    
