WEIGHT_PATH="path for your model weight"
MODEL_TYPE="glm-10b"
MODEL_ARGS="--task-mask True\
            --block-mask-prob 0.0 \
            --vocab-size 50048 \
            --num-layers 48 \
            --hidden-size 4096 \
            --num-attention-heads 64 \
            --max-sequence-length 1025 \
            --tokenizer-type glm_ChineseSPTokenizer \
            --tokenizer-model-type $MODEL_TYPE \
            --load $WEIGHT_PATH"
