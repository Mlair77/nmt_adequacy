data_dir=../data/en_fr_data/onmt_data
signature=nmt_lm_0.01
work_dir=$your_code_dir
output_dir=$your_output_dir/enfr/$signature/

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,

if [ ! -d $output_dir ]; then
    mkdir $output_dir
    chmod 777 $output_dir -R
fi

python  $work_dir/train.py -data $data_dir/wmt14enfr -save_model $output_dir/$signature \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer_lm -position_encoding \
        -train_steps 300000  -max_generator_batches 0 -dropout 0.1 \
        -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 1 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps 5000 -save_checkpoint_steps 5000 \
        -world_size 8 -gpu_ranks 0 1 2 3 4 5 6 7 -keep_checkpoint 60 -report_every 1000 \
        -train_lm -lambda_lm 0.01 -report_lm