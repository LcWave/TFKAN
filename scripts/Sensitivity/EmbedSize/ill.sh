
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Sensitivity/EmbedSize" ]; then
    mkdir ./logs/Sensitivity/EmbedSize
fi
seq_len=96
model_name=FreLinear
embed_sizes=(32 64 256 512)
for embed_size in "${embed_sizes[@]}"
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path national_illness.csv \
    --model_id ill'_'$embed_size'_'$seq_len'_'24 \
    --model $model_name \
    --data ill \
    --features M \
    --seq_len $seq_len \
  --embed_size $embed_size \
    --pred_len 24 \
    --enc_in 7 \
    --itr 1 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --lradj type1 \
    --channel_independence 0 \
    --weight_decay 0.0001 \
    --use_bias  >logs/Sensitivity/EmbedSize/$model_name'_'ill'_'$embed_size'_'$seq_len'_'24.log

  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path national_illness.csv \
    --model_id ill'_'$embed_size'_'$seq_len'_'36 \
    --model $model_name \
    --data ill \
    --features M \
    --seq_len $seq_len \
  --embed_size $embed_size \
    --pred_len 36 \
    --enc_in 7 \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --lradj type1 \
    --channel_independence 0 \
    --weight_decay 0.0001 \
    --use_bias  >logs/Sensitivity/EmbedSize/$model_name'_'ill'_'$embed_size'_'$seq_len'_'36.log

  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path national_illness.csv \
    --model_id ill'_'$embed_size'_'$seq_len'_'48 \
    --model $model_name \
    --data ill \
    --features M \
    --seq_len $seq_len \
  --embed_size $embed_size \
    --pred_len 48 \
    --enc_in 7 \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --lradj type1 \
    --channel_independence 0 \
    --weight_decay 0.0001 \
    --use_bias  >logs/Sensitivity/EmbedSize/$model_name'_'ill'_'$embed_size'_'$seq_len'_'48.log

  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path national_illness.csv \
    --model_id ill'_'$embed_size'_'$seq_len'_'60 \
    --model $model_name \
    --data ill \
    --features M \
    --seq_len $seq_len \
  --embed_size $embed_size \
    --pred_len 60 \
    --enc_in 7 \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --lradj type1 \
    --channel_independence 0 \
    --weight_decay 0.0001 \
    --use_bias  >logs/Sensitivity/EmbedSize/$model_name'_'ill'_'$embed_size'_'$seq_len'_'60.log
done