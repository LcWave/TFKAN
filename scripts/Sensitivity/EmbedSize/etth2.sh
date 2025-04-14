
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
  --data_path ETTh2.csv \
  --model_id ETTh2'_'$embed_size'_'$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --embed_size $embed_size \
  --pred_len 96 \
  --enc_in 7 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0.0001 \
  --use_bias  >logs/Sensitivity/EmbedSize/$model_name'_'ETTh2'_'$embed_size'_'$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2'_'$embed_size'_'$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --embed_size $embed_size \
  --pred_len 192 \
  --enc_in 7 \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0.0001 \
  --use_bias  >logs/Sensitivity/EmbedSize/$model_name'_'ETTh2'_'$embed_size'_'$seq_len'_'192.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2'_'$embed_size'_'$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --embed_size $embed_size \
  --pred_len 336 \
  --enc_in 7 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0.001 \
  --use_bias  >logs/Sensitivity/EmbedSize/$model_name'_'ETTh2'_'$embed_size'_'$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2'_'$embed_size'_'$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --embed_size $embed_size \
  --pred_len 720 \
  --enc_in 7 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0006 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0.0008  >logs/Sensitivity/EmbedSize/$model_name'_'ETTh2'_'$embed_size'_'$seq_len'_'720.log
done