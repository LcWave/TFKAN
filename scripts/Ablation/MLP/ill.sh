
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=MLPLinear

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id ill_$seq_len'_'24 \
  --model $model_name \
  --data ill \
  --features M \
  --seq_len $seq_len \
  --pred_len 24 \
  --enc_in 7 \
  --itr 1 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0.0001 \
  --use_bias  >logs/LongForecasting/$model_name'_'ill_$seq_len'_'24.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id ill_$seq_len'_'36 \
  --model $model_name \
  --data ill \
  --features M \
  --seq_len $seq_len \
  --pred_len 36 \
  --enc_in 7 \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0.0001 \
  --use_bias  >logs/LongForecasting/$model_name'_'ill_$seq_len'_'36.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id ill_$seq_len'_'48 \
  --model $model_name \
  --data ill \
  --features M \
  --seq_len $seq_len \
  --pred_len 48 \
  --enc_in 7 \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0.0001 \
  --use_bias  >logs/LongForecasting/$model_name'_'ill_$seq_len'_'48.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id ill_$seq_len'_'60 \
  --model $model_name \
  --data ill \
  --features M \
  --seq_len $seq_len \
  --pred_len 60 \
  --enc_in 7 \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0.0001 \
  --use_bias  >logs/LongForecasting/$model_name'_'ill_$seq_len'_'60.log