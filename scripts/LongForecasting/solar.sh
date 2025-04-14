
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=FreLinear

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path solar_AL.csv \
  --model_id solar_$seq_len'_'96 \
  --model $model_name \
  --data solar \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 137 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0 \
  --use_bias  >logs/LongForecasting/$model_name'_'solar_$seq_len'_'96_1.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path solar_AL.csv \
  --model_id solar_$seq_len'_'96 \
  --model $model_name \
  --data solar \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 137 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0.0001 \
  --use_bias  >logs/LongForecasting/$model_name'_'solar_$seq_len'_'96_2.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path solar_AL.csv \
  --model_id solar_$seq_len'_'96 \
  --model $model_name \
  --data solar \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 137 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0  >logs/LongForecasting/$model_name'_'solar_$seq_len'_'96_3.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path solar_AL.csv \
  --model_id solar_$seq_len'_'96 \
  --model $model_name \
  --data solar \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 137 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0.0001  >logs/LongForecasting/$model_name'_'solar_$seq_len'_'96_4.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path solar_AL.csv \
  --model_id solar_$seq_len'_'192 \
  --model $model_name \
  --data solar \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 137 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0 \
  --use_bias  >logs/LongForecasting/$model_name'_'solar_$seq_len'_'192_1.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path solar_AL.csv \
  --model_id solar_$seq_len'_'192 \
  --model $model_name \
  --data solar \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 137 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0.0001 \
  --use_bias  >logs/LongForecasting/$model_name'_'solar_$seq_len'_'192_2.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path solar_AL.csv \
  --model_id solar_$seq_len'_'192 \
  --model $model_name \
  --data solar \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 137 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0  >logs/LongForecasting/$model_name'_'solar_$seq_len'_'192_3.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path solar_AL.csv \
  --model_id solar_$seq_len'_'192 \
  --model $model_name \
  --data solar \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 137 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0.0001  >logs/LongForecasting/$model_name'_'solar_$seq_len'_'192_4.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path solar_AL.csv \
  --model_id solar_$seq_len'_'336 \
  --model $model_name \
  --data solar \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 137 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0 \
  --use_bias  >logs/LongForecasting/$model_name'_'solar_$seq_len'_'336_1.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path solar_AL.csv \
  --model_id solar_$seq_len'_'336 \
  --model $model_name \
  --data solar \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 137 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0.0001 \
  --use_bias  >logs/LongForecasting/$model_name'_'solar_$seq_len'_'336_2.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path solar_AL.csv \
  --model_id solar_$seq_len'_'336 \
  --model $model_name \
  --data solar \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 137 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0  >logs/LongForecasting/$model_name'_'solar_$seq_len'_'336_3.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path solar_AL.csv \
  --model_id solar_$seq_len'_'336 \
  --model $model_name \
  --data solar \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 137 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0.0001  >logs/LongForecasting/$model_name'_'solar_$seq_len'_'336_4.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path solar_AL.csv \
  --model_id solar_$seq_len'_'720 \
  --model $model_name \
  --data solar \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 137 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0 \
  --use_bias  >logs/LongForecasting/$model_name'_'solar_$seq_len'_'720_1.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path solar_AL.csv \
  --model_id solar_$seq_len'_'720 \
  --model $model_name \
  --data solar \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 137 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0.0001 \
  --use_bias  >logs/LongForecasting/$model_name'_'solar_$seq_len'_'720_2.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path solar_AL.csv \
  --model_id solar_$seq_len'_'720 \
  --model $model_name \
  --data solar \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 137 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0  >logs/LongForecasting/$model_name'_'solar_$seq_len'_'720_3.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path solar_AL.csv \
  --model_id solar_$seq_len'_'720 \
  --model $model_name \
  --data solar \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 137 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0.0001  >logs/LongForecasting/$model_name'_'solar_$seq_len'_'720_4.log