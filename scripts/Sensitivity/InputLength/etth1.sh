
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Sensitivity/InputLength" ]; then
    mkdir ./logs/Sensitivity/InputLength
fi
seq_lens=(48 192 336 720)
model_name=FreLinear

for seq_len in "${seq_lens[@]}"
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --itr 1 \
  --batch_size 4 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0.00001 \
  --use_bias  >logs/Sensitivity/InputLength/$model_name'_'ETTh1_$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0 \
  --use_bias  >logs/Sensitivity/InputLength/$model_name'_'ETTh1_$seq_len'_'192.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0.0001 \
  --use_bias  >logs/Sensitivity/InputLength/$model_name'_'ETTh1_$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --itr 1 \
  --batch_size 4 \
  --learning_rate 0.00005 \
  --lradj type1 \
  --channel_independence 0 \
  --weight_decay 0.0001 \
  --use_bias  >logs/Sensitivity/InputLength/$model_name'_'ETTh1_$seq_len'_'720.log
done