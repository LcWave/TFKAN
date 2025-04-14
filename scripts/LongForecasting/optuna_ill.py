import os
import re

import optuna
import logging
import subprocess


def objective(trial, pred_len, logger):
    # 定义超参数空间
    learning_rate = trial.suggest_categorical('learning_rate', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])  # 使用对数均匀分布
    # 限制 learning_rate 保留 6 位小数
    # learning_rate = round(learning_rate, 6)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64])  # 离散的批量大小选择
    weight_decay = trial.suggest_categorical('weight_decay', [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])


    seq_len = 96
    model_name = 'FreLinear'

    best_loss = float('inf')


    # 调用你的训练脚本
    command = [
        "python", "-u", "run_longExp.py",
        "--is_training", "1",
        "--root_path", "./dataset/",
        "--data_path", "national_illness.csv",
        "--model_id", f"ill_{seq_len}_{pred_len}",
        "--model", model_name,
        "--data", "ill",
        "--features", "M",
        "--seq_len", str(seq_len),
        "--pred_len", str(pred_len),
        "--enc_in", "7",
        "--itr", "1",
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--lradj", "type1",
        "--channel_independence", "0",
        "--weight_decay", str(weight_decay),
        "--use_bias"
    ]

    # 启动训练
    result = subprocess.run(command, capture_output=True, text=True)

    # 从输出中提取 MAE 和 RMSE
    mae = None
    rmse = None

    # 从输出中提取损失值（假设损失值包含在日志中）
    for line in result.stdout.splitlines():
        # 使用正则表达式匹配 MAE 和 RMSE
        mae_match = re.search(r"mae:\s*([\d\.]+)", line)
        rmse_match = re.search(r"rmse:\s*([\d\.]+)", line)

        if mae_match:
            mae = float(mae_match.group(1))
        if rmse_match:
            rmse = float(rmse_match.group(1))

        # 如果成功提取了 MSE、MAE 和 RMSE
        if mae is not None and rmse is not None:
            logger.info(f"Trial {trial.number}, Pred Length: {pred_len}, Batch Size: {batch_size}, "
                        f"Learning Rate: {learning_rate}, Weight Decay: {weight_decay}, "
                        f"MAE: {mae}, RMSE: {rmse}")

            # 计算加权损失
            w_mae = 0.5  # MAE的权重
            w_rmse = 0.5  # RMSE的权重
            combined_loss = w_mae * mae + w_rmse * rmse

            # 选择最小的 RMSE 或 MAE 作为优化目标
            best_loss = min(best_loss, combined_loss)  # 或者使用 mae 作为损失
        # else:
        #     None
            # logger.warning(f"Trial {trial.number}, Pred Length: {pred_len} did not return valid MAE/RMSE")

    return best_loss  # 返回最小损失作为优化目标


def main():
    # # 配置日志记录器
    # logging.basicConfig(
    #     level=logging.INFO,  # 设置日志记录级别
    #     format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    #     handlers=[logging.FileHandler('./optuna_log/ill.log', mode='w'),  # 输出到文件
    #               logging.StreamHandler()]  # 也输出到控制台
    # )
    #
    # # 获取日志记录器
    # global logger
    # logger = logging.getLogger('optuna')

    # 预测长度
    pred_lens = [24, 36, 48, 60]

    # 用于记录每个预测长度的最佳参数
    best_params = {}

    # 针对每个预测长度执行优化
    for pred_len in pred_lens:
        # 配置日志记录器
        log_dir = './optuna_log'
        os.makedirs(log_dir, exist_ok=True)
        log_file = f'{log_dir}/ill_pred_len_{pred_len}.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file, mode='w'),
                      logging.StreamHandler()]
        )

        # 获取日志记录器
        logger = logging.getLogger(f'optuna_ill_pred_len_{pred_len}')

        # logger.info(f"Optimizing for prediction length: {pred_len}")

        # 创建一个优化器实例
        # study = optuna.create_study(direction='minimize')  # 目标是最小化损失值
        # study = optuna.create_study(direction='minimize', study_name='ill', storage='sqlite:///optunaDB/ill.db', load_if_exists=True)
        # 为每个预测长度创建独立的数据库文件
        study = optuna.create_study(
            direction='minimize',
            study_name=f'ill_{pred_len}',
            storage=f'sqlite:///optunaDB/ill_{pred_len}.db',
            load_if_exists=True
        )
        logger.info(f"Optimizing for prediction length: {pred_len}")
        study.optimize(lambda trial: objective(trial, pred_len, logger), n_trials=30)  # 尝试10次训练（可以根据需要增加次数）

        # 获取该预测长度下的最佳试验和参数
        best_params[pred_len] = {
            'best_params': study.best_trial.params,
            'best_loss': study.best_trial.value
        }
        # 记录最佳试验信息
        logger.info(f"Best trial for pred_len={pred_len}: {study.best_trial.params}")
        logger.info(f"Best loss for pred_len={pred_len}: {study.best_trial.value}")

    # 输出最终的最佳参数
    logger.info(f"Best parameters across all prediction lengths: {best_params}")

    # 打印出最终的最佳参数
    print(f"Best parameters for each prediction length: {best_params}")


if __name__ == "__main__":
    main()
