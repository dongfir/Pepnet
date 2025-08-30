import os
import sys
from config import Config as cfg

def run_train():
    os.system(f"python {cfg.train_script_path}")

def run_eval():
    os.system(f"python {cfg.eval_script_path}")

def run_predict():
    os.system(f"python {cfg.predict_script_path}")

if __name__ == "__main__":
    print("\n可用命令：")
    print("1. 训练模型")
    print("2. 验证/评估")
    print("3. 进行预测")
    cmd = input("输入编号选择运行任务: ")

    if cmd == "1":
        run_train()
    elif cmd == "2":
        run_eval()
    elif cmd == "3":
        run_predict()
    else:
        print("未知命令。")
