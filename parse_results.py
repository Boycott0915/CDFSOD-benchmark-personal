import os
import re

# 你关注的数据集和 shot
datalist = ["artaxor", "clipart1k", "dior", "fish", "neu-det", "uodd"]
shot_list = [1, 5, 10]

print(f"{'Dataset_Shot':<25} | {'mAP':<8} | {'AP50':<8}")
print("-" * 45)

for dataset in datalist:
    for shot in shot_list:
        # 注意这里：确保路径和 shell 脚本里的 OUTPUT_DIR 完全对齐
        # 你的 shell 是 output/vitb/dior_10shot/
        folder = f"output/vitb/{dataset}_{shot}shot"
        
        # 兼容两种可能的日志文件名
        possible_logs = ["log.txt", "eval_fix_log.txt"]
        log_path = None
        
        if os.path.exists(folder):
            for log_name in possible_logs:
                p = os.path.join(folder, log_name)
                if os.path.exists(p):
                    log_path = p
                    break
        
        entry_name = f"{dataset}_{shot}shot"
        mAP, AP50 = "[找不到文件]", "[评价失败]"

        if log_path:
            with open(log_path, 'r') as f:
                content = f.read()
                # 关键：Detectron2 的日志里，评价结果通常在最后
                # 我们寻找 copypaste: 这一行
                matches = re.findall(r"copypaste: ([\d\.]+),([\d\.]+)", content)
                if matches:
                    # 取最后一次评估的结果（因为训练过程中可能会有多次 eval）
                    mAP, AP50 = matches[-1]
                else:
                    # 如果有 log 文件但没搜到关键字，说明还没跑完评估
                    mAP = "[尚未跑完]"
        
        print(f"{entry_name:<25} | {mAP:<8} | {AP50:<8}")