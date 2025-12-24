import os
import re

# 你关注的数据集和 shot
datalist = ["artaxor", "clipart1k", "dior", "fish", "neu-det", "uodd"]
shot_list = [1, 5, 10]

print(f"{'Dataset_Shot':<25} | {'mAP':<8} | {'AP50':<8}")
print("-" * 45)

for dataset in datalist:
    for shot in shot_list:
        folder = f"output/vitb/{dataset}_{shot}shot"
        log_path = os.path.join(folder, "eval_fix_log.txt")
        
        entry_name = f"{dataset}_{shot}shot"
        mAP, AP50 = "[找不到日志]", "[评价失败]"

        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                content = f.read()
                # 寻找 copypaste: 后面那一行数字
                matches = re.findall(r"copypaste: ([\d\.,]+)", content)
                if matches:
                    # 最后一项通常是最新的结果
                    nums = matches[-1].split(',')
                    mAP, AP50 = nums[0], nums[1]
                else:
                    mAP = "[尚未跑完]"
        
        print(f"{entry_name:<25} | {mAP:<8} | {AP50:<8}")