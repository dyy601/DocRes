# 本地训练

环境：D:/anaconda/envs/DocRes/python.exe
数据默认路径由train.py所在目录推导，当前为D:/Project/DocRes-master/data/train。

PowerShell启动：
```powershell
cd D:\Project\DocRes-master
.\start_train.ps1 --total_iter 100000 --tboard
```
也可直接运行：
```powershell
& D:\anaconda\envs\DocRes\python.exe D:\Project\DocRes-master\train.py --data_root D:\Project\DocRes-master\data\train --batch_size 1 --im_size 256 --experiment_name docres_local --tboard
```
默认workers=0，单GPU；不再使用原8卡NCCL启动方式。每5000步以及最后一步保存checkpoint。

恢复：在上述命令后加 `--resume checkpoints/docres_local/5000.pkl`，保持原total_iter以延续原学习率计划。恢复模型、优化器、scheduler和AMP scaler；不保证逐样本、随机增强或随机任务序列完全复现。仅加载可信checkpoint。

验证完成：五任务各一次真实训练更新；checkpoint第4步恢复到第5步；TensorBoard事件文件成功写出。验证输出在checkpoints/readiness_check和readiness_resume，只用于诊断，不是正式预训练权重。未启动长期训练。

训练入口移除未使用的piq和utils导入，避免本环境OpenMP重复加载；没有设置KMP_DUPLICATE_LIB_OK。其他仍导入这些库的脚本可能需要单独检查。

去阴影索引含517对；另四任务各200个样本。去畸变无额外背景时保留UVDoc原有背景，有背景时可放入data/train/dewarping/background。数据来源和测试集转训练限制见data/train/DATASETS_README.md。

本环境实际PyTorch为2.11.0+cu130；原requirements.txt为论文旧环境，勿在此环境直接按旧文件降级重装。TensorBoard已安装。
