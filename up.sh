#!/bin/bash

# 1. 打印提示
echo "🚀 开始自动上传..."

# 2. 如果你需要每次都重置连接（防止网络抽风），请保留下面两行；
#    如果网络正常，可以把下面两行删掉，速度会更快。
git remote remove origin 2>/dev/null  # 忽略报错
git remote add origin https://github.com/yuxingong-dotcom/iaa-dashboard_v1.0.git

# 3. 确保分支名正确
git branch -M main

# 4. 添加所有修改的文件
git add .

# 5. 提交修改（自动加上当前时间作为备注）
current_time=$(date "+%Y-%m-%d %H:%M:%S")
git commit -m "自动更新: $current_time"

# 6. 强制推送到 Github
git push -u origin main --force

echo "✅ 上传完成！"