#!/bin/bash

echo "🚀 开始自动流程..."

# 1. 确保连接配置是最新的 (防呆设计)
git remote remove origin 2>/dev/null
git remote add origin https://github.com/yuxingong-dotcom/iaa-dashboard_v1.0.git

# 2. 准备代码
git branch -M main
git add .

# 3. 提交代码
# 注意：如果刚才已经commit过了但没push成功，这里会提示nothing to commit，是正常的，不影响后续
current_time=$(date "+%Y-%m-%d %H:%M:%S")
git commit -m "自动更新: $current_time" 2>/dev/null

echo "📡 正在尝试上传到 GitHub (网络不稳定可能会卡住，请耐心等待)..."

# 4. 死磕上传 (无限重试逻辑)
# 只要 push 命令返回失败 (非0)，就一直循环
count=1
until git push -u origin main --force; do
    echo "❌ 第 $count 次上传失败 (连接超时)..."
    echo "🔄 等待 2 秒后自动重试..."
    sleep 2
    ((count++))
    
    # 防止无限死循环（比如断网了），尝试 10 次后强制停止
    if [ $count -gt 10 ]; then
        echo "⛔️ 尝试了 10 次都失败了，请检查你的 VPN 是否开启，或者网络是否正常！"
        exit 1
    fi
done

echo "✅ 终于上传成功了！"