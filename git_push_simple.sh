#!/bin/bash
# 简单的 Git 推送脚本

cd /home/mry/sjs/MedQA

echo "=========================================="
echo "  简单 Git 推送"
echo "=========================================="
echo ""

# 获取当前分支
current_branch=$(git rev-parse --abbrev-ref HEAD)
echo "当前分支: $current_branch"
echo ""

# 检查状态
echo "文件状态:"
git status --short
echo ""

# 添加所有文件
echo "添加文件到暂存区..."
git add .

# 提交
read -p "输入提交信息: " commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="Update $(date '+%Y-%m-%d %H:%M:%S')"
fi

git commit -m "$commit_msg" || {
    echo "没有需要提交的更改"
    exit 0
}

# 推送
echo ""
echo "推送到 GitHub (分支: $current_branch)..."
git push -u origin $current_branch

echo ""
echo "✓ 推送完成！"
echo "仓库地址: https://github.com/Senspecial/MedQA"

