#!/bin/bash
# ====================================
# 快速推送到 GitHub 脚本
# ====================================

set -e

echo "=========================================="
echo "  GitHub 快速推送脚本"
echo "=========================================="
echo ""

# 切换到项目目录
cd /home/mry/sjs/MedQA

# 检查是否有 .git 目录
if [ ! -d ".git" ]; then
    echo "📝 初始化 Git 仓库..."
    git init
    git branch -M main
    echo "✓ Git 仓库初始化完成"
else
    echo "✓ Git 仓库已存在"
fi

# 检查远程仓库
if ! git remote | grep -q "origin"; then
    echo ""
    echo "📡 添加远程仓库..."
    git remote add origin https://github.com/Senspecial/MedQA.git
    echo "✓ 远程仓库添加完成"
else
    echo "✓ 远程仓库已配置"
fi

# 显示将要上传的文件
echo ""
echo "📋 检查将要上传的文件..."
echo ""

# 检查是否有大文件
echo "🔍 检查是否有大文件或敏感信息..."
if git status --short | grep -E "(save/|data/.*\.(csv|jsonl|parquet)|\.env|\.key|\.pth|\.bin)"; then
    echo ""
    echo "⚠️  警告: 检测到可能不应上传的文件！"
    echo "请检查 .gitignore 是否正确配置"
    echo ""
    read -p "是否继续？(y/n): " continue_upload
    if [ "$continue_upload" != "y" ] && [ "$continue_upload" != "Y" ]; then
        echo "已取消上传"
        exit 0
    fi
else
    echo "✓ 未检测到大文件"
fi

# 显示文件统计
echo ""
echo "📊 文件统计:"
git status --short | wc -l | xargs echo "  - 修改的文件数量:"
git status --short | grep "^??" | wc -l | xargs echo "  - 未追踪的文件:"

# 询问提交信息
echo ""
read -p "请输入提交信息 (留空使用默认): " commit_message

if [ -z "$commit_message" ]; then
    commit_message="Update: $(date '+%Y-%m-%d %H:%M:%S')"
fi

# 添加所有文件
echo ""
echo "📦 添加文件到暂存区..."
git add .

# 提交
echo "💾 提交更改..."
git commit -m "$commit_message" || {
    echo "ℹ️  没有需要提交的更改"
    exit 0
}

# 询问是否推送
echo ""
read -p "是否推送到 GitHub? (y/n): " do_push

if [ "$do_push" = "y" ] || [ "$do_push" = "Y" ]; then
    echo ""
    echo "🚀 推送到 GitHub..."
    
    # 检查是否是第一次推送
    if git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
        # 已经有上游分支，正常推送
        git push
    else
        # 第一次推送，设置上游分支
        git push -u origin main
    fi
    
    echo ""
    echo "=========================================="
    echo "  ✓ 推送完成！"
    echo "=========================================="
    echo ""
    echo "📍 仓库地址: https://github.com/Senspecial/MedQA"
    echo ""
else
    echo ""
    echo "ℹ️  已提交但未推送到远程仓库"
    echo "稍后可以运行: git push"
fi

