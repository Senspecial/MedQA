# 📤 GitHub 上传指南

## 快速上传步骤

### 1️⃣ 初始化 Git 仓库

```bash
# 进入项目目录
cd /home/mry/sjs/MedQA

# 初始化 Git（如果还没有）
git init

# 设置默认分支为 main
git branch -M main
```

### 2️⃣ 检查忽略文件

```bash
# 查看哪些文件会被上传（应该不包含 save/、data/ 等大文件）
git status

# 查看被忽略的文件
git status --ignored
```

### 3️⃣ 添加远程仓库

```bash
# 添加 GitHub 远程仓库
git remote add origin https://github.com/Senspecial/MedQA.git

# 验证远程仓库
git remote -v
```

### 4️⃣ 提交代码

```bash
# 添加所有文件（会自动排除 .gitignore 中的文件）
git add .

# 提交
git commit -m "Initial commit: Chinese Medical QA System with RAG and Agent"

# 推送到 GitHub
git push -u origin main
```

---

## ⚠️ 重要提示

### 被忽略的文件类型

以下文件类型会被自动忽略，**不会上传到 GitHub**：

✅ **模型权重和检查点**
- `save/` 目录
- `*.bin`, `*.safetensors`, `*.pth`, `*.pt`, `*.ckpt`
- `checkpoint-*/`

✅ **数据文件**
- `data/` 目录（保留目录结构）
- `crawled_data/`, `processed_data/`, `knowledge_base/`
- `*.csv`, `*.jsonl`, `*.parquet`

✅ **日志和缓存**
- `logs/`, `embedding_cache/`
- `__pycache__/`, `*.pyc`
- `.pytest_cache/`, `.coverage`

✅ **虚拟环境**
- `venv/`, `env/`, `ENV/`

✅ **IDE 配置**
- `.vscode/`, `.idea/`, `.cursor/`

✅ **敏感信息**
- `.env`, `*.key`, `secrets/`

### 保留的文件

以下文件会被上传：

✅ 源代码（`src/`）
✅ 脚本（`scripts/`, `examples/`）
✅ 配置文件（`requirements*.txt`）
✅ 文档（`README.md`, `docs/`）
✅ 示例数据（`data/examples/`）

---

## 🔧 常见问题

### Q1: 如果误上传了大文件怎么办？

```bash
# 从 Git 历史中删除文件
git rm --cached save/large_model.bin

# 重新提交
git commit -m "Remove large model file"
git push --force
```

### Q2: 如何查看哪些文件被忽略？

```bash
git status --ignored
```

### Q3: 如何上传部分数据文件作为示例？

在 `.gitignore` 中添加例外：

```gitignore
# 忽略所有数据
data/

# 但保留示例
!data/examples/
```

然后：

```bash
mkdir -p data/examples
# 复制小示例文件到 data/examples/
git add data/examples/
git commit -m "Add example data"
```

### Q4: 如果项目已经有提交历史？

```bash
# 强制推送（⚠️ 会覆盖远程仓库）
git push -u origin main --force
```

### Q5: 如何使用 SSH 方式推送？

```bash
# 移除 HTTPS 远程仓库
git remote remove origin

# 添加 SSH 远程仓库
git remote add origin git@github.com:Senspecial/MedQA.git

# 推送
git push -u origin main
```

---

## 📦 使用 Git LFS（可选，用于大文件）

如果需要上传模型权重（不推荐），可以使用 Git LFS：

```bash
# 安装 Git LFS
git lfs install

# 追踪大文件
git lfs track "*.bin"
git lfs track "*.safetensors"
git lfs track "*.pth"

# 添加 .gitattributes
git add .gitattributes

# 正常提交和推送
git add save/model.bin
git commit -m "Add model with Git LFS"
git push
```

⚠️ **注意**: GitHub 免费账户 LFS 存储限额为 1GB

---

## 📝 推荐的提交信息格式

```bash
# 初始提交
git commit -m "Initial commit: Chinese Medical QA System"

# 功能添加
git commit -m "feat: Add Agent demo with custom tools"

# Bug 修复
git commit -m "fix: Fix import path in agent module"

# 文档更新
git commit -m "docs: Update README with Agent usage"

# 依赖更新
git commit -m "chore: Update requirements.txt"
```

---

## 🚀 快速命令（一键上传）

```bash
#!/bin/bash
# 快速上传脚本

cd /home/mry/sjs/MedQA

# 检查是否有 .git 目录
if [ ! -d ".git" ]; then
    git init
    git branch -M main
fi

# 添加远程仓库（如果不存在）
if ! git remote | grep -q "origin"; then
    git remote add origin https://github.com/Senspecial/MedQA.git
fi

# 添加所有文件
git add .

# 提交
git commit -m "Update: $(date '+%Y-%m-%d %H:%M:%S')"

# 推送
git push -u origin main
```

保存为 `quick_push.sh` 并执行：

```bash
chmod +x quick_push.sh
./quick_push.sh
```

---

## 📊 预估上传大小

根据 `.gitignore` 配置，预估上传内容：

| 目录/文件 | 是否上传 | 预估大小 |
|----------|---------|---------|
| `src/` | ✅ 是 | ~5 MB |
| `scripts/` | ✅ 是 | ~1 MB |
| `examples/` | ✅ 是 | ~2 MB |
| `docs/` | ✅ 是 | ~500 KB |
| `requirements*.txt` | ✅ 是 | ~10 KB |
| `README.md` | ✅ 是 | ~50 KB |
| **总计** | - | **~10 MB** |

❌ **不上传**:
- `save/` - 模型权重（~5-10 GB）
- `data/` - 数据集（~1-5 GB）
- `logs/` - 日志文件（~100 MB）

---

## 🎯 最终检查清单

上传前请确认：

- [ ] `.gitignore` 文件已配置
- [ ] `.gitattributes` 文件已创建
- [ ] 模型权重目录（`save/`）被忽略
- [ ] 大数据文件被忽略
- [ ] 敏感信息（`.env`, `*.key`）被忽略
- [ ] `README.md` 已更新
- [ ] 依赖文件（`requirements*.txt`）已完善

```bash
# 运行检查
git status | grep -E "(save/|data/|\.env|\.key)"
# 如果有输出，说明大文件可能会被上传，需要检查
```

---

需要帮助？查看 [GitHub 文档](https://docs.github.com/zh)

