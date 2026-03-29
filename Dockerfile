# 1. 使用官方轻量级 Python 3.10 镜像作为基础
FROM python:3.10-slim

# 2. 设置环境变量 (防止 python 缓冲日志，且不生成 .pyc 缓存)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 3. 设置容器内的工作目录
WORKDIR /app

# 4. 安装 uv (极速包管理工具)
RUN pip install --no-cache-dir uv

# 5. 复制依赖清单并安装
COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

# 6. 复制项目源代码和数据集 (关键的一步：把代码和数据都放进去)
COPY src/ ./src/
COPY data/ ./data/

# 7. 暴露 FastAPI 的 8000 端口
EXPOSE 8000

# 8. 启动容器时的默认命令
CMD ["uvicorn", "lyricmind.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]