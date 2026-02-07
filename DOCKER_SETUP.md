# Docker Setup Guide

## Yêu cầu / Requirements
- Docker Desktop hoặc Docker Engine cài đặt
- Docker Compose (thường được cài đặt cùng Docker Desktop)

## Cách Sử Dụng / How to Use

### 1. Build Docker Image
```bash
docker build -t medical-lesion-detection:latest .
```

### 2. Chạy Container với Docker
```bash
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/output:/app/output \
  medical-lesion-detection:latest
```

Hoặc trên Windows PowerShell:
```powershell
docker run -p 8501:8501 `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/output:/app/output `
  medical-lesion-detection:latest
```

### 3. Sử dụng Docker Compose (Khuyến nghị)
```bash
# Start container
docker-compose up

# Start in background
docker-compose up -d

# Stop container
docker-compose down

# View logs
docker-compose logs -f
```

### 4. Truy cập Ứng Dụng
Mở trình duyệt và vào: http://localhost:8501

## Cấu Hình / Configuration

### GPU Support (Nếu bạn có NVIDIA GPU)
Uncomment phần GPU trong `docker-compose.yml`:
```yaml
runtime: nvidia
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Thay đổi Port
Sửa `8501` thành port mong muốn trong `docker-compose.yml` hoặc docker command.

## Các Lệnh Hữu Ích / Useful Commands

```bash
# Liệt kê containers đang chạy
docker ps

# Xem logs
docker logs container-name

# Truy cập shell trong container
docker exec -it container-name /bin/bash

# Xóa image
docker rmi medical-lesion-detection:latest

# Xóa unused images
docker image prune

# Check image size
docker images | grep medical
```

## Troubleshooting

### Port 8501 đã được sử dụng
```bash
# Sử dụng port khác
docker run -p 8502:8501 medical-lesion-detection:latest
```

### Memory Issues
```bash
# Gán thêm memory (ví dụ: 4GB)
docker run -m 4g -p 8501:8501 medical-lesion-detection:latest
```

### Permission Denied (Linux)
```bash
sudo chmod 666 /var/run/docker.sock

# Hoặc thêm user vào docker group
sudo usermod -aG docker $USER
newgrp docker
```

## .gitignore Notes

Các file/folder sau sẽ bị ignore:
- `__pycache__/` - Python cache files
- `*.py[cod]` - Compiled Python files
- `models/*.pt` - Trained model files
- `data/train/`, `data/test/`, `data/valid/` - Large datasets
- `.vscode/`, `.idea/` - IDE settings
- `*.ipynb` - Jupyter notebooks
- `.env` - Environment variables
- Các temporary files

Để override gitignore cho một file:
```bash
git add -f filename
```
