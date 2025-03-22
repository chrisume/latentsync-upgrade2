# LatentSync API 使用指南

## 简介

LatentSync API 是基于 FastAPI 开发的唇形同步处理服务，它可以自动从网络下载视频和音频文件，然后进行唇形同步处理。API 服务通过队列管理系统确保同一时间只处理一个任务，避免 GPU 资源竞争。

## 安装

### 安装依赖

```bash
# 安装API服务依赖
pip install -r requirements_api.txt
```

### 准备模型文件

确保 LatentSync 的预训练模型文件放置在正确的位置：
- `checkpoints/unet.ckpt`：模型检查点文件
- `checkpoints/whisper/small.pt` 或 `checkpoints/whisper/tiny.pt`：用于音频特征提取的 Whisper 模型
- `configs/unet.yaml`：模型配置文件

## 启动服务

```bash
python api.py
```

服务将在 `http://localhost:8000` 运行。

可以通过浏览器访问以下地址查看 API 文档：
- Swagger UI：`http://localhost:8000/docs`
- ReDoc：`http://localhost:8000/redoc`

## 使用方法

### 1. 提交唇形同步任务

```python
import requests

# 提交任务
response = requests.post(
    "http://localhost:8000/sync",
    json={
        "video_url": "https://example.com/video.mp4",  # 要处理的视频URL
        "audio_url": "https://example.com/audio.wav",  # 用于唇形同步的音频URL
        # 可选参数
        # "inference_ckpt_path": "checkpoints/unet.ckpt",
        # "inference_steps": 20,
        # "guidance_scale": 1.0
    }
)

# 获取任务ID和队列位置
task_id = response.json()["task_id"]
queue_position = response.json()["queue_position"]
print(f"任务ID: {task_id}, 队列位置: {queue_position + 1}")
```

### 2. 查询任务状态

```python
import requests
import time

task_id = "你的任务ID"

# 轮询任务状态直到完成
while True:
    response = requests.get(f"http://localhost:8000/task/{task_id}")
    data = response.json()
    
    status = data["status"]
    progress = data["progress"] * 100
    message = data["message"]
    
    print(f"状态: {status}, 进度: {progress:.1f}%, 消息: {message}")
    
    # 如果任务完成或失败，退出循环
    if status in ["completed", "failed"]:
        break
    
    # 等待5秒后再次查询
    time.sleep(5)
```

### 3. 获取处理结果

当任务状态为 "completed" 时，可以下载处理后的视频：

```python
# 检查任务是否完成
if data["status"] == "completed":
    # 获取视频URL
    video_url = f"http://localhost:8000{data['output_video_url']}"
    mask_url = f"http://localhost:8000{data['output_mask_url']}" if data['output_mask_url'] else None
    
    # 下载处理后的视频
    response = requests.get(video_url)
    with open("result_video.mp4", "wb") as f:
        f.write(response.content)
    
    print("视频已下载到 result_video.mp4")
    
    # 如果有掩码视频，也下载下来
    if mask_url:
        response = requests.get(mask_url)
        with open("result_mask.mp4", "wb") as f:
            f.write(response.content)
        print("掩码视频已下载到 result_mask.mp4")
```

### 4. 查看队列状态

```python
# 查询当前队列状态
response = requests.get("http://localhost:8000/queue")
queue_data = response.json()

print(f"队列长度: {queue_data['queue_length']}")
print(f"是否有任务处理中: {queue_data['is_processing']}")

# 显示队列中的任务
if queue_data['queued_tasks']:
    print("等待中的任务:")
    for task in queue_data['queued_tasks']:
        print(f"  任务ID: {task['task_id']}, 位置: {task['position'] + 1}")
```

## 示例完整流程

```python
import requests
import time

# 1. 提交任务
response = requests.post(
    "http://localhost:8000/sync",
    json={
        "video_url": "https://example.com/video.mp4",
        "audio_url": "https://example.com/audio.wav"
    }
)

task_id = response.json()["task_id"]
print(f"任务已提交，任务ID: {task_id}")

# 2. 轮询任务状态
while True:
    response = requests.get(f"http://localhost:8000/task/{task_id}")
    data = response.json()
    
    print(f"状态: {data['status']}, 进度: {data['progress']*100:.1f}%, 消息: {data['message']}")
    
    if data["status"] in ["completed", "failed"]:
        break
    
    time.sleep(5)

# 3. 下载结果（如果成功）
if data["status"] == "completed":
    video_url = f"http://localhost:8000{data['output_video_url']}"
    print(f"下载处理后的视频: {video_url}")
    
    response = requests.get(video_url)
    with open("result.mp4", "wb") as f:
        f.write(response.content)
    
    print("视频已保存到 result.mp4")
else:
    print(f"处理失败: {data['message']}")
```

## 注意事项

1. API 服务同一时间只处理一个任务，其他任务会排队等待
2. 所有处理文件保存在 `outputs/{task_id}/` 目录下
3. 定期清理 `outputs` 目录以避免占用过多磁盘空间
4. 保证系统有足够的网络带宽下载视频和音频文件
5. 确保系统有足够的 GPU 内存用于处理 