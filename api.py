import os
import shutil
import tempfile
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests
from omegaconf import OmegaConf
import uvicorn
from typing import Optional, Dict, List, Union, Any
import logging
import time
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import threading
import queue

# 导入inference.py中的main函数
from scripts.inference import main

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用，配置Swagger文档
app = FastAPI(
    title="LatentSync API",
    description="用于唇形同步的API，支持从网络URL下载视频和音频文件并进行处理",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {"name": "唇形同步", "description": "唇形同步处理相关接口"},
        {"name": "任务管理", "description": "任务状态和队列管理相关接口"},
        {"name": "系统", "description": "系统状态相关接口"},
    ]
)

# 任务状态存储
task_status: Dict[str, Dict[str, Any]] = {}

# 任务队列
task_queue = queue.Queue()

# 控制是否有任务正在运行
is_task_running = False
task_lock = threading.Lock()

class SyncRequest(BaseModel):
    video_url: str = Field(..., description="要处理的视频URL")
    audio_url: str = Field(..., description="用于唇形同步的音频URL")
    inference_ckpt_path: str = Field(default="checkpoints/unet.ckpt", description="模型检查点路径")
    unet_config_path: str = Field(default="configs/unet.yaml", description="模型配置文件路径")
    inference_steps: int = Field(default=20, description="推理步数")
    guidance_scale: float = Field(default=1.0, description="引导比例")
    seed: int = Field(default=1247, description="随机种子")
    ignore_cache: bool = Field(default=False, description="是否忽略缓存")

    model_config = {
        "json_schema_extra": {
            "example": {
                "video_url": "https://example.com/video.mp4",
                "audio_url": "https://example.com/audio.wav",
                "inference_ckpt_path": "checkpoints/unet.ckpt",
                "unet_config_path": "configs/unet.yaml",
                "inference_steps": 20,
                "guidance_scale": 1.0,
                "seed": 1247,
                "ignore_cache": False
            }
        }
    }

class SyncResponse(BaseModel):
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    message: str = Field(..., description="任务状态描述信息")
    queue_position: Optional[int] = Field(default=None, description="任务在队列中的位置")
    output_video_path: Optional[str] = Field(default=None, description="输出视频路径")
    output_mask_path: Optional[str] = Field(default=None, description="输出掩码视频路径")

class TaskStatusResponse(BaseModel):
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    message: str = Field(..., description="任务状态描述信息")
    progress: float = Field(default=0.0, description="任务进度（0-1）")
    queue_position: Optional[int] = Field(default=None, description="任务在队列中的位置")
    start_time: float = Field(..., description="任务开始时间（Unix时间戳）")
    end_time: Optional[float] = Field(default=None, description="任务结束时间（Unix时间戳）")
    output_video_url: Optional[str] = Field(default=None, description="输出视频的URL")
    output_mask_url: Optional[str] = Field(default=None, description="输出掩码视频的URL")

class QueuedTask(BaseModel):
    task_id: str = Field(..., description="任务ID")
    position: int = Field(..., description="任务在队列中的位置")
    status: str = Field(..., description="任务状态")

class QueueStatusResponse(BaseModel):
    queue_length: int = Field(..., description="队列中的任务数量")
    is_processing: bool = Field(..., description="是否有任务正在处理")
    queued_tasks: List[QueuedTask] = Field(default=[], description="队列中的任务列表")

class HealthResponse(BaseModel):
    status: str = Field(..., description="服务状态")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="错误详情")

# 任务队列处理线程
def task_processor():
    global is_task_running
    while True:
        try:
            # 从队列获取任务
            task_id, request = task_queue.get()
            
            with task_lock:
                is_task_running = True
                
                # 更新所有队列中任务的位置
                update_queue_positions()
            
            logger.info(f"开始处理任务 {task_id}")
            
            # 处理任务
            process_request(request, task_id)
            
            with task_lock:
                is_task_running = False
                
                # 更新所有队列中任务的位置
                update_queue_positions()
                
            # 标记任务完成
            task_queue.task_done()
            
        except Exception as e:
            logger.error(f"任务处理器错误: {str(e)}")
            with task_lock:
                is_task_running = False

# 更新所有队列中任务的位置
def update_queue_positions():
    # 获取队列中所有任务
    with task_queue.mutex:
        queue_items = list(task_queue.queue)
    
    # 更新每个任务的队列位置
    for position, (task_id, _) in enumerate(queue_items):
        if task_id in task_status and task_status[task_id]["status"] == "queued":
            task_status[task_id]["queue_position"] = position
            task_status[task_id]["message"] = f"等待处理中，队列位置: {position + 1}"

def download_file(url: str, local_path: str):
    """从URL下载文件到本地路径"""
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        return True
    except Exception as e:
        logger.error(f"下载文件失败: {str(e)}")
        return False

def process_request(request: SyncRequest, task_id: str):
    """处理唇形同步请求的任务"""
    # 更新任务状态
    task_status[task_id]["status"] = "downloading"
    task_status[task_id]["message"] = "正在下载文件"
    task_status[task_id]["progress"] = 0.1
    task_status[task_id]["queue_position"] = None
    
    # 创建输出目录存放下载的文件和输出
    output_dir = os.path.join("outputs", task_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义本地文件路径
    video_ext = os.path.splitext(request.video_url.split('/')[-1])[-1] or ".mp4"
    audio_ext = os.path.splitext(request.audio_url.split('/')[-1])[-1] or ".wav"
    
    video_path = os.path.join(output_dir, f"input_video{video_ext}")
    audio_path = os.path.join(output_dir, f"input_audio{audio_ext}")
    video_out_path = os.path.join(output_dir, "output.mp4")
    
    try:
        # 下载视频和音频文件
        logger.info(f"下载视频: {request.video_url}")
        if not download_file(request.video_url, video_path):
            task_status[task_id]["status"] = "failed"
            task_status[task_id]["message"] = "视频下载失败"
            task_status[task_id]["end_time"] = time.time()
            return
            
        task_status[task_id]["progress"] = 0.3
        
        logger.info(f"下载音频: {request.audio_url}")
        if not download_file(request.audio_url, audio_path):
            task_status[task_id]["status"] = "failed"
            task_status[task_id]["message"] = "音频下载失败"
            task_status[task_id]["end_time"] = time.time()
            return
        
        task_status[task_id]["status"] = "processing"
        task_status[task_id]["message"] = "正在处理唇形同步"
        task_status[task_id]["progress"] = 0.5
            
        # 准备参数
        config = OmegaConf.load(request.unet_config_path)
        
        # 创建一个类似于argparse的对象
        class Args:
            def __init__(self):
                self.unet_config_path = request.unet_config_path
                self.inference_ckpt_path = request.inference_ckpt_path
                self.video_path = video_path
                self.audio_path = audio_path
                self.video_out_path = video_out_path
                self.inference_steps = request.inference_steps
                self.guidance_scale = request.guidance_scale
                self.seed = request.seed
                self.ignore_cache = request.ignore_cache
        
        args = Args()
        
        # 调用main函数
        logger.info("开始唇形同步处理")
        main(config, args)
        
        # 生成的掩码文件路径
        mask_out_path = video_out_path.replace(".mp4", "_mask.mp4")
        
        # 检查输出文件是否存在
        if os.path.exists(video_out_path):
            # 更新任务状态
            task_status[task_id]["status"] = "completed"
            task_status[task_id]["message"] = "处理完成"
            task_status[task_id]["progress"] = 1.0
            task_status[task_id]["end_time"] = time.time()
            
            # 生成文件URL
            video_url = f"/results/{task_id}/output.mp4"
            mask_url = f"/results/{task_id}/output_mask.mp4"
            
            task_status[task_id]["output_video_url"] = video_url
            task_status[task_id]["output_mask_url"] = mask_url if os.path.exists(mask_out_path) else None
            
            logger.info(f"处理完成，输出文件: {video_out_path}")
        else:
            task_status[task_id]["status"] = "failed"
            task_status[task_id]["message"] = "处理失败，未生成输出文件"
            task_status[task_id]["end_time"] = time.time()
            logger.error("处理失败，未生成输出文件")
        
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        task_status[task_id]["status"] = "failed"
        task_status[task_id]["message"] = f"处理失败: {str(e)}"
        task_status[task_id]["end_time"] = time.time()

# 创建输出目录
os.makedirs("outputs", exist_ok=True)

# 挂载静态文件服务
app.mount("/results", StaticFiles(directory="outputs"), name="results")

@app.post("/sync", response_model=SyncResponse, tags=["唇形同步"], 
          responses={
              200: {"description": "任务已成功提交到队列", "model": SyncResponse},
              400: {"description": "无效的请求参数", "model": ErrorResponse},
              500: {"description": "服务器内部错误", "model": ErrorResponse}
          })
async def sync_video(request: SyncRequest):
    """
    提交唇形同步任务
    
    提交一个视频URL和音频URL，创建唇形同步任务。任务将会进入队列，每次只处理一个任务。
    
    - **video_url**: 要处理的视频URL
    - **audio_url**: 用于唇形同步的音频URL
    - **inference_ckpt_path**: 模型检查点路径
    - **unet_config_path**: 模型配置文件路径
    - **inference_steps**: 推理步数
    - **guidance_scale**: 引导比例
    - **seed**: 随机种子
    - **ignore_cache**: 是否忽略缓存
    """
    task_id = str(uuid.uuid4())
    
    # 初始化任务状态
    queue_position = task_queue.qsize()
    
    task_status[task_id] = {
        "status": "queued",
        "message": f"任务已加入队列，等待处理，队列位置: {queue_position + 1}",
        "progress": 0.0,
        "queue_position": queue_position,
        "start_time": time.time(),
        "end_time": None,
        "output_video_url": None,
        "output_mask_url": None
    }
    
    # 将任务添加到任务队列
    task_queue.put((task_id, request))
    
    # 更新所有队列中任务的位置
    update_queue_positions()
    
    return SyncResponse(
        task_id=task_id,
        status="queued",
        message=f"处理任务已添加到队列，任务ID: {task_id}，队列位置: {queue_position + 1}",
        queue_position=queue_position
    )

@app.get("/task/{task_id}", response_model=TaskStatusResponse, tags=["任务管理"],
         responses={
             200: {"description": "成功返回任务状态", "model": TaskStatusResponse},
             404: {"description": "任务不存在", "model": ErrorResponse},
             500: {"description": "服务器内部错误", "model": ErrorResponse}
         })
async def get_task_status(task_id: str):
    """
    获取任务处理状态
    
    通过任务ID查询任务的处理状态、进度和结果
    
    - **task_id**: 任务ID
    """
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    status_data = task_status[task_id]
    return TaskStatusResponse(task_id=task_id, **status_data)

@app.get("/queue", response_model=QueueStatusResponse, tags=["任务管理"],
         responses={
             200: {"description": "成功返回队列状态", "model": QueueStatusResponse},
             500: {"description": "服务器内部错误", "model": ErrorResponse}
         })
async def get_queue_status():
    """
    获取队列状态
    
    查询当前任务队列的状态，包括队列长度、是否有任务正在处理以及队列中的任务列表
    """
    queued_tasks = [
        QueuedTask(
            task_id=task_id, 
            position=data["queue_position"], 
            status=data["status"]
        )
        for task_id, data in task_status.items()
        if data["status"] == "queued" and "queue_position" in data
    ]
    
    # 按队列位置排序
    queued_tasks.sort(key=lambda x: x.position if x.position is not None else float('inf'))
    
    return QueueStatusResponse(
        queue_length=task_queue.qsize(),
        is_processing=is_task_running,
        queued_tasks=queued_tasks
    )

@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """
    健康检查
    
    检查API服务是否正常运行
    """
    return HealthResponse(status="healthy")

# 启动任务处理线程
task_processor_thread = threading.Thread(target=task_processor, daemon=True)
task_processor_thread.start()

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False) 