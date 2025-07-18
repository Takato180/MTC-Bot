#!/usr/bin/env python3
"""
学習進捗リアルタイム監視
"""

import os
import time
import psutil
import subprocess
from datetime import datetime
import glob

def get_gpu_status():
    """GPU状況取得"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            values = result.stdout.strip().split(', ')
            return {
                'gpu_util': int(values[0]),
                'mem_used': int(values[1]),
                'mem_total': int(values[2]),
                'temperature': int(values[3]),
                'power': float(values[4])
            }
    except:
        pass
    return None

def get_training_processes():
    """学習プロセス取得"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'create_time']):
        try:
            if 'python' in proc.info['name'].lower():
                mem_gb = proc.info['memory_info'].rss / (1024**3)
                if mem_gb > 1.0:
                    start_time = datetime.fromtimestamp(proc.info['create_time'])
                    processes.append({
                        'pid': proc.info['pid'],
                        'memory_gb': mem_gb,
                        'start_time': start_time
                    })
        except:
            continue
    return processes

def check_model_progress():
    """モデル保存状況確認"""
    patterns = [
        "models/official_patchtst/checkpoints/best_model.pth",
        "models/official_patchtst/training_results.json",
        "models/official_patchtst/logs/events.out.tfevents.*"
    ]
    
    status = {}
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            latest = max(files, key=os.path.getmtime)
            status[pattern] = {
                'exists': True,
                'file': latest,
                'size': os.path.getsize(latest) / (1024**2) if os.path.exists(latest) else 0,
                'modified': datetime.fromtimestamp(os.path.getmtime(latest))
            }
        else:
            status[pattern] = {'exists': False}
    
    return status

def monitor_once():
    """1回監視"""
    print(f"正規PatchTST学習監視 - {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)
    
    # GPU状況
    gpu_status = get_gpu_status()
    if gpu_status:
        print(f"GPU使用率: {gpu_status['gpu_util']:3d}%")
        print(f"VRAM: {gpu_status['mem_used']:,}MB / {gpu_status['mem_total']:,}MB " +
              f"({gpu_status['mem_used']/gpu_status['mem_total']*100:.1f}%)")
        print(f"温度: {gpu_status['temperature']:2d}°C")
        print(f"電力: {gpu_status['power']:5.1f}W")
        
        if gpu_status['gpu_util'] >= 90:
            print("状態: 学習中（高負荷）")
        elif gpu_status['gpu_util'] >= 30:
            print("状態: 学習中（中負荷）")
        else:
            print("状態: アイドル/完了")
    else:
        print("GPU情報取得不可")
    
    print("-"*60)
    
    # プロセス情報
    processes = get_training_processes()
    if processes:
        print(f"学習プロセス: {len(processes)}個")
        for proc in processes:
            elapsed = datetime.now() - proc['start_time']
            hours = int(elapsed.total_seconds() // 3600)
            minutes = int((elapsed.total_seconds() % 3600) // 60)
            print(f"  PID {proc['pid']}: {proc['memory_gb']:.1f}GB, 経過 {hours:02d}:{minutes:02d}")
    else:
        print("学習プロセス: なし")
    
    print("-"*60)
    
    # モデル進捗
    model_status = check_model_progress()
    
    checkpoint_status = model_status.get("models/official_patchtst/checkpoints/best_model.pth", {})
    if checkpoint_status.get('exists'):
        print(f"[OK] モデル保存済み ({checkpoint_status['size']:.1f}MB)")
        print(f"  最終更新: {checkpoint_status['modified'].strftime('%H:%M:%S')}")
    else:
        print("[--] モデル未保存")
    
    log_status = model_status.get("models/official_patchtst/logs/events.out.tfevents.*", {})
    if log_status.get('exists'):
        print(f"[OK] TensorBoardログ更新中")
        print(f"  最終更新: {log_status['modified'].strftime('%H:%M:%S')}")
    else:
        print("[--] TensorBoardログなし")
    
    print("="*60)

def monitor_live():
    """リアルタイム監視"""
    print("正規PatchTST学習リアルタイム監視開始")
    print("Ctrl+C で終了")
    print()
    
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            monitor_once()
            print("5秒後に更新...")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n監視終了")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--live':
        monitor_live()
    else:
        monitor_once()