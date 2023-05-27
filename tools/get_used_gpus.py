import subprocess
import time

result = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv'], stdout=subprocess.PIPE)
gpu_count = int(result.stdout.decode('utf-8').split('\n')[1])
print(f"Total GPU count: {gpu_count}")

while True:
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv'], stdout=subprocess.PIPE)
    pid_list = result.stdout.decode('utf-8').split('\n')[1:-1]
    used_gpu_count = len(pid_list)
    print(f"Used GPU count: {used_gpu_count}/{gpu_count}")

    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv'],
                            stdout=subprocess.PIPE)
    gpu_utilization = result.stdout.decode('utf-8').split('\n')[1:-1]
    for i, util in enumerate(gpu_utilization):
        gpu_util, gpu_mem, gpu_total = util.strip().split(',')
        print(f"GPU {i}: {gpu_util}% (used memory: {gpu_mem.strip()}, total memory：{gpu_total.strip()})")
    print("=" * 100)
    print("\n")
    time.sleep(1)