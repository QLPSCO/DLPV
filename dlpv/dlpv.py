import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import subprocess
import os
import psutil
sb.set_style("darkgrid")

log_path = "dlpv.log"

def record(log_path=log_path, timeout=50):
    subprocess.Popen("timeout " + timeout + " nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv -l 1 | sed s/%//g > ./" + log_path,shell=True)
    pid = os.getpid()
    base = psutil.cpu_percent()/psutil.cpu_count()
    psutil.virtual_memory()

def visualize(log_path=log_path, cpu=True, gpu=True, pci=True, ):
    gpu = pd.read_csv(log_path)
    gpu.plot()
    plt.show()


