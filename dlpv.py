import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import subprocess
import os
import psutil
sb.set_style("darkgrid")

log_path = "dlpv.log"

def dlpv_trace(log_path=log_path, timeout=50, optimization=None):
    subprocess.Popen("timeout " + str(timeout) + " nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv -l 1 | sed s/%//g > ./" + log_path,shell=True)
    
    if optimization == "gradient_checkpointing":
        tf.__dict__["gradients"] = gradients_memory
        K.__dict__["gradients"] = gradients_memory
        print("Applying gradient checkpoint optimization and benchmarking...")    

    if optimization == "multithreaded_queue":
        Model.fit_generator.setattr("Model.fit_generator.__defaults__", (None, 1, 1, None, None, None, None, 1000, 2, True, True, 0))
        print("Applying multithreaded queue optimization and benchmarking...")


    #pid = os.getpid()
    #base = psutil.cpu_percent()/psutil.cpu_count()
    #psutil.virtual_memory()

def dlpv_visualize(log_path=log_path, cpu=True, gpu=True, pci=True, ):
    gpu = pd.read_csv(log_path)
    gpu.plot()
    plt.show()


