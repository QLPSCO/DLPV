import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style("darkgrid")

gpu = pd.read_csv("./5_gradient_and_parallel.log")   
gpu.plot()
plt.show()
