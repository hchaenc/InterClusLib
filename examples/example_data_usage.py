import numpy as np
import pandas as pd
import sys
import os

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加 `interClusLib` 的上级目录到 Python 路径
sys.path.append(os.path.join(current_dir, ".."))
import interClusLib
from interClusLib import IntervalData

print(interClusLib.__file__)

random_data = IntervalData.random_data(10,3)
print("\n随机生成的区间数据:")
print(random_data.data)
random_data.summary()