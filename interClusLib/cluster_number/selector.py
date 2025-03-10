"""
包含各种确定最佳聚类数量的方法
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class ClusterNumberSelector:
    """
    用于确定聚类数量的类，实现多种常用方法
    """
    def __init__(self, min_clusters=2, max_clusters=20):
        """
        初始化聚类数量选择器
        
        参数:
        min_clusters: int, 最小聚类数
        max_clusters: int, 最大聚类数
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.eval_results = None
        self.optimal_k = None
    
    def l_method(self, eval_data, iterative=True):
        """
        使用L方法确定最佳聚类数量
        
        参数:
        eval_data: 字典或数组，包含聚类数量和对应的评估指标值
        iterative: bool, 是否使用迭代L方法，默认为True
        
        返回:
        int: 最佳聚类数量
        """
        # 转换输入数据格式
        if isinstance(eval_data, dict):
            data = np.array([(k, v) for k, v in sorted(eval_data.items())])
        else:
            data = np.array(eval_data)
            
        # 确保数据按聚类数排序
        data = data[data[:, 0].argsort()]
        
        # 应用聚类数限制
        mask = (data[:, 0] >= self.min_clusters) & (data[:, 0] <= self.max_clusters)
        data = data[mask]
        
        if len(data) < 3:
            print("警告: 数据点不足，无法应用L方法")
            return self.min_clusters
        
        if iterative:
            self.optimal_k = self._iterative_l_method(data)
        else:
            self.optimal_k = self._standard_l_method(data)
        
        self.eval_results = data
        return self.optimal_k
    
    def _standard_l_method(self, data):
        """
        标准L方法实现
        
        参数:
        data: 排序后的评估数据，形状为(n, 2)
        
        返回:
        int: 最佳聚类数量
        """
        n = len(data)
        min_rmse = float('inf')
        knee_idx = 0
        
        # 遍历所有可能的拐点位置
        for c in range(1, n-1):
            # 左侧序列 Lc
            Lc = data[:c+1]
            # 右侧序列 Rc
            Rc = data[c+1:]
            
            if len(Lc) < 2 or len(Rc) < 2:
                continue
            
            # 左侧线性回归
            X_left = Lc[:, 0].reshape(-1, 1)
            y_left = Lc[:, 1]
            model_left = LinearRegression().fit(X_left, y_left)
            y_pred_left = model_left.predict(X_left)
            rmse_left = np.sqrt(np.mean((y_left - y_pred_left) ** 2))
            
            # 右侧线性回归
            X_right = Rc[:, 0].reshape(-1, 1)
            y_right = Rc[:, 1]
            model_right = LinearRegression().fit(X_right, y_right)
            y_pred_right = model_right.predict(X_right)
            rmse_right = np.sqrt(np.mean((y_right - y_pred_right) ** 2))
            
            # 计算加权总RMSE
            w_left = len(Lc) / float(n)
            w_right = len(Rc) / float(n)
            total_rmse = w_left * rmse_left + w_right * rmse_right
            
            if total_rmse < min_rmse:
                min_rmse = total_rmse
                knee_idx = c
        
        # 返回拐点对应的聚类数量
        return int(data[knee_idx, 0])
    
    def _iterative_l_method(self, data):
        """
        迭代改进版L方法实现
        
        参数:
        data: 排序后的评估数据，形状为(n, 2)
        
        返回:
        int: 最佳聚类数量
        """
        # 初始化
        last_knee = len(data)
        current_knee = len(data)
        cutoff = None
        
        # 迭代直到收敛
        while True:
            # 如果没有指定cutoff，则使用全部数据
            if cutoff is None:
                focus_data = data
            else:
                # 只关注cutoff以内的数据点
                focus_data = data[data[:, 0] <= cutoff]
                
                # 确保我们有足够的数据点
                if len(focus_data) < 5:  # 确保至少有5个点
                    focus_data = data[:min(5, len(data))]
            
            # 使用标准L方法找到当前关注区域的拐点
            knee_idx = 0
            min_rmse = float('inf')
            n = len(focus_data)
            
            for c in range(1, n-1):
                Lc = focus_data[:c+1]
                Rc = focus_data[c+1:]
                
                if len(Lc) < 2 or len(Rc) < 2:
                    continue
                
                X_left = Lc[:, 0].reshape(-1, 1)
                y_left = Lc[:, 1]
                model_left = LinearRegression().fit(X_left, y_left)
                y_pred_left = model_left.predict(X_left)
                rmse_left = np.sqrt(np.mean((y_left - y_pred_left) ** 2))
                
                X_right = Rc[:, 0].reshape(-1, 1)
                y_right = Rc[:, 1]
                model_right = LinearRegression().fit(X_right, y_right)
                y_pred_right = model_right.predict(X_right)
                rmse_right = np.sqrt(np.mean((y_right - y_pred_right) ** 2))
                
                w_left = len(Lc) / float(n)
                w_right = len(Rc) / float(n)
                total_rmse = w_left * rmse_left + w_right * rmse_right
                
                if total_rmse < min_rmse:
                    min_rmse = total_rmse
                    knee_idx = c
            
            current_knee = int(focus_data[knee_idx, 0])
            
            # 如果拐点不再向左移动或者达到最小聚类数，则收敛
            if current_knee >= last_knee or current_knee <= self.min_clusters:
                break
            
            # 更新结果，并设置新的cutoff（拐点的2倍）
            last_knee = current_knee
            cutoff = current_knee * 2
        
        return current_knee

    def elbow_method(self, eval_data, convex=False):
        """
        使用肘部法则确定最佳聚类数量
        
        参数:
        eval_data: 评估数据，格式同l_method
        convex: bool, 是否寻找凸拐点而不是凹拐点
        
        返回:
        int: 最佳聚类数量
        """
        # 转换输入数据
        if isinstance(eval_data, dict):
            data = np.array([(k, v) for k, v in sorted(eval_data.items())])
        else:
            data = np.array(eval_data)
            
        # 确保数据按聚类数排序
        data = data[data[:, 0].argsort()]
        
        # 应用聚类数限制
        mask = (data[:, 0] >= self.min_clusters) & (data[:, 0] <= self.max_clusters)
        data = data[mask]
        
        if len(data) <= 2:
            return self.min_clusters
        
        x = data[:, 0]
        y = data[:, 1]
        
        # 计算相邻点之间的一阶差分
        diffs = np.diff(y) / np.diff(x)
        
        # 计算二阶差分（拐点的指示器）
        second_diffs = np.diff(diffs)
        
        # 获取最大/最小二阶差分的索引，取决于是否寻找凸拐点
        if convex:
            # 寻找最大正二阶差分（凸拐点）
            idx = np.argmax(second_diffs) + 1
        else:
            # 寻找最小负二阶差分（凹拐点）
            idx = np.argmin(second_diffs) + 1
        
        self.eval_results = data
        self.optimal_k = int(x[idx])
        return self.optimal_k
    
    def optimize_metric(self, eval_data, maximize=True):
        """
        找到评估指标的最优值（最大或最小）
        
        参数:
        eval_data: 评估数据，格式同l_method
        maximize: bool, 是否最大化而不是最小化评估指标
        
        返回:
        int: 最佳聚类数量
        """
        # 转换输入数据
        if isinstance(eval_data, dict):
            data = np.array([(k, v) for k, v in sorted(eval_data.items())])
        else:
            data = np.array(eval_data)
            
        # 确保数据按聚类数排序
        data = data[data[:, 0].argsort()]
        
        # 应用聚类数限制
        mask = (data[:, 0] >= self.min_clusters) & (data[:, 0] <= self.max_clusters)
        data = data[mask]
        
        # 寻找最优值对应的聚类数
        if maximize:
            idx = np.argmax(data[:, 1])
        else:
            idx = np.argmin(data[:, 1])
        
        self.eval_results = data
        self.optimal_k = int(data[idx, 0])
        return self.optimal_k
    
    def plot_evaluation(self, title=None, show_optimal=True, figsize=(10, 6), 
                        xlabel='聚类数量', ylabel='评估指标值'):
        """
        绘制评估图和确定的最佳聚类数
        
        参数:
        title: str, 图表标题
        show_optimal: bool, 是否显示最佳聚类数
        figsize: tuple, 图形大小
        xlabel: str, x轴标签
        ylabel: str, y轴标签
        
        返回:
        plt: matplotlib图形对象
        """
        if self.eval_results is None:
            raise ValueError("必须先运行方法确定最佳聚类数")
        
        plt.figure(figsize=figsize)
        
        x = self.eval_results[:, 0]
        y = self.eval_results[:, 1]
        
        plt.plot(x, y, 'bo-', linewidth=2, markersize=8)
        plt.grid(True, alpha=0.3)
        
        if show_optimal and self.optimal_k is not None:
            plt.axvline(x=self.optimal_k, color='red', linestyle='--', 
                        label=f'最佳聚类数 (k={self.optimal_k})')
            plt.legend()
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title or '确定最佳聚类数量')
        
        return plt


# 独立函数版本，方便直接调用
def l_method(eval_data, min_clusters=2, max_clusters=20, iterative=True):
    """
    L方法独立函数版本
    
    参数:
    eval_data: 评估数据
    min_clusters: int, 最小聚类数
    max_clusters: int, 最大聚类数
    iterative: bool, 是否使用迭代L方法
    
    返回:
    int: 最佳聚类数量
    """
    selector = ClusterNumberSelector(min_clusters, max_clusters)
    return selector.l_method(eval_data, iterative)

def elbow_method(eval_data, min_clusters=2, max_clusters=20, convex=False):
    """
    肘部法则独立函数版本
    
    参数:
    eval_data: 评估数据
    min_clusters: int, 最小聚类数
    max_clusters: int, 最大聚类数
    convex: bool, 是否寻找凸拐点
    
    返回:
    int: 最佳聚类数量
    """
    selector = ClusterNumberSelector(min_clusters, max_clusters)
    return selector.elbow_method(eval_data, convex)

def optimize_metric(eval_data, min_clusters=2, max_clusters=20, maximize=True):
    """
    优化指标独立函数版本
    
    参数:
    eval_data: 评估数据
    min_clusters: int, 最小聚类数
    max_clusters: int, 最大聚类数
    maximize: bool, 是否最大化指标
    
    返回:
    int: 最佳聚类数量
    """
    selector = ClusterNumberSelector(min_clusters, max_clusters)
    return selector.optimize_metric(eval_data, maximize)