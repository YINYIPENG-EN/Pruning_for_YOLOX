
import numpy as np
import matplotlib.pyplot as plt

# 卷积层的可视化
from IPython import embed


def NV(x, save_path=''):
    y = x.mean([2, 3])
    y = y.cpu().numpy()
    index = np.argmax(y)  # 可以计算得出哪个通道均值最大
    feature = x.cpu().numpy()
    plt.imshow(feature[0, index, :, :], cmap='viridis')
    plt.axis('off')
    plt.savefig(save_path + '特征图.jpg', bbox_inches='tight', pad_inches=-0.1)  # 注意两个参数
    plt.show()
