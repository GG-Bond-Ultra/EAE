import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter, FixedLocator

def forward(x):
    return x**(1/3)
 
 
def inverse(x):
    return x**3


def plot_exit_flops_acc(CIFAR10_AGX, NWPU_AGX, UCM_AGX, 
                        CIFAR10_AGX_CPU, NWPU_AGX_CPU, UCM_AGX_CPU, 
                        CIFAR10_Nano, NWPU_Nano, UCM_Nano,model, ylabel, title):
    fig, ax2 = plt.subplots(figsize=(10,6))
    ax2.set_xlabel("Model Type")
    color2 = 'black'
    ax2.set_ylabel(ylabel, color=color2)
    ax2.plot(model, CIFAR10_AGX, 'g^-', label="CIFAR10_AGX", zorder=2, linewidth=2,markersize=13, alpha=0.5)
    ax2.plot(model, NWPU_AGX, 'r^-', label="NWPU_AGX", zorder=2, linewidth=2,markersize=13, alpha=0.5)
    ax2.plot(model, UCM_AGX, 'y^-', label="UCM_AGX", zorder=2, linewidth=2,markersize=13, alpha=0.5)
    ax2.plot(model, CIFAR10_Nano, 'c*--', label="CIFAR10_Nano", zorder=2, linewidth=2,markersize=13, alpha=0.5)
    ax2.plot(model, NWPU_Nano, 'm*--', label="NWPU_Nano", zorder=2, linewidth=2,markersize=13, alpha=0.5)
    ax2.plot(model, UCM_Nano, 'b*--', label="UCM_Nano", zorder=2, linewidth=2,markersize=13, alpha=0.5)
    ax2.plot(model, CIFAR10_AGX_CPU, 'gd-', label="CIFAR10_AGX_CPU", zorder=2, linewidth=2,markersize=13, alpha=0.5)
    ax2.plot(model, NWPU_AGX_CPU, 'rd-', label="NWPU_AGX_CPU", zorder=2, linewidth=2,markersize=13, alpha=0.5)
    ax2.plot(model, UCM_AGX_CPU, 'yd-', label="UCM_AGX_CPU", zorder=2, linewidth=2,markersize=13, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_yscale("function",functions=(forward, inverse))
    ax2.yaxis.set_major_locator(FixedLocator((0,40,160)))
    ax2.set_ylim(13, 160)
    # ax2.set_ylim(30, 1100)
    y_line1 = 51
    ax2.axhline(y=51, color='gray', linestyle='--', linewidth=0.5)
    ax2.text(ax2.get_xlim()[0], y_line1, f'y={y_line1}', color='gray', verticalalignment='bottom', horizontalalignment='left')
    ax2.axhline(y=15, color='gray', linestyle='--', linewidth=0.5)
    y_line2 = 15
    ax2.text(ax2.get_xlim()[0], y_line2, f'y={y_line2}', color='gray', verticalalignment='bottom', horizontalalignment='left')
    ax2.grid(True, linestyle='--', color='gray', linewidth=0.3)
    # 添加图例
    fig.legend(loc="upper right", bbox_to_anchor=(0.8,0.85))

    # 添加标题
    plt.title(title)

    # 保存图像
    plt.savefig(f"/home/nvidia/heShaoWei/GFNet/dynn/IMAGE/{ylabel}.png")

    # 显示图表
    plt.show()


def plot_scatter(x, y, x1, y1, x2, y2):
    """
    绘制散点图的函数

    参数:
    x (list): X轴数据
    y (list): Y轴数据
    title (str): 图表标题
    xlabel (str): X轴标签
    ylabel (str): Y轴标签
    color (str): 点的颜色
    marker (str): 点的形状
    markersize (int): 点的大小
    """
    title='Latency and Energy Consumption',
    xlabel='Latency'
    ylabel='Energy Consumption'
    fig, ax = plt.subplots(figsize=(6, 4))  # 创建图形和轴对象，并设置图形大小
    ax.scatter(x[0], y[0], c='darkorange', marker='s', s=250, alpha=0.3)
    ax.scatter(x[1], y[1], c='green', marker='s', s=250, alpha=0.3)
    ax.scatter(x[2], y[2], c='red', marker='s', s=250, alpha=0.3)
    ax.scatter(x1[0], y1[0], c='darkorange', marker='o', s=250, alpha=0.3)
    ax.scatter(x1[1], y1[1], c='green', marker='o', s=250, alpha=0.3)
    ax.scatter(x1[2], y1[2], c='red', marker='o', s=250, alpha=0.3)
    ax.scatter(x2[0], y2[0], c='darkorange', marker='^', s=250, alpha=0.3)
    ax.scatter(x2[1], y2[1], c='green', marker='^', s=250, alpha=0.3)
    ax.scatter(x2[2], y2[2], c='red', marker='^', s=250, alpha=0.3)
    ax.set_yscale("function",functions=(forward, inverse))
    ax.yaxis.set_major_locator(FixedLocator((0,60,150)))
    ax.set_title(title)  # 设置标题
    ax.set_xlabel(xlabel)  # 设置X轴标签
    ax.set_ylabel(ylabel)  # 设置Y轴标签
    # plt.grid(True)  # 显示网格
    plt.savefig(f"/home/nvidia/heShaoWei/GFNet/dynn/IMAGE/scatter.png")
    plt.show()  # 显示图形


import matplotlib.pyplot as plt
import numpy as np

# 这两行代码解决 plt 中文显示的问题


def plot_drink_survey(dataset, GFNet, distil, dynn):
    """
    绘制购买饮用水情况调查结果的条形图

    参数:
    waters (tuple): 饮料名称
    buy_number_male (list): 男性购买量
    buy_number_female (list): 女性购买量
    bar_width (float): 条形图宽度
    """
    index = np.arange(len(dataset)) 
    bar_width = 0.3
    # 使用两次 bar 函数画出两组条形图
    plt.bar(index, height=GFNet, width=bar_width, color='#f0833a', label='GFNet-12-384',alpha=0.8)
    plt.bar(index + bar_width, height=distil, width=bar_width, color='#b36ff6', label='GFNet-distil-12-192',alpha=0.8)
    plt.bar(index + 2 * bar_width, height=dynn, width=bar_width, color='#b5ce08', label='GFNet-dynn-12-192',alpha=0.8)

    plt.legend()  # 显示图例
    plt.xticks(index + bar_width ,dataset)  # 让横坐标轴刻度显示 waters 里的饮用水名称
    plt.title('Params of Model')
    plt.xlabel('Datesets')
    plt.ylabel('Params of Model(M)')
    plt.savefig(f"/home/nvidia/heShaoWei/GFNet/dynn/IMAGE/Params.png")
    plt.show()


def plot_acc_CIFAR10_NWPU_UCM(CIFAR10, NWPU, UCM,model, ylabel, title):
    plt.rcParams.update({'font.size': 18})
    fig, ax2 = plt.subplots(figsize=(10,8))
    ax2.set_xlabel("Number of Layer")
    color2 = 'black'
    ax2.set_ylabel(ylabel, color=color2)
    ax2.plot(model, CIFAR10, marker='*', label="GFDE_CIFAR10",linestyle='None',markersize=25, alpha=0.9)
    ax2.plot(model, NWPU, marker='*', label="GFDE_NWPU",linestyle='None',markersize=25, alpha=0.9)
    ax2.plot(model, UCM, marker='*', label="GFDE_UCM", linestyle='None',markersize=25, alpha=0.9)
    # ax1.plot(index, accuracy_data[i], marker='*', label=f'{labels[i]} Accuracy', linestyle='None',markersize=15)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # ax2.set_ylim(20, 110)
   
    # ax2.grid(True, linestyle='--', color='gray', linewidth=0.3)
    # 添加图例
    fig.legend(loc="upper right", bbox_to_anchor=(0.4,0.85))

    # 添加标题
    plt.title(title)

    # 保存图像
    plt.savefig(f"/home/nvidia/heShaoWei/GFNet/dynn/IMAGE/every_classifier_acc.png")

    # 显示图表
    plt.show()


def plot_exit_and_accuracy(exit_data, accuracy_data, labels, bar_width=0.2, title='Exit Values and Accuracy', xlabel='Number of Layers', ylabel_bar='Exit rate (%)', ylabel_line='Accuracy (%)'):
    """
    绘制并列柱状图和折线图的函数

    参数:
    exit_data (list of lists): 包含每组数据的退出值 (例如 [[exit_CIFAR10], [exit_NWPU], [exit_UCM]])
    accuracy_data (list of lists): 包含每组数据的准确率 (例如 [[exit_acc_CIFAR10], [exit_acc_NWPU], [exit_acc_UCM]])
    labels (list of str): 每组数据的标签 (例如 ['CIFAR10', 'NWPU', 'UCM'])
    bar_width (float): 条形图宽度
    title (str): 图形标题
    xlabel (str): X轴标签
    ylabel_bar (str): 纵坐标轴标题（条形图）
    ylabel_line (str): 纵坐标轴标题（折线图）
    """
    # 数据处理
    plt.rcParams.update({'font.size': 20})
    num_datasets = len(exit_data)
    index = np.arange(len(exit_data[0]))

    # 创建图形和轴对象
    fig, ax2 = plt.subplots(figsize=(10, 8))

    # 绘制柱状图
    for i in range(num_datasets):
        ax2.bar(index + i * bar_width - (num_datasets - 1) * bar_width / 2, exit_data[i], bar_width, label=f'{labels[i]} Exit rate', alpha=1)

    # 创建第二个 Y 轴用于绘制折线图
    ax1 = ax2.twinx()
    marker=['o', 's', '^']
    # 绘制折线图
    for i in range(num_datasets):
        # 过滤掉值为0的数据点，但保留横坐标
        accuracy_array = np.array(accuracy_data[i])
        valid_index = accuracy_array != 0
        filtered_index = index[valid_index]
        filtered_data = accuracy_array[valid_index]

        # 绘制所有点
        ax1.plot(index, accuracy_array, linestyle='None', color=f'C{i}', zorder=2)  # 绘制所有点的底图
        ax1.scatter(filtered_index, filtered_data,  label=f'{labels[i]} Accuracy', marker=marker[i], alpha=0.85, s=300, color=f'C{i}', zorder=3)  # 仅绘制非零点的标记
    # for i in range(num_datasets):
    #     ax1.plot(index, accuracy_data[i], marker='*', label=f'{labels[i]} Accuracy', linestyle='None',markersize=15)
    ax1.set_ylim(0, 110)
    # 设置标签、标题和图例
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel_bar, color='black')
    ax1.set_ylabel(ylabel_line, color='black')
    # ax2.set_title("Exit rate on every point and Accuracy on exited points")
    ax2.legend(loc='upper left', bbox_to_anchor=(0,0.6))#
    ax1.legend(loc='upper left', bbox_to_anchor=(0,0.85))#
    plt.savefig(f"/home/nvidia/heShaoWei/GFNet/dynn/IMAGE/every_classifier_exit-rate_acc333.png")
    # 显示图形
    plt.show()



Params_CIFAR10 = [15.6, 4.25, 4.27]
Params_NWPU = [15.93,4.56, 4.75]
Params_UCM = [15.92,4.55, 4.64]
GFNet = [15.6, 15.93, 15.92]
distil = [4.25, 4.56, 4.55]
dynn = [4.29, 4.75, 4.64]
dataset = ['CIFAR10', 'NWPU', 'UCM']
# 调用函数绘制条形图
# plot_drink_survey(dataset, GFNet, distil, dynn)


# 示例数据
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

# 使用函数绘制散点图


import matplotlib.pyplot as plt
import numpy as np

def plot_energy_consumption(model, device,energy_consumption_CIFAR10, energy_consumption_NWPU, energy_consumption_UCM):
    # 横坐标的模型名称
    # model = ['GFNet-12-384', 'GFNet-distil-12-192', 'GFNet-dynn-12-192']
    plt.rcParams.update({'font.size': 18})
    # 各数据集的能耗数据
    # energy_consumption_CIFAR10 = [136.81, 59.18, 47.92]
    # energy_consumption_NWPU = [139.10, 58.62, 49.31]
    # energy_consumption_UCM = [137.17, 52.79, 42.37]
    
    # 设置柱状图的宽度
    bar_width = 0.25
    
    # 设置横坐标的位置
    r1 = np.arange(len(model))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # 创建图表
    plt.figure(figsize=(10, 8))
    
    # 绘制柱状图
    plt.bar(r1, energy_consumption_CIFAR10, color=f'C{0}', width=bar_width, label='CIFAR10')
    plt.bar(r2, energy_consumption_NWPU, color=f'C{1}', width=bar_width, label='NWPU')
    plt.bar(r3, energy_consumption_UCM, color=f'C{2}', width=bar_width, label='UCM')
    
    # 添加标签和标题
    plt.xlabel('Model')
    plt.ylabel('Energy Consumption (J)')
    plt.title(f'Energy Consumption on {device}')
    
    # 添加模型名称作为横坐标
    plt.xticks([r + bar_width for r in range(len(model))], model)
    
    # 添加图例
    plt.legend()
    # plt.savefig(f"/home/nvidia/heShaoWei/GFNet/dynn/IMAGE/AGX_energy.png")
    plt.savefig(f"/home/nvidia/heShaoWei/GFNet/dynn/IMAGE/Nano_energy.png")
    # 显示图表
    plt.show()

# 调用函数绘制图表
def plot_Latency(model, device,latency_CIFAR10, latency_NWPU, latency_UCM):
    plt.rcParams.update({'font.size': 18})
    # 设置柱状图的宽度
    bar_width = 0.25
    
    # 设置横坐标的位置
    r1 = np.arange(len(model))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # 创建图表
    plt.figure(figsize=(10, 8))
    
    # 绘制柱状图
    plt.bar(r1, latency_CIFAR10, color=f'C{3}', width=bar_width, label='CIFAR10')
    plt.bar(r2, latency_NWPU, color=f'C{4}', width=bar_width, label='NWPU')
    plt.bar(r3, latency_UCM, color=f'C{5}', width=bar_width, label='UCM')
    
    # 添加标签和标题
    plt.xlabel('Model')
    plt.ylabel('Latency (ms)')
    plt.title(f'Latency on {device}')
    
    # 添加模型名称作为横坐标
    plt.xticks([r + bar_width for r in range(len(model))], model)
    
    # 添加图例
    plt.legend()
    # plt.savefig(f"/home/nvidia/heShaoWei/GFNet/dynn/IMAGE/AGX_latency.png")
    plt.savefig(f"/home/nvidia/heShaoWei/GFNet/dynn/IMAGE/Nano_latency.png")
    # 显示图表
    plt.show()



model = ['GFNet-12-384','GFNet-distil-12-192','GFNet-dynn-12-192']
acc = [98.47, 97.10, 96.06]
Latency = [26.31, 16.44, 15.97]

Energy_consumption_CIFAR10_AGX_GPU = [136.81, 59.18, 47.92] 
Energy_consumption_NWPU_AGX_GPU = [139.10, 58.62, 49.31]
Energy_consumption_UCM_AGX_GPU = [137.17, 52.79, 42.37]
device_AGX = 'Orin AGX'
Energy_consumption_CIFAR10_AGX_CPU = [842.19, 360.22, 310.08] 
Energy_consumption_NWPU_AGX_CPU = [1013.16, 513.72, 465.60]
Energy_consumption_UCM_AGX_CPU = [961.09, 472.92, 389.14]

Energy_consumption_CIFAR10_Nano_GPU = [135.04, 54.3, 48.32] 
Energy_consumption_NWPU_Nano_GPU = [134.47, 52.96, 38.95]
Energy_consumption_UCM_Nano_GPU = [134.56, 52.65, 43.80]
device_Nano = 'Nano Orin'
# plot_energy_consumption(model, device_Nano ,Energy_consumption_CIFAR10_Nano_GPU, 
#                         Energy_consumption_NWPU_Nano_GPU ,Energy_consumption_UCM_Nano_GPU)
# plot_exit_flops_acc(Energy_consumption_CIFAR10_AGX_GPU,
#                     Energy_consumption_NWPU_AGX_GPU,
#                     Energy_consumption_UCM_AGX_GPU,
#                     Energy_consumption_CIFAR10_AGX_CPU,
#                     Energy_consumption_NWPU_AGX_CPU,
#                     Energy_consumption_UCM_AGX_CPU,
#                     Energy_consumption_CIFAR10_Nano_GPU,
#                     Energy_consumption_NWPU_Nano_GPU, 
#                     Energy_consumption_UCM_Nano_GPU, model=model,
#                     ylabel='Energy_Consumption(mJ)',
#                     title='Inference energy consumption in three datasets')

Params_CIFAR10 = [15.6, 4.25, 4.27]
Params_NWPU = [15.93,4.56, 4.75]
Params_UCM = [15.92,4.55, 4.64]

latency_CIFAR10_AGX_GPU = [26.31, 16.44, 15.97] 
latency_NWPU_AGX_GPU = [26.75, 16.75, 14.09]
latency_UCM_AGX_GPU = [25.88, 15.08, 14.12]

latency_CIFAR10_Nano_GPU = [33.76, 18.10, 18.58] 
latency_NWPU_Nano_GPU = [32.01, 17.08, 17.70]
latency_UCM_Nano_GPU = [32.04, 17.55, 16.84]

latency_CIFAR10_AGX_CPU = [125.7, 58.1, 51.68] 
latency_NWPU_AGX_CPU = [153.51, 85.62, 77.60]
latency_UCM_AGX_CPU = [145.62, 84.45, 69.49]
# plot_Latency(model, device_Nano ,latency_CIFAR10_Nano_GPU, 
#                         latency_NWPU_Nano_GPU,latency_UCM_Nano_GPU)
# plot_exit_flops_acc(latency_CIFAR10_AGX_GPU,
#                     latency_NWPU_AGX_GPU,
#                     latency_UCM_AGX_GPU,
#                     latency_CIFAR10_AGX_CPU,
#                     latency_NWPU_AGX_CPU,
#                     latency_UCM_AGX_CPU,
#                     latency_CIFAR10_Nano_GPU,
#                     latency_NWPU_Nano_GPU,
#                     latency_UCM_Nano_GPU,model=model,
#                     ylabel='Inference_Latency(ms)',
#                     title='Inference latency in three datasets')

# plot_scatter(latency_CIFAR10_AGX_GPU, Energy_consumption_CIFAR10_AGX_GPU,
#              latency_NWPU_AGX_GPU, Energy_consumption_NWPU_AGX_GPU,
#              latency_UCM_AGX_GPU, Energy_consumption_UCM_AGX_GPU)

acc_CIFAR10 = acc_layer_val = [
    33.43,
    34.05,
    36.98,
    40.32,
    41.23,
    55.08,
    60.72,
    66.06,
    73.75,
    83.88,
    86.94,
    97.1
]

acc_NWPU = [
    25.047619047619047,
    29.619047619047617,
    31.238095238095237,
    31.603174603174605,
    39.12698412698413,
    49.142857142857146,
    51.587301587301596,
    65.80952380952381,
    78.93650793650794,
    89.52380952380953,
    93.12698412698413,
    94.96825396825398
]
acc_UCM = [
    26.666666666666668,
    32.142857142857146,
    35.23809523809524,
    41.904761904761905,
    49.047619047619044,
    56.19047619047619,
    66.42857142857143,
    79.52380952380952,
    92.85714285714286,
    95.95238095238095,
    97.61904761904762,
    98.09523809523809
]
m = [0,1,2,3,4,5,6,7,8,9,10,11]
# plot_acc_CIFAR10_NWPU_UCM(CIFAR10=acc_CIFAR10, NWPU=acc_NWPU, UCM=acc_UCM,model=m, ylabel='Accuracy (%)', title="Accuracy on all points")


exit_CIFAR10 = [ #threshold=0.55
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    9.0600004196167,
    0.1899999976158142,
    18.729999542236328,
    32.34000015258789,
    17.65999984741211,
    22.020000457763672
]

exit_acc_CIFAR10 = [
    0,
    0,
    0,
    0,
    0,
    0,
    96.90949249267578,
    100.0,
    98.13134002685547,
    97.55720520019531,
    89.24122619628906,
    91.4168930053711
]

exit_NWPU = [ #threshold=0.45
    0.0,
    0.8253968358039856,
    0.2857142984867096,
    0.0,
    0.1269841343164444,
    3.9206349849700928,
    1.8571428060531616,
    17.904762268066406,
    13.984127044677734,
    28.507936477661133,
    20.73015785217285,
    11.857142448425293
]

exit_acc_NWPU = [
    0,
    100.0,
    100.0,
    0,
    100.0,
    95.1417007446289,
    98.29059600830078,
    95.83333587646484,
    99.0919418334961,
    95.9910888671875,
    92.11331939697266,
    77.9116439819336
]

exit_UCM = [ #threshold=0.65
    0.0,
    0.0,
    0.0,
    0.0,
    3.3333332538604736,
    8.809523582458496,
    11.904762268066406,
    17.85714340209961,
    27.14285659790039,
    25.714284896850586,
    5.238095283508301,
    0.0
]

exit_acc_UCM = [
    0,
    0,
    0,
    0,
    100.0,
    97.29729461669922,
    100.0,
    100.0,
    99.12281036376953,
    93.51851654052734,
    72.7272720336914,
    0
]

plot_exit_and_accuracy(
    exit_data=[exit_CIFAR10, exit_NWPU, exit_UCM],
    accuracy_data=[exit_acc_CIFAR10, exit_acc_NWPU, exit_acc_UCM],
    labels=['CIFAR10', 'RESISC', 'UCM']
)
# ax2.bar(index + i * bar_width - (num_datasets - 1) * bar_width / 2, exit_data[i], bar_width, label=f'{labels[i]} Exit rate', alpha=0.7)
def plot_bars_and_scatter(CIFAR10, real_exit_rate_CIFAR10, real_exit_acc_CIFAR10,labell,acc):
    plt.rcParams.update({'font.size': 30})
    fig, ax1 = plt.subplots(figsize=(10, 10))
    index = np.arange(len(CIFAR10)) 
    # 绘制柱状图
    bar_width = 0.3
    ax1.bar(index, real_exit_rate_CIFAR10, bar_width, color='#4AA1EE', label='Exit Rate')
    ax1.set_xlabel('Number of Layer')
    # ax1.set_ylabel('Exit Rate (%)')
    ax1.tick_params(axis='y')
    # ax1.set_xticks()  # 设置x轴刻度
    if labell == 'NWPU':
        ax1.set_ylim(0, 70) #NWPU
        ax1.set_xticks(CIFAR10)
    elif labell =='CIFAR10':
        ax1.set_ylim(0, 50) #CIFAR10
        ax1.set_xticks(CIFAR10)
    elif labell == 'UCM':
        ax1.set_ylim(0, 60) #UCM
    # 在同一图上绘制散点图
    ax2 = ax1.twinx()
    ax2.bar(index + bar_width, real_exit_acc_CIFAR10, bar_width, color='#D16BA5', label='Exit Accuracy')
    # ax2.scatter(CIFAR10, real_exit_acc_CIFAR10, color='#D16BA5',marker='*', label='Exit Accuracy',s=300, zorder=3)
    # ax2.set_ylabel('Exit Accuracy (%)')
    plt.xticks(index + bar_width ,CIFAR10)
    # ax1.set_xticks(index + bar_width)
    # ax1.set_xticklabels(CIFAR10)
    ax2.tick_params(axis='y')
    ax2.set_ylim(50, 105)
    y_line1 = acc
    ax2.axhline(y=acc, color='red', linestyle='--', linewidth=0.5)
    ax2.text(ax2.get_xlim()[1], y_line1, f'acc={acc}', verticalalignment='bottom', horizontalalignment='right')
    # 添加图例
    fig.legend(loc='upper left', bbox_to_anchor=(0, 0))
    
    # 显示图形
    # plt.title(f'{labell} Exit Rate and Exit Accuracy')
    plt.savefig(f"/home/nvidia/heShaoWei/GFNet/dynn/IMAGE/{labell}_Exit_Rate_and_Exit_Accuracy123.png")
    plt.show()

CIFAR10 = [8, 9, 10, 11]
real_exit_acc_CIFAR10 = [98.18920916481892, 97.55065013607499, 96.25779625779626, 92.46280991735537]
real_exit_rate_CIFAR10 = [27.060000000000002, 33.07, 9.62, 30.25]
plot_bars_and_scatter(CIFAR10, real_exit_rate_CIFAR10, real_exit_acc_CIFAR10,labell='CIFAR10',acc=96.06)
NWPU = [9, 10, 11]
real_exit_acc_NWPU = [97.55169954837176, 91.55455904334828, 77.6158940397351]
real_exit_rate_NWPU = [66.77777777777779, 21.23809523809524, 11.984126984126984]
plot_bars_and_scatter(NWPU, real_exit_rate_NWPU, real_exit_acc_NWPU,labell='NWPU',acc=93.89)
UCM = ['7','9','10','11']
real_exit_acc_UCM = [100.0, 96.41255605381166, 72.72727272727273, 0.0]
real_exit_rate_UCM = [41.66666666666667, 53.095238095238095, 5.238095238095238, 0.0]
plot_bars_and_scatter(UCM, real_exit_rate_UCM, real_exit_acc_UCM,labell='UCM',acc=96.66)