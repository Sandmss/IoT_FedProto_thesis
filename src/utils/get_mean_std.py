from statistics import mean # 从 statistics 模块导入 mean 函数，用于计算列表的平均值
import numpy as np # 导入 numpy 库，并将其别名为 np，用于进行科学计算，特别是标准差的计算

file_name = input() + '.out' # 获取用户输入作为文件名的一部分，并拼接 '.out' 后缀，形成完整的文件名

acc = [] # 初始化一个空列表，用于存储从文件中读取的最佳准确率数值

with open(file_name, 'r') as f: # 以只读模式 ('r') 打开指定的文件，并将其赋值给文件对象 f
    is_best = False # 初始化一个布尔变量 is_best，用于标记下一行是否包含“最佳准确率”
    for l in f.readlines(): # 遍历文件中的每一行
        if is_best: # 如果 is_best 为 True，表示上一行是“Best accuracy”标记行
            acc.append(float(l)) # 将当前行（应为准确率数值）转换为浮点数并添加到 acc 列表中
            is_best = False # 将 is_best 重置为 False，等待下一个“Best accuracy”标记
        elif 'Best accuracy' in l: # 如果当前行包含字符串 'Best accuracy'
            is_best = True # 将 is_best 设置为 True，表示下一行将是准确率数值

print(acc) # 打印包含所有提取到的最佳准确率的列表
print(mean(acc)*100, np.std(acc)*100) # 计算 acc 列表中数值的平均值和标准差，并将它们乘以 100 转换为百分比形式后打印
