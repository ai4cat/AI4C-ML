import pandas as pd

# 读取Excel文件
# file1_path = 'D:/PKU\PKU240918/code_data/task_data/3.DFT_select-for_trainning.xlsx'

file1_path = '/home/a/PKU/PKU240918/code_data/task_data/base_model_data.xlsx'
file2_path = '/home/a/PKU/PKU240918/code_data/task_data/acitve_learning_data.xlsx'
file3_path = '/home/a/PKU/PKU240918/code_data/task_data/prediction_data.xlsx'
skiprows=1

df1 = pd.read_excel(file1_path,skiprows=skiprows)
df2 = pd.read_excel(file2_path,skiprows=skiprows)
df3 = pd.read_excel(file3_path,skiprows=skiprows)
# 处理跳过的行
index_skip_rows=3

# 分割数据并检查每部分数据的指定行范围是否存在重复
part1 = df1.iloc[:,5:25]
part2 = df2.iloc[ :,5:25]
part3 = df3.iloc[ :,5:25]
print("原始数据个数: ",len(part1))
# print("原始数据0: ",part1.iloc[0])
print("主动学习个数: ",len(part2))
# print("主动学习0: ",part2.iloc[0])
print("预测个数: ",len(part3))
# print("预测0: ",part3.iloc[0])

# 检查重复行并打印行号
def find_duplicates(a,a_index,b,b_index):
    for i in range(len(a)):
        for j in range(len(b)):
            if a.iloc[i].equals(b.iloc[j]):
                print(f"重复序号: 第 {a_index[i]+index_skip_rows} 行和第 {b_index[j]+index_skip_rows} 行")
# 比较各部分
print("检查第一部分和第二部分：")
find_duplicates(part1,df1.index, part2,df2.index)
print("检查第一部分和第三部分：")
find_duplicates(part1,df1.index, part3,df3.index)
print("检查第二部分和第三部分：")
find_duplicates(part2,df2.index, part3,df3.index)


# 检查重复行并打印行号
def find_column_duplicates(df):
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if df.iloc[i, :].equals(df.iloc[j, :]):
                print(f"重复序号: 第 {df.index[i]+index_skip_rows} 行和第 {df.index[j]+index_skip_rows} 行重复")

# 执行检查
print("第一部分自查：")
find_column_duplicates(df1)
# 执行检查
print("第二部分自查：")
find_column_duplicates(df2)
# 执行检查
print("第三部分自查：")
find_column_duplicates(df3)
