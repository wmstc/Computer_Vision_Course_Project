import pandas as pd

df = pd.read_excel('e:/cv_project/2/class_names.xlsx')
print("=== Excel文件内容 ===")
print(df.head(20))
print(f"\n总行数: {len(df)}")
print(f"\n列名: {df.columns.tolist()}")
print(f"\n所有数据:\n{df.to_dict('records')}")
