import pandas as pd
df = pd.read_excel("D:\24Spring\cps4951\Pen_Grip\Final_version\DataAnalysis\hand_gestures_data.xlsx")
df = pd.get_dummies(df)
df.to_csv('ProcessedData.csv',index = False)
