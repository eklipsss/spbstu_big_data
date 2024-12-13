import pandas as pd


data_table = pd.read_html("http://www.pogodaiklimat.ru/history/22546.htm", header=0)
# data_table = pd.read_html("http://www.pogodaiklimat.ru/history/26063.htm", header=0)

data_list_1 = data_table[0]
data_list_2 = data_table[1]

df = pd.concat([data_list_1['год'], data_list_2['за год']],  axis=1, keys=['год', 'температура за год'])
df.set_index('год', inplace=True)

df_data = df.tail(61).head(60)
print(df_data)
df_data.to_csv('average_year_temperature_Severodvinsk.csv', index=True)

# df_data = df.tail(201).head(200)
# print(df_data)
# df_data.to_csv('average_year_temperature_Spb.csv', index=True)
