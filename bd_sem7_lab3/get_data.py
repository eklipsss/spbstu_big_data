import pandas as pd


df_list = []
# month_int = 2
year = 2021
for year in [2021, 2022]:
    for month_int in range(1, 13):
        month_from_site = \
        pd.read_html(f'http://www.pogodaiklimat.ru/monitor.php?id=26063&month={month_int}&year={year}')[0]
        month = month_from_site.iloc[2:, [0, 2]].copy().reset_index(drop=True)
        month[0] = month[0].astype('str')
        month[0] = [f'{date}.{month_int}.{year}' for date in month[0]]
        month = month.dropna()

        df_list.append(month)

res = pd.concat(df_list, ignore_index=True)
res.columns = ['Date', 'Average_daily_temp']
print(res)
res.to_csv('average_daily_temperature_2021_2022.csv', index=False)
