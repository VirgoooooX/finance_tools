# -*- coding: utf-8 -*-
import sys, os, glob
sys.path.insert(0, r'L:\Web\Financial data analysis')
sys.path.insert(0, r'L:\Web\Financial data analysis\tools')

import pandas as pd
from report_ingestor.core import clean_bs, clean_pl, clean_cf, load_config

cfg = load_config(r'L:\Web\Financial data analysis\tools\report_ingestor\config.json')

# 找到报表文件
files = glob.glob(os.path.join(r'L:\Web\Financial data analysis', '报表-2026年3期月报.xlsx'))
if not files:
    files = glob.glob(os.path.join(r'L:\Web\Financial data analysis', '*月报*.xlsx'))
print("Found:", files)
fp = files[0]

ef = pd.ExcelFile(fp)
print('Sheets:', ef.sheet_names)

df_bs = clean_bs('', ef.sheet_names[0], cfg, excel_file=ef)
huobi = df_bs[df_bs['科目'].str.strip() == '货币资金'][['科目','时间属性','金额']]
print('\n=== BS 货币资金 ===')
print(huobi.to_string())

df_pl = clean_pl('', ef.sheet_names[1], cfg, excel_file=ef)
rev = df_pl[df_pl['科目'].str.strip().str.contains('营业收入')][['科目','时间属性','金额']]
print('\n=== PL 营业收入 ===')
print(rev.to_string())

df_cf = clean_cf('', ef.sheet_names[2], cfg, excel_file=ef)
opcf = df_cf[df_cf['科目'].str.strip().str.contains('经营活动产生的现金流量净')][['科目','时间属性','金额']]
print('\n=== CF 经营净额 ===')
print(opcf.to_string())

print(f'\nBS:{len(df_bs)} PL:{len(df_pl)} CF:{len(df_cf)} - ALL OK')
