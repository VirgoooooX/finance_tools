import pandas as pd
import datetime
import re
import glob
import os

# ================= 配置区域 =================
# 脚本将自动处理当前目录下所有的 .xlsx 文件
OUTPUT_PATH = '清洗后的AI标准财务表.xlsx'
# ===========================================

def clean_date_str(date_val):
    """
    清洗日期：支持 Excel数字、'2025年11月'、'2025-11-30' 等格式
    """
    if pd.isna(date_val) or date_val == '':
        return "未知日期"
    
    # 1. Excel 数字格式 (例如 45991)
    if isinstance(date_val, (int, float)):
        try:
            return (datetime.datetime(1899, 12, 30) + datetime.timedelta(days=int(date_val))).strftime('%Y-%m-%d')
        except:
            return str(date_val)
            
    # 2. 字符串格式处理
    text = str(date_val)
    # 提取所有数字，简单拼接 (处理 "2025年11月")
    digits = re.findall(r'\d+', text)
    if len(digits) >= 2:
        year = digits[0]
        month = digits[1].zfill(2)
        day = digits[2].zfill(2) if len(digits) > 2 else "01" # 如果没有日，默认为01号
        return f"{year}-{month}-{day}"
        
    return text.split(' ')[0]

def clean_bs(file_path, sheet_name):
    """处理资产负债表 (包含BS的sheet) - 图1格式"""
    print(f"正在处理: {sheet_name} ...")
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        
        # 1. 抓取日期 (图1显示在第3行左右)
        date_val = df.iloc[2, 3] # 盲猜A3
        if pd.isna(date_val): date_val = df.iloc[2, 2] # 试试C3
        report_date = clean_date_str(date_val)
        
        # 2. 定位表头 (包含 '期末余额')
        header_row = df[df.apply(lambda x: x.astype(str).str.contains('期末余额').any(), axis=1)].index[0]
        
        # 3. 拆解左右分栏
        # 左边资产: [科目, 年初, 期末] -> A, B, C (Index 0,1,2)
        df_left = df.iloc[header_row+1:, [0, 1, 2]].copy()
        df_left.columns = ['科目', '年初余额', '期末余额']
        df_left['大类'] = '资产'
        
        # 右边负债: [科目, 年初, 期末] -> D, E, F (Index 3,4,5)
        df_right = df.iloc[header_row+1:, [3, 4, 5]].copy()
        df_right.columns = ['科目', '年初余额', '期末余额']
        df_right['大类'] = '负债及权益'
        
        # 4. 合并与清洗
        df_clean = pd.concat([df_left, df_right], ignore_index=True)
        df_clean = df_clean.dropna(subset=['科目']) # 删除空行
        df_clean = df_clean[df_clean['科目'].astype(str).str.strip() != '']
        
        # 5. 逆透视
        df_final = df_clean.melt(id_vars=['大类', '科目'], 
                                 value_vars=['年初余额', '期末余额'],
                                 var_name='时间属性', value_name='金额')
        
        df_final['报表类型'] = '资产负债表'
        df_final['日期'] = report_date
        df_final['来源Sheet'] = sheet_name
        return df_final
    except Exception as e:
        print(f"❌ {sheet_name} 处理失败: {e}")
        return pd.DataFrame()

def clean_pl(file_path, sheet_name):
    """处理利润表 (包含PL的sheet) - 图2格式"""
    print(f"正在处理: {sheet_name} ...")
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        
        # 1. 抓取日期 (图2显示在第3行左右)
        date_val = df.iloc[2, 0] 
        if pd.isna(date_val) or "报表期间" not in str(date_val): date_val = df.iloc[2, 2] # C3
        report_date = clean_date_str(date_val)
        
        # 2. 定位表头 (包含 '本期金额')
        header_row = df[df.apply(lambda x: x.astype(str).str.contains('本期金额').any(), axis=1)].index[0]
        
        # 3. 提取数据
        # 结构: [科目(A), 行次(B), 本期(C), 累计(D)] -> 取 Index 0, 2, 3
        df_clean = df.iloc[header_row+1:, [0, 2, 3]].copy()
        df_clean.columns = ['科目', '本期金额', '本年累计金额']
        
        df_clean = df_clean.dropna(subset=['科目'])
        
        # 4. 逆透视
        df_final = df_clean.melt(id_vars=['科目'], 
                                 value_vars=['本期金额', '本年累计金额'],
                                 var_name='时间属性', value_name='金额')
        
        df_final['大类'] = '损益'
        df_final['报表类型'] = '利润表'
        df_final['日期'] = report_date
        df_final['来源Sheet'] = sheet_name
        return df_final
    except Exception as e:
        print(f"❌ {sheet_name} 处理失败: {e}")
        return pd.DataFrame()

def clean_cf(file_path, sheet_name):
    """处理现金流量表 (包含CF的sheet) - 图3格式"""
    print(f"正在处理: {sheet_name} ...")
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        
        # 1. 抓取日期
        date_val = df.iloc[2, 4] # 图3显示日期在E列附近
        if pd.isna(date_val): date_val = df.iloc[2, 0]
        report_date = clean_date_str(date_val)
        
        # 2. 定位表头 (包含 '本期金额')
        header_row = df[df.apply(lambda x: x.astype(str).str.contains('本期金额').any(), axis=1)].index[0]
        
        # 3. 拆解左右分栏 (特别注意：中间夹着行次列)
        
        # 左边: [科目(A), 行次(B), 本期(C), 累计(D)] -> 取 Index 0, 2, 3
        df_left = df.iloc[header_row+1:, [0, 2, 3]].copy()
        df_left.columns = ['科目', '本期金额', '本年累计金额']
        
        # 右边: [科目(E), 行次(F), 本期(G), 累计(H)] -> 取 Index 4, 6, 7
        # 先检查是否有足够的列，防止报错
        if df.shape[1] >= 8:
            df_right = df.iloc[header_row+1:, [4, 6, 7]].copy()
            df_right.columns = ['科目', '本期金额', '本年累计金额']
            df_combined = pd.concat([df_left, df_right], ignore_index=True)
        else:
            df_combined = df_left
            
        # 4. 清洗
        df_combined = df_combined.dropna(subset=['科目'])
        df_combined = df_combined[df_combined['科目'].astype(str).str.strip() != '']
        
        # 5. 逆透视
        df_final = df_combined.melt(id_vars=['科目'], 
                                    value_vars=['本期金额', '本年累计金额'],
                                    var_name='时间属性', value_name='金额')
        
        df_final['大类'] = '现金流'
        df_final['报表类型'] = '现金流量表'
        df_final['日期'] = report_date
        df_final['来源Sheet'] = sheet_name
        return df_final
    except Exception as e:
        print(f"❌ {sheet_name} 处理失败: {e}")
        return pd.DataFrame()

# ================= 数据验证与财务指标计算 =================

def extract_amount(df, keywords, sheet_type=None, time_attr=None, category=None):
    """
    从DataFrame中提取符合条件的科目金额
    
    参数:
        df: 数据DataFrame
        keywords: 科目关键字列表，匹配任意一个即可
        sheet_type: 报表类型筛选（资产负债表、利润表、现金流量表）
        time_attr: 时间属性筛选（期末余额、年初余额等）
        category: 大类筛选（资产、负债及权益、损益、现金流）
    
    返回: 匹配到的第一个金额值，未找到返回0
    """
    filtered_df = df.copy()
    
    # 筛选条件
    if sheet_type:
        filtered_df = filtered_df[filtered_df['报表类型'] == sheet_type]
    if time_attr:
        filtered_df = filtered_df[filtered_df['时间属性'] == time_attr]
    if category:
        filtered_df = filtered_df[filtered_df['大类'] == category]
    
    # 科目名称匹配（模糊匹配，忽略空格）
    for keyword in keywords:
        matched = filtered_df[filtered_df['科目'].str.contains(keyword, case=False, na=False)]
        if not matched.empty:
            return matched.iloc[0]['金额']
    
    return 0

def validate_balance_sheet(df_group):
    """
    验证资产负债表的会计恒等式：资产 = 负债 + 所有者权益
    
    参数:
        df_group: 单个分组的数据（同一文件、Sheet、日期、时间点）
    
    返回: dict包含验证结果
    """
    # 提取关键科目
    assets = extract_amount(df_group, ['资产总计', '资产总额', '资产合计'], 
                           sheet_type='资产负债表', category='资产')
    liabilities = extract_amount(df_group, ['负债合计', '负债总计', '负债总额'], 
                                 sheet_type='资产负债表', category='负债及权益')
    equity = extract_amount(df_group, ['所有者权益合计', '股东权益合计', '所有者权益总计', '权益合计'], 
                           sheet_type='资产负债表', category='负债及权益')
    
    # 计算差额
    diff = abs(assets - (liabilities + equity))
    tolerance = 0.01  # 容差阈值
    is_balanced = diff <= tolerance
    
    return {
        '资产总计': assets,
        '负债合计': liabilities,
        '所有者权益合计': equity,
        '差额': diff,
        '是否平衡': '是' if is_balanced else '否',
        '验证结果': '通过' if is_balanced else f'不平衡(差额:{diff:.2f})'
    }

def calculate_financial_metrics(df_group):
    """
    计算财务指标
    
    参数:
        df_group: 单个分组的数据（同一文件、Sheet、日期、时间点）
    
    返回: dict包含各类财务指标
    """
    metrics = {}
    
    # ===== 提取基础科目金额 =====
    # 资产负债表科目
    assets_total = extract_amount(df_group, ['资产总计', '资产总额'], sheet_type='资产负债表')
    current_assets = extract_amount(df_group, ['流动资产合计', '流动资产总计'], sheet_type='资产负债表')
    cash = extract_amount(df_group, ['货币资金', '现金及现金等价物'], sheet_type='资产负债表')
    inventory = extract_amount(df_group, ['存货'], sheet_type='资产负债表')
    
    liabilities_total = extract_amount(df_group, ['负债合计', '负债总计'], sheet_type='资产负债表')
    current_liabilities = extract_amount(df_group, ['流动负债合计', '流动负债总计'], sheet_type='资产负债表')
    equity_total = extract_amount(df_group, ['所有者权益合计', '股东权益合计', '权益合计'], sheet_type='资产负债表')
    
    # 利润表科目
    revenue = extract_amount(df_group, ['营业收入', '主营业务收入'], sheet_type='利润表')
    cost = extract_amount(df_group, ['营业成本', '主营业务成本'], sheet_type='利润表')
    operating_profit = extract_amount(df_group, ['营业利润'], sheet_type='利润表')
    net_profit = extract_amount(df_group, ['净利润'], sheet_type='利润表')
    
    # 现金流量表科目
    operating_cf = extract_amount(df_group, ['经营活动产生的现金流量净额', '经营活动现金流量净额'], sheet_type='现金流量表')
    investing_cf = extract_amount(df_group, ['投资活动产生的现金流量净额', '投资活动现金流量净额'], sheet_type='现金流量表')
    financing_cf = extract_amount(df_group, ['筹资活动产生的现金流量净额', '筹资活动现金流量净额'], sheet_type='现金流量表')
    
    # ===== 计算流动性指标 =====
    metrics['流动比率'] = current_assets / current_liabilities if current_liabilities != 0 else None
    metrics['速动比率'] = (current_assets - inventory) / current_liabilities if current_liabilities != 0 else None
    metrics['现金比率'] = cash / current_liabilities if current_liabilities != 0 else None
    
    # ===== 计算偿债能力指标 =====
    metrics['资产负债率'] = liabilities_total / assets_total if assets_total != 0 else None
    metrics['产权比率'] = liabilities_total / equity_total if equity_total != 0 else None
    metrics['权益乘数'] = assets_total / equity_total if equity_total != 0 else None
    
    # ===== 计算盈利能力指标 =====
    metrics['毛利率'] = (revenue - cost) / revenue if revenue != 0 else None
    metrics['营业利润率'] = operating_profit / revenue if revenue != 0 else None
    metrics['净利率'] = net_profit / revenue if revenue != 0 else None
    metrics['ROE(净资产收益率)'] = net_profit / equity_total if equity_total != 0 else None
    metrics['ROA(总资产收益率)'] = net_profit / assets_total if assets_total != 0 else None
    
    # ===== 现金流指标 =====
    metrics['经营活动现金流净额'] = operating_cf
    metrics['投资活动现金流净额'] = investing_cf
    metrics['筹资活动现金流净额'] = financing_cf
    metrics['现金流量比率'] = operating_cf / current_liabilities if current_liabilities != 0 else None
    
    return metrics


# ================= 主程序执行 =================
if __name__ == '__main__':
    # 1. 查找当前目录下所有的 Excel 文件
    excel_files = glob.glob('*.xlsx')
    
    if not excel_files:
        print("⚠️ 当前目录下未找到任何 .xlsx 文件！")
        exit()
    
    print(f"找到 {len(excel_files)} 个Excel文件:")
    for f in excel_files:
        print(f"  - {f}")
    print()
    
    # 2. 用于存储所有文件的数据
    all_files_data = []
    
    # 3. 循环处理每个文件
    for file_path in excel_files:
        print(f"\n{'='*50}")
        print(f"正在处理文件: {file_path}")
        print(f"{'='*50}")
        
        # 读取Excel文件的所有sheet名称
        try:
            excel_file = pd.ExcelFile(file_path)
            all_sheets = excel_file.sheet_names
            print(f"发现 {len(all_sheets)} 个Sheet: {all_sheets}")
            
            # 查找包含关键字的sheets
            bs_sheets = [s for s in all_sheets if 'BS' in s.upper()]
            pl_sheets = [s for s in all_sheets if 'PL' in s.upper()]
            cf_sheets = [s for s in all_sheets if 'CF' in s.upper()]
            
            print(f"  - 资产负债表(BS)相关: {bs_sheets if bs_sheets else '无'}")
            print(f"  - 利润表(PL)相关: {pl_sheets if pl_sheets else '无'}")
            print(f"  - 现金流量表(CF)相关: {cf_sheets if cf_sheets else '无'}")
            
            # 存储当前文件所有sheet的数据
            file_sheets_data = []
            
            # 处理所有BS相关的sheet
            for sheet in bs_sheets:
                df = clean_bs(file_path, sheet)
                if not df.empty:
                    file_sheets_data.append(df)
            
            # 处理所有PL相关的sheet
            for sheet in pl_sheets:
                df = clean_pl(file_path, sheet)
                if not df.empty:
                    file_sheets_data.append(df)
            
            # 处理所有CF相关的sheet
            for sheet in cf_sheets:
                df = clean_cf(file_path, sheet)
                if not df.empty:
                    file_sheets_data.append(df)
            
            # 合并当前文件的所有sheet数据
            if file_sheets_data:
                file_data = pd.concat(file_sheets_data, ignore_index=True)
                # 添加文件来源标识
                file_data['源文件'] = file_path
                all_files_data.append(file_data)
                print(f"✅ {file_path} 处理完成，提取 {len(file_data)} 行数据")
            else:
                print(f"⚠️ {file_path} 未提取到任何数据，可能缺少包含BS/PL/CF的Sheet")
                
        except Exception as e:
            print(f"❌ {file_path} 读取失败: {e}")
    
    # 4. 合并所有文件的数据
    if all_files_data:
        all_data = pd.concat(all_files_data, ignore_index=True)
        
        # 5. 最终数值清洗
        # 替换 '-' 为 0，转为数字
        all_data['金额'] = all_data['金额'].astype(str).str.replace('—', '0').str.replace(',', '')
        all_data['金额'] = pd.to_numeric(all_data['金额'], errors='coerce').fillna(0)
        
        # 去掉科目名称里的空格 (比如 ' 货币资金 ' -> '货币资金')
        all_data['科目'] = all_data['科目'].astype(str).str.strip()
        
        # 6. 重新排列列顺序，符合人类阅读习惯
        cols = ['源文件', '来源Sheet', '日期', '报表类型', '大类', '科目', '时间属性', '金额']
        # 防止某些列不存在（如BS里没有大类），做个交集处理
        final_cols = [c for c in cols if c in all_data.columns]
        all_data = all_data[final_cols]
        
        # 7. 数据验证与财务指标计算
        print(f"\n{'='*50}")
        print("📊 开始数据验证与财务指标计算...")
        print(f"{'='*50}")
        
        # 按（源文件、来源Sheet、日期、时间属性）分组
        group_cols = ['源文件', '来源Sheet', '日期', '时间属性']
        existing_group_cols = [col for col in group_cols if col in all_data.columns]
        
        validation_results = []
        metrics_results = []
        
        if existing_group_cols:
            grouped = all_data.groupby(existing_group_cols, dropna=False)
            
            for group_keys, df_group in grouped:
                # 构建分组标识
                group_info = dict(zip(existing_group_cols, group_keys if isinstance(group_keys, tuple) else [group_keys]))
                
                # 数据验证（仅对资产负债表）
                if '资产负债表' in df_group['报表类型'].values:
                    validation = validate_balance_sheet(df_group)
                    validation.update(group_info)
                    validation_results.append(validation)
                
                # 财务指标计算
                metrics = calculate_financial_metrics(df_group)
                metrics.update(group_info)
                metrics_results.append(metrics)
        
        # 8. 输出结果
        # 8.1 原始清洗数据
        all_data.to_excel(OUTPUT_PATH, index=False)
        print(f"✅ 原始数据已保存: {OUTPUT_PATH}")
        
        # 8.2 数据验证报告
        if validation_results:
            df_validation = pd.DataFrame(validation_results)
            validation_output = OUTPUT_PATH.replace('.xlsx', '_验证报告.xlsx')
            df_validation.to_excel(validation_output, index=False)
            print(f"✅ 验证报告已保存: {validation_output}")
            
            # 显示不平衡的记录
            unbalanced = df_validation[df_validation['是否平衡'] == '否']
            if not unbalanced.empty:
                print(f"\n⚠️ 发现 {len(unbalanced)} 条不平衡记录：")
                print(unbalanced[['源文件', '来源Sheet', '日期', '时间属性', '差额', '验证结果']].to_string(index=False))
            else:
                print("\n✅ 所有资产负债表均通过会计恒等式验证！")
        
        # 8.3 财务指标汇总
        if metrics_results:
            df_metrics = pd.DataFrame(metrics_results)
            metrics_output = OUTPUT_PATH.replace('.xlsx', '_财务指标.xlsx')
            df_metrics.to_excel(metrics_output, index=False)
            print(f"✅ 财务指标已保存: {metrics_output}")
        
        # 9. 总结输出
        print(f"\n{'='*50}")
        print("✅ 所有处理完成！")
        print(f"{'='*50}")
        print(f"📁 共处理 {len(excel_files)} 个Excel文件")
        print(f"📊 合并 {len(all_data)} 行原始数据")
        print(f"📈 生成 {len(metrics_results)} 组财务指标")
        if validation_results:
            print(f"🔍 验证 {len(validation_results)} 组资产负债表数据")
        
        print(f"\n📂 输出文件：")
        print(f"  1. {OUTPUT_PATH}")
        if validation_results:
            print(f"  2. {validation_output}")
        if metrics_results:
            print(f"  3. {metrics_output}")
        
        print("\n📋 原始数据前10行预览：")
        print(all_data.head(10))
        
        if metrics_results and len(df_metrics) > 0:
            print("\n📊 财务指标前5组预览：")
            # 选择关键指标显示
            key_metrics = ['源文件', '日期', '时间属性', '流动比率', '资产负债率', '毛利率', '净利率', 'ROE(净资产收益率)']
            display_cols = [col for col in key_metrics if col in df_metrics.columns]
            print(df_metrics[display_cols].head(5).to_string(index=False))
    else:
        print("\n⚠️ 所有文件均未提取到有效数据，请检查Sheet名是否正确。")

