import pandas as pd
import datetime
import re
import glob
import os

class FinancialAnalyzer:
    """财务数据分析核心类"""
    
    def __init__(self, config=None, progress_callback=None):
        """
        初始化分析器
        
        Args:
            config: 配置字典（由ConfigManager提供）
            progress_callback: 进度回调函数，签名：f(current, total, message)
        """
        self.config = config or {}
        self.progress_callback = progress_callback
        self.output_path = self.config.get('输出选项', {}).get('输出文件名', '清洗后的AI标准财务表') + '.xlsx'

    def update_progress(self, current, total, message):
        """更新进度回调"""
        if self.progress_callback:
            self.progress_callback(current, total, message)
        print(message)

    def clean_date_str(self, date_val):
        """清洗日期：支持 Excel数字、'2025年11月'、'2025-11-30' 等格式"""
        if pd.isna(date_val) or date_val == '':
            return "未知日期"
        
        if isinstance(date_val, (int, float)):
            try:
                return (datetime.datetime(1899, 12, 30) + datetime.timedelta(days=int(date_val))).strftime('%Y-%m-%d')
            except:
                return str(date_val)
                
        text = str(date_val)
        digits = re.findall(r'\d+', text)
        if len(digits) >= 2:
            year = digits[0]
            month = digits[1].zfill(2)
            day = digits[2].zfill(2) if len(digits) > 2 else "01"
            return f"{year}-{month}-{day}"
            
        return text.split(' ')[0]

    def clean_bs(self, file_path, sheet_name):
        """处理资产负债表 (包含BS的sheet)"""
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            date_val = df.iloc[2, 3] if not pd.isna(df.iloc[2, 3]) else df.iloc[2, 2]
            report_date = self.clean_date_str(date_val)
            
            header_row = df[df.apply(lambda x: x.astype(str).str.contains('期末余额').any(), axis=1)].index[0]
            
            df_left = df.iloc[header_row+1:, [0, 1, 2]].copy()
            df_left.columns = ['科目', '年初余额', '期末余额']
            df_left['大类'] = '资产'
            
            df_right = df.iloc[header_row+1:, [3, 4, 5]].copy()
            df_right.columns = ['科目', '年初余额', '期末余额']
            df_right['大类'] = '负债及权益'
            
            df_clean = pd.concat([df_left, df_right], ignore_index=True)
            df_clean = df_clean.dropna(subset=['科目'])
            df_clean = df_clean[df_clean['科目'].astype(str).str.strip() != '']
            
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

    def clean_pl(self, file_path, sheet_name):
        """处理利润表 (包含PL的sheet)"""
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            date_val = df.iloc[2, 0] if "报表期间" in str(df.iloc[2, 0]) else df.iloc[2, 2]
            report_date = self.clean_date_str(date_val)
            
            header_row = df[df.apply(lambda x: x.astype(str).str.contains('本期金额').any(), axis=1)].index[0]
            
            df_clean = df.iloc[header_row+1:, [0, 2, 3]].copy()
            df_clean.columns = ['科目', '本期金额', '本年累计金额']
            df_clean = df_clean.dropna(subset=['科目'])
            
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

    def clean_cf(self, file_path, sheet_name):
        """处理现金流量表 (包含CF的sheet)"""
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            date_val = df.iloc[2, 4] if not pd.isna(df.iloc[2, 4]) else df.iloc[2, 0]
            report_date = self.clean_date_str(date_val)
            
            header_row = df[df.apply(lambda x: x.astype(str).str.contains('本期金额').any(), axis=1)].index[0]
            
            df_left = df.iloc[header_row+1:, [0, 2, 3]].copy()
            df_left.columns = ['科目', '本期金额', '本年累计金额']
            
            if df.shape[1] >= 8:
                df_right = df.iloc[header_row+1:, [4, 6, 7]].copy()
                df_right.columns = ['科目', '本期金额', '本年累计金额']
                df_combined = pd.concat([df_left, df_right], ignore_index=True)
            else:
                df_combined = df_left
                
            df_combined = df_combined.dropna(subset=['科目'])
            df_combined = df_combined[df_combined['科目'].astype(str).str.strip() != '']
            
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

    def extract_amount(self, df, account_key, sheet_type=None, time_attr=None, category=None):
        """从配置好的关键字中提取金额"""
        keywords = self.config.get('科目映射', {}).get(account_key, [account_key])
        
        filtered_df = df.copy()
        if sheet_type:
            filtered_df = filtered_df[filtered_df['报表类型'] == sheet_type]
        if time_attr:
            filtered_df = filtered_df[filtered_df['时间属性'] == time_attr]
        if category:
            filtered_df = filtered_df[filtered_df['大类'] == category]
        
        for keyword in keywords:
            matched = filtered_df[filtered_df['科目'].str.contains(keyword, case=False, na=False)]
            if not matched.empty:
                return matched.iloc[0]['金额']
        return 0

    def validate_balance_sheet(self, df_group):
        """验证会计恒等式"""
        assets = self.extract_amount(df_group, '资产总计', sheet_type='资产负债表', category='资产')
        liabilities = self.extract_amount(df_group, '负债合计', sheet_type='资产负债表', category='负债及权益')
        equity = self.extract_amount(df_group, '所有者权益合计', sheet_type='资产负债表', category='负债及权益')
        
        diff = abs(assets - (liabilities + equity))
        tolerance = self.config.get('验证选项', {}).get('容差阈值', 0.01)
        is_balanced = diff <= tolerance
        
        return {
            '资产总计': assets,
            '负债合计': liabilities,
            '所有者权益合计': equity,
            '差额': diff,
            '是否平衡': '是' if is_balanced else '否',
            '验证结果': '通过' if is_balanced else f'不平衡(差额:{diff:.2f})'
        }

    def calculate_financial_metrics(self, df_group):
        """计算财务指标"""
        metrics = {}
        opt = self.config.get('指标选项', {})
        
        # 基础提取
        assets_total = self.extract_amount(df_group, '资产总计')
        current_assets = self.extract_amount(df_group, '流动资产合计')
        cash = self.extract_amount(df_group, '货币资金')
        inventory = self.extract_amount(df_group, '存货')
        liabilities_total = self.extract_amount(df_group, '负债合计')
        current_liabilities = self.extract_amount(df_group, '流动负债合计')
        equity_total = self.extract_amount(df_group, '所有者权益合计')
        
        revenue = self.extract_amount(df_group, '营业收入')
        cost = self.extract_amount(df_group, '营业成本')
        operating_profit = self.extract_amount(df_group, '营业利润')
        net_profit = self.extract_amount(df_group, '净利润')
        
        operating_cf = self.extract_amount(df_group, '经营活动现金流')
        investing_cf = self.extract_amount(df_group, '投资活动现金流')
        financing_cf = self.extract_amount(df_group, '筹资活动现金流')

        if opt.get('计算流动性指标', True) and current_liabilities != 0:
            metrics['流动比率'] = current_assets / current_liabilities
            metrics['速动比率'] = (current_assets - inventory) / current_liabilities
            metrics['现金比率'] = cash / current_liabilities
            
        if opt.get('计算偿债能力指标', True):
            if assets_total != 0: metrics['资产负债率'] = liabilities_total / assets_total
            if equity_total != 0:
                metrics['产权比率'] = liabilities_total / equity_total
                metrics['权益乘数'] = assets_total / equity_total
                
        if opt.get('计算盈利能力指标', True):
            if revenue != 0:
                metrics['毛利率'] = (revenue - cost) / revenue
                metrics['营业利润率'] = operating_profit / revenue
                metrics['净利率'] = net_profit / revenue
            if equity_total != 0: metrics['ROE(净资产收益率)'] = net_profit / equity_total
            if assets_total != 0: metrics['ROA(总资产收益率)'] = net_profit / assets_total
            
        if opt.get('计算现金流指标', True):
            metrics['经营活动现金流净额'] = operating_cf
            metrics['投资活动现金流净额'] = investing_cf
            metrics['筹资活动现金流净额'] = financing_cf
            if current_liabilities != 0: metrics['现金流量比率'] = operating_cf / current_liabilities
            
        return metrics

    def process_directory(self, directory):
        """处理整个目录"""
        excel_files = glob.glob(os.path.join(directory, '*.xlsx'))
        # 排除掉输出文件本身
        excel_files = [f for f in excel_files if os.path.basename(f) != os.path.basename(self.output_path)]
        
        if not excel_files:
            self.update_progress(0, 0, "⚠️ 未找到Excel文件")
            return

        all_files_data = []
        total_files = len(excel_files)
        
        for i, file_path in enumerate(excel_files):
            file_name = os.path.basename(file_path)
            self.update_progress(i, total_files, f"正在读取: {file_name}")
            
            try:
                excel_file = pd.ExcelFile(file_path)
                all_sheets = excel_file.sheet_names
                
                file_sheets_data = []
                bs_sheets = [s for s in all_sheets if 'BS' in s.upper()]
                pl_sheets = [s for s in all_sheets if 'PL' in s.upper()]
                cf_sheets = [s for s in all_sheets if 'CF' in s.upper()]
                
                for sheet in bs_sheets: file_sheets_data.append(self.clean_bs(file_path, sheet))
                for sheet in pl_sheets: file_sheets_data.append(self.clean_pl(file_path, sheet))
                for sheet in cf_sheets: file_sheets_data.append(self.clean_cf(file_path, sheet))
                
                if file_sheets_data:
                    file_data = pd.concat([d for d in file_sheets_data if not d.empty], ignore_index=True)
                    if not file_data.empty:
                        file_data['源文件'] = file_name
                        all_files_data.append(file_data)
            except Exception as e:
                self.update_progress(i, total_files, f"❌ {file_name} 读取失败: {e}")

        if not all_files_data:
            self.update_progress(total_files, total_files, "⚠️ 未能提取有效数据")
            return

        self.update_progress(total_files, total_files, "正在整合数据并计算指标...")
        all_data = pd.concat(all_files_data, ignore_index=True)
        
        # 清洗
        all_data['金额'] = all_data['金额'].astype(str).str.replace('—', '0').str.replace(',', '')
        all_data['金额'] = pd.to_numeric(all_data['金额'], errors='coerce').fillna(0)
        all_data['科目'] = all_data['科目'].astype(str).str.strip()
        
        # 排序
        cols = ['源文件', '来源Sheet', '日期', '报表类型', '大类', '科目', '时间属性', '金额']
        all_data = all_data[[c for c in cols if c in all_data.columns]]
        
        # 分组计算
        group_cols = ['源文件', '来源Sheet', '日期', '时间属性']
        existing_group_cols = [col for col in group_cols if col in all_data.columns]
        
        validation_results = []
        metrics_results = []
        
        grouped = all_data.groupby(existing_group_cols, dropna=False)
        for keys, df_group in grouped:
            group_info = dict(zip(existing_group_cols, keys if isinstance(keys, tuple) else [keys]))
            
            if self.config.get('验证选项', {}).get('启用会计恒等式验证', True):
                if '资产负债表' in df_group['报表类型'].values:
                    v = self.validate_balance_sheet(df_group)
                    v.update(group_info)
                    validation_results.append(v)
            
            m = self.calculate_financial_metrics(df_group)
            m.update(group_info)
            metrics_results.append(m)

        # 导出
        output_opts = self.config.get('输出选项', {})
        
        if output_opts.get('生成原始数据', True):
            all_data.to_excel(self.output_path, index=False)
        
        if validation_results and output_opts.get('生成验证报告', True):
            v_path = self.output_path.replace('.xlsx', '_验证报告.xlsx')
            pd.DataFrame(validation_results).to_excel(v_path, index=False)
            
        if metrics_results and output_opts.get('生成财务指标', True):
            m_path = self.output_path.replace('.xlsx', '_财务指标.xlsx')
            pd.DataFrame(metrics_results).to_excel(m_path, index=False)

        self.update_progress(total_files, total_files, "✅ 处理完成！")

if __name__ == '__main__':
    # 兼容原有的命令行运行模式
    from config_manager import ConfigManager
    cm = ConfigManager()
    analyzer = FinancialAnalyzer(config=cm.config)
    analyzer.process_directory('.')

