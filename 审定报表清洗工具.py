import pandas as pd
import os
import re
import warnings

# 忽略Excel格式警告
warnings.filterwarnings('ignore')

# ================= 配置区 =================
FOLDER_PATH = r"."  
# =========================================

def get_file_date(filename):
    """从文件名只要能提取到20xx就默认为当年12-31"""
    match = re.search(r'(20\d{2})', filename)
    if match:
        return f"{match.group(1)}-12-31"
    return "未知日期"

def get_report_type(sheet_name):
    """判断Sheet是哪种报表"""
    name = sheet_name.upper()
    # 关键词判断
    if "BS" in name or "资产" in name or "01" in name: return "资产负债表"
    if "PL" in name or "利润" in name or "损益" in name or "02" in name: return "利润表"
    if "CF" in name or "现金" in name or "03" in name: return "现金流量表"
    return "其他报表"

def clean_amount(val):
    """清洗金额：去逗号，转数字"""
    try:
        s = str(val).replace(',', '').replace('，', '').replace(' ', '')
        if s in ['-', '', 'nan', 'None']: return 0.0
        return float(s)
    except:
        return None

def find_col(headers, keywords, start=0, end=999):
    """在表头里寻找关键词所在的列索引"""
    limit = min(end, len(headers))
    for i in range(start, limit):
        h = str(headers[i])
        for kw in keywords:
            if kw in h: return i, h
    return None, None

def process_excel_final():
    print(">>> 启动定制规则提取程序...")
    print("规则：BS左->资产 | BS右->负债和权益 | PL->权益 | CF->现金流")
    
    all_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith('.xlsx') and not f.startswith('~$')]
    if not all_files:
        print("【错误】目录下没有Excel文件！")
        return

    result_list = []

    for filename in all_files:
        if "AI清洗" in filename: continue
        
        file_path = os.path.join(FOLDER_PATH, filename)
        print(f"正在处理: {filename}")
        file_date = get_file_date(filename)
        
        try:
            xls = pd.ExcelFile(file_path)
        except Exception as e:
            print(f"  读取错误: {e}")
            continue

        for sheet_name in xls.sheet_names:
            rpt_type = get_report_type(sheet_name)
            if rpt_type == "其他报表": continue # 跳过不相关的表
            
            # 读取数据
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            
            # 1. 找金额行作为表头
            header_idx = -1
            for idx, row in df.iterrows():
                txt = "".join([str(x) for x in row.values])
                if "余额" in txt or "金额" in txt:
                    header_idx = idx
                    break
            
            if header_idx == -1: continue

            # 整理表头
            raw_headers = df.iloc[header_idx].fillna("").astype(str).tolist()
            headers = [h.replace("\n", "").replace(" ", "") for h in raw_headers]
            data_rows = df.iloc[header_idx + 1:]

            # 2. 根据报表类型生成提取任务
            tasks = []
            
            # === 规则 A: 资产负债表 (切分为左右两块) ===
            if rpt_type == "资产负债表":
                # 找左边科目列 (通常含"资产")
                subj_L_idx, _ = find_col(headers, ["资产", "项目"])
                # 找右边科目列 (通常含"负债")，且必须在左边的后面
                subj_R_idx, _ = find_col(headers, ["负债"], start=(subj_L_idx+1 if subj_L_idx else 0))
                
                # 如果找不到右边列，就设为无穷大（当作单栏表处理）
                if subj_R_idx is None: subj_R_idx = 9999

                # --- 配置左边任务 (资产) ---
                if subj_L_idx is not None:
                    range_end = min(subj_R_idx, len(headers))
                    # 找当期
                    idx_curr, n_curr = find_col(headers, ["期末", "本期"], subj_L_idx+1, range_end)
                    if idx_curr: tasks.append({"大类": "资产", "科目列": subj_L_idx, "金额列": idx_curr, "属性": n_curr})
                    # 找上期
                    idx_prev, n_prev = find_col(headers, ["上年", "上期", "年初"], subj_L_idx+1, range_end)
                    if idx_prev: tasks.append({"大类": "资产", "科目列": subj_L_idx, "金额列": idx_prev, "属性": n_prev})

                # --- 配置右边任务 (负债和权益) ---
                if subj_R_idx != 9999:
                    range_start = subj_R_idx + 1
                    range_end = len(headers)
                    # 找当期
                    idx_curr, n_curr = find_col(headers, ["期末", "本期"], range_start, range_end)
                    if idx_curr: tasks.append({"大类": "负债和权益", "科目列": subj_R_idx, "金额列": idx_curr, "属性": n_curr})
                    # 找上期
                    idx_prev, n_prev = find_col(headers, ["上年", "上期", "年初"], range_start, range_end)
                    if idx_prev: tasks.append({"大类": "负债和权益", "科目列": subj_R_idx, "金额列": idx_prev, "属性": n_prev})

            # === 规则 B: 利润表 (大类强制为"权益") ===
            elif rpt_type == "利润表":
                # 找科目列
                subj_idx, _ = find_col(headers, ["项目", "摘要"])
                if subj_idx is None: subj_idx = 0 # 找不到就默认第0列
                
                # 找当期
                idx_curr, n_curr = find_col(headers, ["本期", "本年", "金额"], subj_idx+1)
                if idx_curr: tasks.append({"大类": "权益", "科目列": subj_idx, "金额列": idx_curr, "属性": n_curr})
                
                # 找上期
                idx_prev, n_prev = find_col(headers, ["上期", "上年"], subj_idx+1)
                if idx_prev: tasks.append({"大类": "权益", "科目列": subj_idx, "金额列": idx_prev, "属性": n_prev})

            # === 规则 C: 现金流量表 (大类强制为"现金流") ===
            elif rpt_type == "现金流量表":
                subj_idx, _ = find_col(headers, ["项目", "摘要"])
                if subj_idx is None: subj_idx = 0
                
                # 找当期
                idx_curr, n_curr = find_col(headers, ["本期", "本年", "金额"], subj_idx+1)
                if idx_curr: tasks.append({"大类": "现金流", "科目列": subj_idx, "金额列": idx_curr, "属性": n_curr})
                
                # 找上期
                idx_prev, n_prev = find_col(headers, ["上期", "上年"], subj_idx+1)
                if idx_prev: tasks.append({"大类": "现金流", "科目列": subj_idx, "金额列": idx_prev, "属性": n_prev})

            # 3. 执行所有任务
            for task in tasks:
                for _, row in data_rows.iterrows():
                    # 提取科目
                    try:
                        s_val = row[task['科目列']]
                        if pd.isna(s_val): continue
                        subject = str(s_val).strip().replace(" ", "")
                        if subject in ["", "nan", "None"]: continue
                    except: continue

                    # 提取金额
                    try:
                        amt = clean_amount(row[task['金额列']])
                        if amt is None: continue
                    except: continue

                    result_list.append({
                        "源文件": filename,
                        "来源Sheet": sheet_name,
                        "日期": file_date,
                        "报表类型": rpt_type,
                        "大类": task['大类'],       # 强制指定的大类
                        "科目": subject,
                        "时间属性": task['属性'],
                        "金额": amt
                    })

    # 保存
    if result_list:
        df_out = pd.DataFrame(result_list)
        cols = ["源文件", "来源Sheet", "日期", "报表类型", "大类", "科目", "时间属性", "金额"]
        df_out = df_out[cols]
        out_name = "AI清洗结果_最终版.xlsx"
        df_out.to_excel(out_name, index=False)
        print("="*50)
        print(f"【成功】处理完毕，生成文件：{out_name}")
        print("已严格执行大类映射规则。")
    else:
        print("未提取到数据。")

if __name__ == "__main__":
    process_excel_final()
    input("按回车键退出...")
