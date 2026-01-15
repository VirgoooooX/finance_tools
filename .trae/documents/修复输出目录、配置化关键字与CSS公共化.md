## 问题诊断
- 输出目录：两工具的 core.py 都以 cfg.output_dir 为根，并按 cfg.tool_id 与时间戳拼接；若 tool_id 缺失或被首个工具的配置复用，切换工具时仍落到首个工具目录。另有对“已是时间戳目录”的判断，可能导致跳过工具子目录。
- 关键字识别：audit_report_cleaner 的表类型与列头定位主要用硬编码字符串；其 config.json 已有 sheet/header 关键字字段但未被实际使用。monthly_report_cleaner 已实现从配置与 rules.json 读取并扩展关键词。
- CSS：两工具的 web/styles.css 内容高度重合，类名与变量一致；项目根 web/styles.css 也包含相同基础样式，可抽取公共样式。

## 实施方案
### 1) 输出目录修复
1. 在每个工具核心模块内显式计算 tool_id：默认取工具包目录名（如 os.path.basename(os.path.dirname(__file__))），当 cfg.tool_id 提供时使用配置值。
2. 统一输出路径结构：始终采用 output_root/<tool_id>/<timestamp>，不再以“是否时间戳目录”分支跳过工具层级。
3. 校正 output_root 解析：cfg.output_dir 若为相对路径，则解析到项目根的 output；若为绝对路径，仍在其下追加 <tool_id>/<timestamp>。
4. 为历史兼容提供迁移：若检测到旧式路径（无工具层级），将自动创建规范目录并将新生成文件写入规范位置。

### 2) 审定工具关键字配置化（对齐 monthly）
1. 读取 audit_report_cleaner/config.json 中的 sheet_keyword_bs/pl/cf 与 header_keyword_* 字段；新增对 rules.json 的可选支持，用于别名扩展。
2. 用配置驱动的表选择：按 cfg.sheet_keyword_* 在工作表名中匹配（大小写无关），若配置缺失再回退到原硬编码启发式。
3. 用配置驱动的表头定位：_find_header_row(df, cfg.header_keyword_*)；支持关键词集合与别名扩展。
4. 时间属性与列匹配：将 find_col 调用的硬编码关键词改为从配置/规则集合获取；保留健壮的回退路径与日志提示（便于排查缺少配置时的行为）。
5. 单元测试/样例：为 BS/PL/CF 三类最小样例表构造用例，验证不同配置下识别与抽取的稳定性。

### 3) CSS 公共化
1. 创建 web/common.css，提取两工具 styles.css 的公共变量、布局与组件样式（.app-container, .header, .card, .nav-tabs-custom, .form-control 等）。
2. 工具页面改为同时引入 common.css 与各自的 styles.css（该文件仅保留极少的工具特定覆盖）。
3. 清理重复样式：若工具 styles.css 与 common.css 完全一致，可直接删除工具私有样式，统一使用 common.css。
4. 可视验证：打开两工具页面检查导航、表格、表单与主题色一致性，无破版后再确认收敛。

## 验证与回退
- 运行两工具分别生成 out，确认落目录为 output/<tool_id>/<timestamp>，且 tool_id 随工具切换正确变化。
- 用包含多种命名的工作表与表头文档做回归，确保 audit 的识别与 monthly 一致。
- 页面样式对比与交互检查；如出现差异，使用工具私有 styles.css 进行最小覆盖修正。
- 所有变更保持向后兼容，并提供详细变更日志与配置示例。