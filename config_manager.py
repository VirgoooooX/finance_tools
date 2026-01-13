"""
配置管理模块
负责读取、写入和验证配置文件
"""
import yaml
import os
from pathlib import Path

class ConfigManager:
    """配置管理器"""
    
    DEFAULT_CONFIG_FILE = "config.yaml"
    
    def __init__(self, config_path=None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认为当前目录下的config.yaml
        """
        if config_path is None:
            config_path = self.DEFAULT_CONFIG_FILE
        self.config_path = Path(config_path)
        self.config = self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        if not self.config_path.exists():
            print(f"⚠️ 配置文件不存在，使用默认配置: {self.config_path}")
            return self.get_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ 配置文件加载成功: {self.config_path}")
            return config
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            return self.get_default_config()
    
    def save_config(self, config=None):
        """保存配置到文件"""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
            print(f"✅ 配置文件保存成功: {self.config_path}")
            return True
        except Exception as e:
            print(f"❌ 配置文件保存失败: {e}")
            return False
    
    def get(self, key, default=None):
        """获取配置项"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def set(self, key, value):
        """设置配置项"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def get_account_keywords(self, account_name):
        """
        获取科目的关键字列表
        
        Args:
            account_name: 科目名称
            
        Returns:
            关键字列表，如果未找到返回[account_name]
        """
        mappings = self.config.get('科目映射', {})
        return mappings.get(account_name, [account_name])
    
    @staticmethod
    def get_default_config():
        """获取默认配置"""
        return {
            '科目映射': {
                '资产总计': ['资产总计', '资产总额', '资产合计'],
                '流动资产合计': ['流动资产合计', '流动资产总计'],
                '货币资金': ['货币资金', '现金及现金等价物'],
                '存货': ['存货'],
                '负债合计': ['负债合计', '负债总计'],
                '流动负债合计': ['流动负债合计', '流动负债总计'],
                '所有者权益合计': ['所有者权益合计', '股东权益合计', '权益合计'],
                '营业收入': ['营业收入', '主营业务收入'],
                '营业成本': ['营业成本', '主营业务成本'],
                '营业利润': ['营业利润'],
                '净利润': ['净利润'],
                '经营活动现金流': ['经营活动产生的现金流量净额', '经营活动现金流量净额'],
                '投资活动现金流': ['投资活动产生的现金流量净额'],
                '筹资活动现金流': ['筹资活动产生的现金流量净额'],
            },
            '输出选项': {
                '生成原始数据': True,
                '生成验证报告': True,
                '生成财务指标': True,
                '输出格式': 'excel',
                '输出文件名': '清洗后的AI标准财务表',
            },
            '验证选项': {
                '启用会计恒等式验证': True,
                '容差阈值': 0.01,
            },
            '指标选项': {
                '计算流动性指标': True,
                '计算偿债能力指标': True,
                '计算盈利能力指标': True,
                '计算现金流指标': True,
            }
        }
    
    def create_default_config_file(self):
        """创建默认配置文件"""
        self.config = self.get_default_config()
        return self.save_config()


if __name__ == '__main__':
    # 测试配置管理器
    config = ConfigManager()
    print("\n配置测试:")
    print(f"资产总计关键字: {config.get_account_keywords('资产总计')}")
    print(f"容差阈值: {config.get('验证选项.容差阈值')}")
