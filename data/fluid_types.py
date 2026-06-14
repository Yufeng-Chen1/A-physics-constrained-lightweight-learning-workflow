#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
储层流体类型定义
定义所有支持的储层流体类型及其属性
"""

import pandas as pd

class FluidTypes:
    """储层流体类型定义类"""
    
    # 流体类型定义（5个主要类别：油层、水层、干层、差油层、油水层）
    # 注意：非这5类的标签在数据预处理时会被过滤，不参与模型训练
    FLUID_TYPES = {
        0: "油层",           # Oil Layer
        1: "水层",           # Water Layer  
        2: "干层",           # Dry Layer
        3: "差油层",         # Poor Oil Layer
        4: "油水层",         # Oil-Water Layer (merged: 油水同层+含油水层)
    }
    
    # 英文名称映射
    FLUID_TYPES_EN = {
        0: "Oil_Layer",
        1: "Water_Layer",
        2: "Dry_Layer", 
        3: "Poor_Oil_Layer",
        4: "Oil_Water_Layer",
    }
    
    # 流体类型描述
    FLUID_DESCRIPTIONS = {
        0: "纯油层，含油饱和度较高，不含水或含水很少",
        1: "纯水层，含水饱和度较高，不含油或含油很少", 
        2: "干层，基本不含流体或含流体很少",
        3: "差油层，含油饱和度较低，可能含有一定水分",
        4: "油水层，油和水同时存在（油水同层+含油水层合并）",
    }
    
    # 流体类型颜色映射（用于可视化）
    FLUID_COLORS = {
        0: "#FF6B6B",  # 红色 - 油层
        1: "#4ECDC4",  # 青色 - 水层
        2: "#95A5A6",  # 灰色 - 干层
        3: "#F39C12",  # 橙色 - 差油层
        4: "#9B59B6",  # 紫色 - 油水层
    }
    
    # 流体类型优先级（用于排序）
    FLUID_PRIORITY = {
        0: 1,  # 油层 - 最高优先级
        4: 2,  # 油水层 - 第二优先级
        3: 3,  # 差油层 - 第三优先级
        1: 4,  # 水层 - 第四优先级
        2: 5,  # 干层 - 第五优先级
    }
    
    @classmethod
    def get_fluid_name(cls, class_id: int) -> str:
        """获取流体类型名称"""
        return cls.FLUID_TYPES.get(class_id, "未知类型")
    
    @classmethod
    def get_fluid_name_en(cls, class_id: int) -> str:
        """获取流体类型英文名称"""
        return cls.FLUID_TYPES_EN.get(class_id, "Unknown")
    
    @classmethod
    def get_fluid_description(cls, class_id: int) -> str:
        """获取流体类型描述"""
        return cls.FLUID_DESCRIPTIONS.get(class_id, "未知流体类型")
    
    @classmethod
    def get_fluid_color(cls, class_id: int) -> str:
        """获取流体类型颜色"""
        return cls.FLUID_COLORS.get(class_id, "#95A5A6")
    
    @classmethod
    def get_fluid_priority(cls, class_id: int) -> int:
        """获取流体类型优先级"""
        return cls.FLUID_PRIORITY.get(class_id, 999)
    
    @classmethod
    def get_all_fluid_names(cls) -> list:
        """获取所有流体类型名称"""
        return list(cls.FLUID_TYPES.values())
    
    @classmethod
    def get_all_fluid_ids(cls) -> list:
        """获取所有流体类型ID"""
        return list(cls.FLUID_TYPES.keys())
    
    @classmethod
    def get_num_classes(cls) -> int:
        """获取流体类型总数"""
        return len(cls.FLUID_TYPES)
    
    @classmethod
    def get_fluid_id_by_name(cls, name: str) -> int:
        """根据名称获取流体类型ID"""
        for fluid_id, fluid_name in cls.FLUID_TYPES.items():
            if fluid_name == name:
                return fluid_id
        return -1
    
    @classmethod
    def get_label_id(cls, label_str: str) -> int:
        """根据标签字符串获取流体类型ID"""
        if not label_str or pd.isna(label_str):
            return None
        
        label_clean = str(label_str).strip()
        
        # 直接匹配
        for fluid_id, fluid_name in cls.FLUID_TYPES.items():
            if fluid_name == label_clean:
                return fluid_id
        
        # 模糊匹配
        if '油层' in label_clean and '差' not in label_clean and '水' not in label_clean:
            return 0  # 油层
        elif '水层' in label_clean and '油' not in label_clean:
            return 1  # 水层
        elif '干层' in label_clean:
            return 2  # 干层
        elif '差油层' in label_clean or ('差' in label_clean and '油' in label_clean):
            return 3  # 差油层
        elif ('油水' in label_clean) or ('含油水' in label_clean) or ('油水同层' in label_clean):
            return 4  # 油水层
        
        return None
    
    @classmethod
    def is_valid_fluid_id(cls, class_id: int) -> bool:
        """检查流体类型ID是否有效"""
        return class_id in cls.FLUID_TYPES
    
    @classmethod
    def get_fluid_summary(cls) -> dict:
        """获取流体类型摘要信息"""
        return {
            'num_classes': cls.get_num_classes(),
            'fluid_types': cls.FLUID_TYPES,
            'descriptions': cls.FLUID_DESCRIPTIONS,
            'colors': cls.FLUID_COLORS,
            'priorities': cls.FLUID_PRIORITY
        }
    
    @classmethod
    def print_fluid_types(cls):
        """打印所有流体类型信息"""
        print("🏷️  储层流体类型定义:")
        print("=" * 60)
        for fluid_id, fluid_name in cls.FLUID_TYPES.items():
            description = cls.FLUID_DESCRIPTIONS[fluid_id]
            color = cls.FLUID_COLORS[fluid_id]
            priority = cls.FLUID_PRIORITY[fluid_id]
            print(f"   {fluid_id}: {fluid_name}")
            print(f"      描述: {description}")
            print(f"      颜色: {color}")
            print(f"      优先级: {priority}")
            print()
        print(f"   总计: {cls.get_num_classes()} 个流体类型")


# 创建全局实例
fluid_types = FluidTypes()

# 导出常用函数
def get_fluid_name(class_id: int) -> str:
    """获取流体类型名称"""
    return FluidTypes.get_fluid_name(class_id)

def get_fluid_color(class_id: int) -> str:
    """获取流体类型颜色"""
    return FluidTypes.get_fluid_color(class_id)

def get_num_fluid_classes() -> int:
    """获取流体类型总数"""
    return FluidTypes.get_num_classes()

def get_all_fluid_names() -> list:
    """获取所有流体类型名称"""
    return FluidTypes.get_all_fluid_names()

if __name__ == "__main__":
    # 测试流体类型定义
    FluidTypes.print_fluid_types()
    
    print("\n🧪 测试流体类型功能:")
    print(f"   流体类型总数: {get_num_fluid_classes()}")
    print(f"   所有流体类型: {get_all_fluid_names()}")
    
    # 测试单个流体类型
    for i in range(6):
        print(f"   类型 {i}: {get_fluid_name(i)} (颜色: {get_fluid_color(i)})")
