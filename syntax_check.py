#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语法检查脚本 - 验证代码结构是否正确
不依赖PyTorch等深度学习库
"""

import ast
import os
import sys

def check_python_syntax(file_path):
    """
    检查Python文件的语法是否正确
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # 尝试解析AST
        ast.parse(source_code)
        return True, "语法正确"
        
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    except Exception as e:
        return False, f"其他错误: {e}"

def check_imports(file_path):
    """
    检查文件中的导入语句
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        tree = ast.parse(source_code)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        return imports
        
    except Exception as e:
        return [f"解析导入失败: {e}"]

def check_class_definitions(file_path):
    """
    检查类定义
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        tree = ast.parse(source_code)
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(item.name)
                classes.append({
                    'name': node.name,
                    'methods': methods,
                    'bases': [base.id if hasattr(base, 'id') else str(base) for base in node.bases]
                })
        
        return classes
        
    except Exception as e:
        return [f"解析类定义失败: {e}"]

def main():
    """
    主检查函数
    """
    print("🔍 开始语法和结构检查")
    print("=" * 60)
    
    # 要检查的文件列表
    files_to_check = [
        'enhanced_gated_attention.py',
        'model.py',
        'train.py'
    ]
    
    all_passed = True
    
    for file_name in files_to_check:
        print(f"\n📁 检查文件: {file_name}")
        print("-" * 40)
        
        if not os.path.exists(file_name):
            print(f"❌ 文件不存在: {file_name}")
            all_passed = False
            continue
        
        # 语法检查
        syntax_ok, syntax_msg = check_python_syntax(file_name)
        if syntax_ok:
            print(f"✅ 语法检查: {syntax_msg}")
        else:
            print(f"❌ 语法检查: {syntax_msg}")
            all_passed = False
            continue
        
        # 导入检查
        imports = check_imports(file_name)
        print(f"📦 导入模块数量: {len(imports)}")
        if len(imports) <= 10:  # 只显示前10个导入
            for imp in imports[:10]:
                print(f"   - {imp}")
        else:
            for imp in imports[:5]:
                print(f"   - {imp}")
            print(f"   ... 还有 {len(imports) - 5} 个导入")
        
        # 类定义检查
        classes = check_class_definitions(file_name)
        if classes:
            print(f"🏗️  类定义数量: {len(classes)}")
            for cls in classes:
                if isinstance(cls, dict):
                    print(f"   类: {cls['name']}")
                    print(f"     继承: {cls['bases']}")
                    print(f"     方法数: {len(cls['methods'])}")
                    if cls['methods']:
                        print(f"     主要方法: {', '.join(cls['methods'][:3])}{'...' if len(cls['methods']) > 3 else ''}")
                else:
                    print(f"   {cls}")
    
    # 特殊检查：验证关键类是否存在
    print("\n🎯 关键组件检查")
    print("-" * 40)
    
    # 检查enhanced_gated_attention.py中的关键类
    if os.path.exists('enhanced_gated_attention.py'):
        classes = check_class_definitions('enhanced_gated_attention.py')
        key_classes = ['ClusterContrastiveLoss', 'EnhancedGatedAttention']
        
        for key_class in key_classes:
            found = False
            for cls in classes:
                if isinstance(cls, dict) and cls['name'] == key_class:
                    print(f"✅ 找到关键类: {key_class}")
                    found = True
                    break
            if not found:
                print(f"❌ 缺少关键类: {key_class}")
                all_passed = False
    
    # 检查model.py中的集成
    if os.path.exists('model.py'):
        try:
            with open('model.py', 'r', encoding='utf-8') as f:
                model_content = f.read()
            
            if 'EnhancedGatedAttention' in model_content:
                print("✅ model.py已集成EnhancedGatedAttention")
            else:
                print("❌ model.py未集成EnhancedGatedAttention")
                all_passed = False
                
            if 'enhanced_gated_attention' in model_content:
                print("✅ model.py已导入enhanced_gated_attention模块")
            else:
                print("❌ model.py未导入enhanced_gated_attention模块")
                all_passed = False
                
        except Exception as e:
            print(f"❌ 检查model.py集成时出错: {e}")
            all_passed = False
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 检查结果汇总")
    print("=" * 60)
    
    if all_passed:
        print("🎉 所有语法和结构检查通过！")
        print("\n📝 代码结构验证成功，主要组件：")
        print("   ✅ ClusterContrastiveLoss - 聚类对比损失")
        print("   ✅ EnhancedGatedAttention - 增强门控注意力")
        print("   ✅ 模型集成 - Transformer_Based_Model")
        print("   ✅ 训练脚本 - train.py")
        
        print("\n🚀 下一步建议：")
        print("   1. 在有PyTorch环境的机器上运行完整测试")
        print("   2. 使用小批次数据验证训练流程")
        print("   3. 监控聚类对比损失的收敛情况")
        print("   4. 对比启用/禁用聚类对比学习的性能差异")
        
        return True
    else:
        print("⚠️  发现语法或结构问题，请检查上述错误信息")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)