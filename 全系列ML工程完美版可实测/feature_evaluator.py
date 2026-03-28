"""
特征评估器
评估候选特征的重要性，精选最佳特征
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from feature_extractor import FeatureExtractor


class FeatureEvaluator:
    """
    特征评估器
    
    使用多种方法评估特征重要性：
    1. 互信息（Mutual Information）
    2. 随机森林特征重要性
    3. 置换重要性（Permutation Importance）
    """
    
    def __init__(self):
        """初始化评估器"""
        self.extractor = FeatureExtractor()
        self.feature_names = self.extractor.feature_names
    
    def evaluate_features(self, shoes_data, target_k=12):
        """
        评估所有候选特征，精选Top K
        
        Args:
            shoes_data (list): 所有靴牌数据列表
                [{'name': 'shoe1', 'data': 'BBPPTP...'}, ...]
            target_k (int): 目标特征数量
            
        Returns:
            dict: 评估结果
        """
        print(f"开始评估 {len(self.feature_names)} 个候选特征...")
        print(f"使用 {len(shoes_data)} 靴牌数据进行评估\n")
        
        # 1. 构建训练数据集
        X, y = self._build_dataset(shoes_data)
        
        if len(X) < 50:
            print(f"⚠️ 警告：样本数量较少（{len(X)}），评估结果可能不够准确")
        
        print(f"✓ 数据集构建完成：{len(X)} 个样本，{len(self.feature_names)} 个特征\n")
        
        # 2. 计算3种重要性分数
        print("正在计算特征重要性...")
        mi_scores = self._mutual_information_scores(X, y)
        rf_scores = self._random_forest_scores(X, y)
        perm_scores = self._permutation_scores(X, y)
        
        # 3. 相关性分析
        print("正在分析特征相关性...")
        correlation_matrix = self._correlation_analysis(X)
        
        # 4. 综合评分
        print("正在计算综合评分...\n")
        final_scores = self._combine_scores(mi_scores, rf_scores, perm_scores)
        
        # 5. 移除高度相关的冗余特征
        selected_features = self._select_features(
            final_scores,
            correlation_matrix,
            target_k
        )
        
        return {
            'selected_features': selected_features,
            'all_scores': final_scores,
            'mi_scores': mi_scores,
            'rf_scores': rf_scores,
            'perm_scores': perm_scores,
            'correlation_matrix': correlation_matrix,
            'dataset_info': {
                'samples': len(X),
                'shoes': len(shoes_data),
                'features': len(self.feature_names)
            }
        }
    
    def _build_dataset(self, shoes_data):
        """
        构建训练数据集
        
        从每个靴牌中提取多个样本
        """
        X_list = []
        y_list = []
        
        for shoe in shoes_data:
            shoe_data_str = shoe['data']
            shoe_data_list = list(shoe_data_str)
            
            # 从每个靴牌中提取样本
            # 每隔3局提取一次特征，预测下一局
            for i in range(10, len(shoe_data_list) - 1, 3):
                # 当前数据
                current_data = shoe_data_list[:i]
                
                # 提取特征
                features = self.extractor.extract_features(current_data)
                feature_vector = [features[name] for name in self.feature_names]
                
                # 目标：下一局结果
                next_result = shoe_data_list[i]
                
                # 只预测B/P，跳过和局
                if next_result in ['B', 'P']:
                    X_list.append(feature_vector)
                    y_list.append(1 if next_result == 'B' else 0)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        return X, y
    
    def _mutual_information_scores(self, X, y):
        """计算互信息分数"""
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # 归一化到0-1
        if mi_scores.max() > 0:
            mi_scores = mi_scores / mi_scores.max()
        
        return {name: score for name, score in zip(self.feature_names, mi_scores)}
    
    def _random_forest_scores(self, X, y):
        """计算随机森林特征重要性"""
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        
        # 归一化到0-1
        if importances.max() > 0:
            importances = importances / importances.max()
        
        return {name: score for name, score in zip(self.feature_names, importances)}
    
    def _permutation_scores(self, X, y):
        """计算置换重要性"""
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        result = permutation_importance(
            rf, X, y,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        
        importances = result.importances_mean
        
        # 归一化到0-1
        if importances.max() > 0:
            importances = importances / importances.max()
        
        return {name: score for name, score in zip(self.feature_names, importances)}
    
    def _correlation_analysis(self, X):
        """分析特征相关性"""
        df = pd.DataFrame(X, columns=self.feature_names)
        correlation_matrix = df.corr().abs()
        return correlation_matrix
    
    def _combine_scores(self, mi_scores, rf_scores, perm_scores):
        """
        综合3种评分
        
        权重：互信息35% + 随机森林35% + 置换重要性30%
        """
        final_scores = {}
        
        for name in self.feature_names:
            combined = (
                mi_scores[name] * 0.35 +
                rf_scores[name] * 0.35 +
                perm_scores[name] * 0.30
            )
            final_scores[name] = combined
        
        return final_scores
    
    def _select_features(self, scores, correlation_matrix, target_k):
        """
        精选特征
        
        策略：
        1. 按综合得分排序
        2. 移除与已选特征高度相关(>0.8)的特征
        3. 保留Top K
        """
        # 按分数排序
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        selected = []
        
        for feature_name, score in sorted_features:
            # 检查是否与已选特征高度相关
            is_redundant = False
            
            for selected_feature in selected:
                correlation = correlation_matrix.loc[feature_name, selected_feature]
                if correlation > 0.8:
                    is_redundant = True
                    break
            
            # 如果不冗余，加入选择
            if not is_redundant:
                selected.append(feature_name)
            
            # 达到目标数量，停止
            if len(selected) >= target_k:
                break
        
        return selected
    
    def print_evaluation_report(self, evaluation_result):
        """打印评估报告"""
        selected = evaluation_result['selected_features']
        all_scores = evaluation_result['all_scores']
        mi_scores = evaluation_result['mi_scores']
        rf_scores = evaluation_result['rf_scores']
        perm_scores = evaluation_result['perm_scores']
        
        print("=" * 80)
        print("特征评估报告")
        print("=" * 80)
        
        # 数据集信息
        info = evaluation_result['dataset_info']
        print(f"\n数据集信息:")
        print(f"  靴牌数量: {info['shoes']} 靴")
        print(f"  样本数量: {info['samples']} 个")
        print(f"  候选特征: {info['features']} 个")
        
        # 精选特征
        print(f"\n{'=' * 80}")
        print(f"✅ 精选特征（Top {len(selected)}）")
        print("=" * 80)
        
        descriptions = self.extractor.get_feature_descriptions()
        
        for i, feature_name in enumerate(selected, 1):
            desc = descriptions[feature_name]
            combined_score = all_scores[feature_name]
            mi = mi_scores[feature_name]
            rf = rf_scores[feature_name]
            perm = perm_scores[feature_name]
            
            print(f"\n{i}. {feature_name}")
            print(f"   描述: {desc}")
            print(f"   综合得分: {combined_score:.3f}")
            print(f"   - 互信息: {mi:.3f}")
            print(f"   - 随机森林: {rf:.3f}")
            print(f"   - 置换重要性: {perm:.3f}")
        
        # 未选中的高分特征（可能因为相关性被排除）
        print(f"\n{'=' * 80}")
        print("⚠️ 未选中的高分特征（可能因相关性被排除）")
        print("=" * 80)
        
        sorted_all = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        excluded_high_scores = [
            (name, score) for name, score in sorted_all
            if name not in selected and score > 0.5
        ]
        
        if excluded_high_scores:
            for name, score in excluded_high_scores[:5]:
                desc = descriptions[name]
                print(f"\n  • {name}")
                print(f"    描述: {desc}")
                print(f"    得分: {score:.3f}")
        else:
            print("\n  无")
        
        # 相关性警告
        corr_matrix = evaluation_result['correlation_matrix']
        print(f"\n{'=' * 80}")
        print("📊 特征相关性检查")
        print("=" * 80)
        
        high_corr_pairs = []
        for i, feat1 in enumerate(selected):
            for feat2 in selected[i+1:]:
                corr = corr_matrix.loc[feat1, feat2]
                if corr > 0.7:
                    high_corr_pairs.append((feat1, feat2, corr))
        
        if high_corr_pairs:
            print("\n⚠️ 发现高相关性特征对（>0.7）：")
            for feat1, feat2, corr in high_corr_pairs:
                print(f"  • {feat1} <-> {feat2}: {corr:.2f}")
            print("\n建议: 考虑移除其中一个")
        else:
            print("\n✅ 所有特征之间相关性<0.7，良好！")
        
        print("\n" + "=" * 80)
        print("✅ 评估完成")
        print("=" * 80)

