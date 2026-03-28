"""
可视化模块
提供路单显示、统计图表等可视化功能
"""

from colorama import Fore, Back, Style, init
from utils import result_to_chinese, format_percentage_bar

# 初始化colorama
init(autoreset=True)


class ConsoleVisualizer:
    """控制台可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        pass
    
    def print_header(self, title):
        """打印标题"""
        width = 60
        print("\n" + "=" * width)
        print(f"{title:^{width}}")
        print("=" * width)
    
    def print_section(self, title):
        """打印章节标题"""
        print(f"\n{Fore.CYAN}{'─' * 60}")
        print(f"{Fore.CYAN}{title}")
        print(f"{Fore.CYAN}{'─' * 60}{Style.RESET_ALL}")
    
    def print_road_map(self, data, max_rows=6, max_cols=20):
        """
        打印路单（大路）
        
        Args:
            data: 数据列表
            max_rows: 最大行数
            max_cols: 最大列数
        """
        if not data:
            print("暂无数据")
            return
        
        # 去除和局
        data_no_tie = [x for x in data if x != 'T']
        
        if not data_no_tie:
            print("暂无有效数据（全是和局）")
            return
        
        # 构建路单矩阵
        road = []
        current_column = []
        last_result = None
        
        for result in data_no_tie:
            if result == last_result or last_result is None:
                current_column.append(result)
            else:
                if current_column:
                    road.append(current_column)
                current_column = [result]
            last_result = result
        
        if current_column:
            road.append(current_column)
        
        # 显示最近的列
        road = road[-max_cols:] if len(road) > max_cols else road
        
        # 转换为显示矩阵
        matrix = [[' ' for _ in range(len(road))] for _ in range(max_rows)]
        
        for col_idx, column in enumerate(road):
            for row_idx, result in enumerate(column):
                if row_idx < max_rows:
                    matrix[row_idx][col_idx] = result
        
        # 打印路单
        self.print_section("📊 路单图（大路）")
        
        for row in matrix:
            row_str = ""
            for cell in row:
                if cell == 'B':
                    row_str += f"{Fore.RED}●{Style.RESET_ALL} "
                elif cell == 'P':
                    row_str += f"{Fore.BLUE}●{Style.RESET_ALL} "
                else:
                    row_str += "  "
            print(f"  {row_str}")
        
        print(f"\n  {Fore.RED}● 庄{Style.RESET_ALL}  {Fore.BLUE}● 闲{Style.RESET_ALL}")
    
    def print_recent_results(self, data, count=20):
        """
        打印最近结果
        
        Args:
            data: 数据列表
            count: 显示数量
        """
        if not data:
            print("暂无数据")
            return
        
        recent = data[-count:] if len(data) > count else data
        
        self.print_section(f"📝 最近 {len(recent)} 局结果")
        
        result_str = ""
        for i, result in enumerate(recent):
            if result == 'B':
                result_str += f"{Fore.RED}庄{Style.RESET_ALL} "
            elif result == 'P':
                result_str += f"{Fore.BLUE}闲{Style.RESET_ALL} "
            else:
                result_str += f"{Fore.YELLOW}和{Style.RESET_ALL} "
            
            if (i + 1) % 10 == 0:
                result_str += "\n  "
        
        print(f"  {result_str}")
    
    def print_statistics(self, data):
        """
        打印统计信息
        
        Args:
            data: 数据列表
        """
        if not data:
            print("暂无数据")
            return
        
        from utils import calculate_statistics, get_current_streak, get_max_streak
        
        stats = calculate_statistics(data)
        
        self.print_section("📈 统计信息")
        
        print(f"  总局数: {stats['total']} 局")
        print()
        print(f"  庄: {stats['banker_count']} 局")
        print(f"     {format_percentage_bar(stats['banker_rate'])}")
        print()
        print(f"  闲: {stats['player_count']} 局")
        print(f"     {format_percentage_bar(stats['player_rate'])}")
        print()
        print(f"  和: {stats['tie_count']} 局")
        print(f"     {format_percentage_bar(stats['tie_rate'])}")
        print()
        
        # 连胜信息
        streak_type, streak_count = get_current_streak(data)
        if streak_type:
            streak_cn = result_to_chinese(streak_type)
            print(f"  当前: {Fore.YELLOW}{streak_cn} 连 {streak_count} 次{Style.RESET_ALL}")
        
        max_b = get_max_streak(data, 'B')
        max_p = get_max_streak(data, 'P')
        print(f"  最长: 庄 {max_b} 次 | 闲 {max_p} 次")
    
    def print_prediction(self, prediction):
        """
        打印预测结果
        
        Args:
            prediction: 预测字典
        """
        self.print_section("🎯 预测结果")
        
        # 推荐
        rec = prediction['recommendation_cn']
        strength = prediction['strength_desc']
        confidence = prediction['confidence']
        
        print(f"\n  {Fore.GREEN}{'=' * 50}{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}{Style.BRIGHT}推荐投注: {rec} ({strength}){Style.RESET_ALL}")
        print(f"  {Fore.GREEN}置信度: {confidence:.1f}%{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}{'=' * 50}{Style.RESET_ALL}\n")
        
        # 概率分布
        print("  概率分布:")
        print(f"    庄: {format_percentage_bar(prediction['B'], width=30)}")
        print(f"    闲: {format_percentage_bar(prediction['P'], width=30)}")
        print(f"    和: {format_percentage_bar(prediction['T'], width=30)}")
        print()
        
        # 分析原因
        if 'reason' in prediction:
            print(f"  分析: {prediction['reason']}")
        
        # 决策说明（如果有矛盾）
        if prediction.get('decision_note'):
            print(f"  {Fore.YELLOW}💡 {prediction['decision_note']}{Style.RESET_ALL}")
        
        print()
        
        # 模型详情
        if 'model_count' in prediction:
            print(f"  使用模型数: {prediction['model_count']}")
            print(f"  模型一致性: {prediction.get('consistency', 0):.1f}%")
        
        # 投票情况
        if 'votes' in prediction:
            votes = prediction['votes']
            vote_winner = prediction.get('vote_winner', '')
            prob_winner = prediction.get('prob_winner', '')
            is_tie_vote = prediction.get('is_tie_vote', False)
            
            vote_str = f"庄 {votes['B']} | 闲 {votes['P']} | 和 {votes['T']}"
            
            # 显示投票结果
            if is_tie_vote:
                # 平票情况
                max_vote = max(votes.values())
                tie_options = [self._result_cn(k) for k, v in votes.items() if v == max_vote]
                print(f"  模型投票: {vote_str} {Fore.YELLOW}({'/'.join(tie_options)}平票){Style.RESET_ALL}")
            elif vote_winner != prob_winner:
                # 投票和概率不一致
                print(f"  模型投票: {vote_str} {Fore.YELLOW}(多数支持{self._result_cn(vote_winner)}){Style.RESET_ALL}")
            else:
                print(f"  模型投票: {vote_str}")
    
    def _result_cn(self, result):
        """辅助方法：转换结果为中文"""
        mapping = {'B': '庄', 'P': '闲', 'T': '和'}
        return mapping.get(result, result)
    
    def print_shoe_analysis(self, shoe_analysis):
        """
        打印靴牌分析（阶段3新增）
        
        Args:
            shoe_analysis: 靴牌分析字典
        """
        if not shoe_analysis:
            return
        
        self.print_section("👞 靴牌特征分析")
        
        # 1. 靴牌类型
        shoe_type = shoe_analysis.get('shoe_type', {})
        if shoe_type:
            print(f"\n  {Fore.CYAN}▸ 靴牌类型:{Style.RESET_ALL} {shoe_type.get('characteristics', '未知')}")
            
            stats = shoe_type.get('stats', {})
            if stats:
                print(f"    庄: {stats.get('banker_rate', 0):.1f}% | 闲: {stats.get('player_rate', 0):.1f}% | 和: {stats.get('tie_rate', 0):.1f}%")
                if stats.get('long_streaks', 0) > 0:
                    print(f"    长龙数: {stats.get('long_streaks', 0)} 个")
        
        # 2. 靴牌阶段
        shoe_phases = shoe_analysis.get('shoe_phases', {})
        if shoe_phases:
            current_phase = shoe_phases.get('current_phase', 'unknown')
            phase_names = {'early': '前期', 'middle': '中期', 'late': '后期'}
            print(f"\n  {Fore.CYAN}▸ 当前阶段:{Style.RESET_ALL} {phase_names.get(current_phase, '未知')} ({shoe_phases.get('total_count', 0)}局)")
            
            for phase in ['early', 'middle', 'late']:
                phase_data = shoe_phases.get(phase)
                if phase_data:
                    phase_cn = phase_names[phase]
                    print(f"    {phase_cn}({phase_data['range']}): {phase_data['characteristics']}")
        
        # 3. 相似靴牌
        similar_shoes = shoe_analysis.get('similar_shoes', [])
        if similar_shoes:
            print(f"\n  {Fore.CYAN}▸ 相似历史靴牌:{Style.RESET_ALL}")
            for i, sim in enumerate(similar_shoes[:3], 1):
                shoe = sim['shoe']
                similarity = sim['similarity'] * 100
                sim_type = sim['type']
                print(f"    {i}. {shoe['name']} (相似度{similarity:.0f}%) - {sim_type.get('characteristics', '')}")
        
        print()
    
    def print_anomaly_detection(self, anomaly_result):
        """
        打印异常检测信息（2025-10-31新增）
        
        Args:
            anomaly_result: 异常检测字典
        """
        if not anomaly_result:
            return
        
        # 如果是数据不足，不显示
        if anomaly_result.get('severity_level') == 'insufficient_data':
            return
        
        # 如果没有异常，显示简单提示
        if not anomaly_result.get('is_anomaly'):
            self.print_section("🔍 异常检测")
            print(f"\n  {Fore.GREEN}✅ 靴牌数据正常{Style.RESET_ALL}")
            print()
            return
        
        # 有异常，显示详细信息
        self.print_section("🔍 异常检测")
        
        severity_level = anomaly_result.get('severity_level', 'unknown')
        severity_score = anomaly_result.get('severity_score', 0)
        
        # 根据严重程度选择颜色
        if severity_level == 'critical':
            color = Fore.RED
            emoji = '⛔'
        elif severity_level == 'high':
            color = Fore.RED
            emoji = '🚫'
        elif severity_level == 'moderate':
            color = Fore.YELLOW
            emoji = '⚠️'
        else:
            color = Fore.YELLOW
            emoji = '💡'
        
        print(f"\n  {color}{emoji} 检测到异常 (严重度: {severity_score}){Style.RESET_ALL}")
        
        # 显示具体异常项
        anomalies = anomaly_result.get('anomalies', [])
        if anomalies:
            print(f"\n  {Fore.CYAN}异常项目:{Style.RESET_ALL}")
            for anomaly in anomalies:
                print(f"    • {anomaly}")
        
        # 显示建议
        recommendation = anomaly_result.get('recommendation', '')
        if recommendation:
            print(f"\n  {color}{recommendation}{Style.RESET_ALL}")
        
        # 显示置信度调整
        adjustment = anomaly_result.get('confidence_adjustment', 1.0)
        if adjustment < 1.0:
            print(f"\n  {Fore.YELLOW}💡 置信度已自动调整至 {adjustment*100:.0f}%{Style.RESET_ALL}")
        
        print()
    
    def print_betting_suggestion(self, suggestion):
        """
        打印投注建议
        
        Args:
            suggestion: 投注建议字典
        """
        self.print_section("💰 投注建议")
        
        print(f"  投注目标: {Fore.YELLOW}{suggestion['target']}{Style.RESET_ALL}")
        print(f"  建议金额: {Fore.YELLOW}{suggestion['suggested_amount']:.2f}{Style.RESET_ALL}")
        print(f"  风险等级: {suggestion['risk_level']}")
        print(f"  推荐强度: {suggestion['strength']}")
        
        if suggestion.get('warning'):
            print(f"\n  {Fore.RED}⚠️  {suggestion['warning']}{Style.RESET_ALL}")
    
    def print_individual_predictions(self, predictions):
        """
        打印各个模型的预测
        
        Args:
            predictions: 预测列表
        """
        self.print_section("🤖 各模型预测详情")
        
        for pred_info in predictions:
            name = pred_info['name']
            weight = pred_info['weight']
            pred = pred_info['prediction']
            
            print(f"\n  【{name}】(权重: {weight})")
            
            # 找出最高概率
            max_prob = max(pred['B'], pred['P'], pred.get('T', 0))
            if pred['B'] == max_prob:
                rec = '庄'
            elif pred['P'] == max_prob:
                rec = '闲'
            else:
                rec = '和'
            
            print(f"    预测: {rec} | 置信度: {pred['confidence']:.1f}%")
            print(f"    概率: 庄 {pred['B']:.1f}% | 闲 {pred['P']:.1f}% | 和 {pred.get('T', 0):.1f}%")
            
            if 'reason' in pred:
                print(f"    原因: {pred['reason']}")
    
    def print_menu(self, options):
        """
        打印菜单
        
        Args:
            options: 选项列表
        """
        print("\n" + "─" * 60)
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        print("─" * 60)
    
    def print_data_stats(self, stats):
        """
        打印数据统计
        
        Args:
            stats: 统计字典
        """
        print(f"\n  当前靴牌: {stats['current_shoe_count']} 局")
        print(f"  历史数据: {stats['history_count']} 局")
        print(f"  总计数据: {stats['total_count']} 局")
        print(f"  历史靴数: {stats['total_shoes']} 靴")
        
        if stats['has_enough_for_ml']:
            print(f"  {Fore.GREEN}✓ 数据充足，已启用高级ML模型{Style.RESET_ALL}")
        else:
            print(f"  {Fore.YELLOW}⚠ 数据不足300局，部分ML模型未启用{Style.RESET_ALL}")
    
    def print_welcome(self):
        """打印欢迎信息"""
        print("\n" + "=" * 60)
        print(f"{Fore.CYAN}{Style.BRIGHT}")
        print("     ____                                  _   ")
        print("    | __ )  __ _  ___ ___ __ _ _ __ __ _| |_ ")
        print("    |  _ \\ / _` |/ __/ __/ _` | '__/ _` | __|")
        print("    | |_) | (_| | (_| (_| (_| | | | (_| | |_ ")
        print("    |____/ \\__,_|\\___\\___\\__,_|_|  \\__,_|\\__|")
        print()
        print("           智能预测分析系统 v1.0")
        print(f"{Style.RESET_ALL}")
        print("=" * 60)
        print(f"\n  {Fore.YELLOW}⚠️  本系统仅供学习研究，请理性娱乐{Style.RESET_ALL}\n")
    
    def print_error(self, message):
        """打印错误信息"""
        print(f"\n  {Fore.RED}❌ 错误: {message}{Style.RESET_ALL}\n")
    
    def print_success(self, message):
        """打印成功信息"""
        print(f"\n  {Fore.GREEN}✓ {message}{Style.RESET_ALL}\n")
    
    def print_warning(self, message):
        """打印警告信息"""
        print(f"\n  {Fore.YELLOW}⚠️  {message}{Style.RESET_ALL}\n")
    
    def print_info(self, message):
        """打印提示信息"""
        print(f"\n  {Fore.CYAN}ℹ️  {message}{Style.RESET_ALL}\n")

