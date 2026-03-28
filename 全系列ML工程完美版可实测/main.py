"""
百家乐智能预测分析系统 - 主程序
"""

import sys
from data_manager import DataManager
from ensemble import EnsemblePredictor
from visualizer import ConsoleVisualizer
from utils import normalize_input
from prediction_history import PredictionHistory


class BaccaratPredictor:
    """百家乐预测系统主类"""
    
    def __init__(self):
        """初始化系统"""
        self.data_manager = DataManager()
        self.visualizer = ConsoleVisualizer()
        self.running = True
        self.prediction_history = PredictionHistory()  # 预测历史记录器
        self.last_ensemble = None  # 最后一次的预测器实例
        
        # 渐进式缓存（阶段2-A新增）
        self._analysis_cache = None  # 缓存的分析结果
        self._cache_round_count = 0  # 缓存时的局数
        self._cache_update_interval = 10  # 每10局更新一次缓存
        
        # Ensemble缓存计数（避免重复训练ML模型）
        self._last_shoes_count = 0
        self._last_history_count = 0
    
    def run(self):
        """运行主程序"""
        self.visualizer.print_welcome()
        
        # 显示数据统计
        stats = self.data_manager.get_statistics()
        self.visualizer.print_data_stats(stats)
        
        # 初始化缓存计数（启动时）
        self._update_cache_count()
        
        # 主循环
        while self.running:
            self.show_main_menu()
    
    def _update_cache_count(self):
        """更新缓存计数"""
        total_count = self.data_manager.get_statistics()['total_count']
        self._cache_round_count = total_count
    
    def _clear_analysis_cache(self):
        """清空分析缓存（当历史数据更新时）"""
        self._analysis_cache = None
        self._update_cache_count()
        # 历史数据更新时，也清空ensemble缓存（需要重新训练）
        self.last_ensemble = None
        self._last_shoes_count = 0
        self._last_history_count = 0
        print("  📊 缓存已更新")
    
    def show_main_menu(self):
        """显示主菜单"""
        options = [
            "实时记录模式（边玩边记）",
            "批量导入数据",
            "查看当前数据",
            "获取预测",
            "查看详细分析",
            "开始新靴牌",
            "数据管理",
            "运行历史回测",
            "退出系统"
        ]
        
        self.visualizer.print_menu(options)
        
        try:
            choice = input("\n  请选择功能 (1-9): ").strip()
            
            if choice == '1':
                self.real_time_mode()
            elif choice == '2':
                self.batch_import_mode()
            elif choice == '3':
                self.view_current_data()
            elif choice == '4':
                self.get_prediction()
            elif choice == '5':
                self.view_detailed_analysis()
            elif choice == '6':
                self.start_new_shoe()
            elif choice == '7':
                self.data_management()
            elif choice == '8':
                self.run_backtest()
            elif choice == '9':
                self.exit_system()
            else:
                self.visualizer.print_error("无效的选择，请输入1-9")
        
        except KeyboardInterrupt:
            print("\n")
            self.exit_system()
        except Exception as e:
            self.visualizer.print_error(f"发生错误: {e}")

    
    def _print_realtime_help(self):
        """打印实时记录模式的帮助信息"""
        print("\n  输入每局结果:")
        print("    B 或 庄 = 庄家赢")
        print("    P 或 闲 = 闲家赢")
        print("    T 或 和 = 和局")
        print("\n  其他指令:")
        print("    . = 查看详细分析（路单+统计+预测）")
        print("    s = 快速保存当前靴牌到历史")
        print("    u = 撤销上一局（输错了可以撤回）")
        print("    h 或 ? = 显示帮助")
        print("    q = 返回主菜单（退出时会询问是否保存）")
    
    def _show_quick_prediction(self):
        """显示快速预测（不需要按回车确认）"""
        try:
            current = self.data_manager.get_current_shoe()
            history = self.data_manager.all_history
            shoes = self.data_manager.get_shoes()  # 获取靴牌列表
            
            # 优化：检查是否需要重新创建ensemble（避免重复训练ML模型）
            # 只有在靴牌数量或历史数据变化时才重新创建
            need_recreate = (
                self.last_ensemble is None or
                len(shoes) != getattr(self, '_last_shoes_count', 0) or
                len(history) != getattr(self, '_last_history_count', 0)
            )
            
            if need_recreate:
                # 使用持久化的prediction_history
                print("  🔄 初始化预测模型（首次或数据更新）...")
                ensemble = EnsemblePredictor(history, current, shoes, self.prediction_history)
                self.last_ensemble = ensemble
                self._last_shoes_count = len(shoes)
                self._last_history_count = len(history)
                print("  ✓ 模型已准备就绪")
            else:
                # 重用现有ensemble，使用专用方法深层同步数据
                ensemble = self.last_ensemble
                ensemble.update_data(history, current)
            
            prediction = ensemble.predict_next()
            
            # 显示简洁的预测
            rec = prediction['recommendation_cn']
            confidence = prediction['confidence']
            strength = prediction['strength_desc']
            
            # 使用颜色高亮
            from colorama import Fore, Style
            
            print(f"\n  {Fore.GREEN}{'━' * 50}{Style.RESET_ALL}")
            print(f"  {Fore.YELLOW}⭐ 预测第 {len(current)+1} 局: {Fore.CYAN}{Style.BRIGHT}{rec}{Style.RESET_ALL} ({strength}, 置信度 {confidence:.1f}%)")
            print(f"  {Fore.GREEN}{'━' * 50}{Style.RESET_ALL}")
            
            # 概率分布
            print(f"  概率: 庄 {prediction['B']:.1f}%  |  闲 {prediction['P']:.1f}%  |  和 {prediction['T']:.1f}%")
            
            # 分析原因
            if prediction.get('reason'):
                reason = prediction['reason'][:60]  # 截断太长的原因
                print(f"  原因: {reason}")
            
            # 决策说明（如果概率和推荐不一致）
            if prediction.get('decision_note'):
                print(f"  {Fore.YELLOW}💡 {prediction['decision_note']}{Style.RESET_ALL}")
            
            # 显示权重调整信息（如果有）
            if prediction.get('weight_adjustment', {}).get('applied'):
                adj = prediction['weight_adjustment']
                print(f"  {Fore.YELLOW}🔧 {adj['reason']}{Style.RESET_ALL}")
            
            # 显示异常检测警告（如果有，2025-10-31新增）
            if prediction.get('anomaly_adjusted'):
                anomaly = prediction.get('anomaly_detection', {})
                if anomaly.get('is_anomaly'):
                    severity = anomaly.get('severity_level', '')
                    if severity in ['critical', 'high']:
                        color = Fore.RED
                        emoji = '⛔' if severity == 'critical' else '🚫'
                    else:
                        color = Fore.YELLOW
                        emoji = '⚠️'
                    
                    # 显示第一个异常项
                    first_anomaly = anomaly.get('anomalies', ['检测到异常'])[0]
                    print(f"  {color}{emoji} 异常检测: {first_anomaly}{Style.RESET_ALL}")
                    print(f"  {color}{anomaly.get('recommendation', '')}{Style.RESET_ALL}")
            
            print()
            
        except Exception as e:
            print(f"  预测出错: {e}")
    
    def real_time_mode(self):
        """实时记录模式"""
        self.visualizer.print_header("实时记录模式")
        
        self._print_realtime_help()
        print("  💡 每次输入结果后，系统会自动预测下一局！")
        print("  💡 输入 h 或 ? 可随时查看帮助")
        print("  💡 输入 s 可快速保存，输入 q 退出时也会询问保存")
        
        # 修复越界缓存：如果进来时是一靴新牌（0局），强制清空上一靴的预测缓存
        if len(self.data_manager.get_current_shoe()) == 0:
            self._clear_analysis_cache()
        
        while True:
            current_count = len(self.data_manager.get_current_shoe())
            next_round = current_count + 1  # 下一局的编号
            
            try:
                user_input = input(f"\n  [第{next_round}局] 输入结果: ").strip().upper()
                
                if not user_input:
                    continue
                
                if user_input == 'Q':
                    # 退出前询问是否保存
                    current = self.data_manager.get_current_shoe()
                    if current and len(current) > 0:
                        print(f"\n  当前靴牌有 {len(current)} 局数据")
                        save_choice = input("  是否保存到历史? (y/n): ").strip().lower()
                        
                        if save_choice == 'y':
                            shoe_name = input("  靴牌名称（留空自动生成）: ").strip()
                            shoe_name = shoe_name if shoe_name else None
                            
                            if self.data_manager.save_current_shoe(shoe_name):
                                self.visualizer.print_success("已保存当前靴牌到历史")
                                self._clear_analysis_cache()  # 清空缓存
                                # 不清空，可能用户还想继续使用这些数据
                            else:
                                self.visualizer.print_error("保存失败")
                        else:
                            print("  未保存，数据仍在当前靴牌中")
                    
                    break
                elif user_input in ['H', '?', 'HELP']:
                    # 显示帮助
                    print("\n  " + "─" * 50)
                    self._print_realtime_help()
                    print("  " + "─" * 50)
                elif user_input == 'S':
                    # 快速保存
                    current = self.data_manager.get_current_shoe()
                    if current and len(current) > 0:
                        print(f"\n  当前靴牌有 {len(current)} 局数据")
                        shoe_name = input("  靴牌名称（留空自动生成）: ").strip()
                        shoe_name = shoe_name if shoe_name else None
                        
                        if self.data_manager.save_current_shoe(shoe_name):
                            self.visualizer.print_success(f"已保存 {len(current)} 局数据到历史")
                            print("  数据已累积到历史库，高级ML模型会更准确！")
                            self._clear_analysis_cache()  # 清空缓存
                        else:
                            self.visualizer.print_error("保存失败")
                    else:
                        self.visualizer.print_warning("当前靴牌没有数据")
                elif user_input == '.':
                    # 查看详细分析
                    self.view_current_data(show_menu=False)
                    print("\n  " + "─" * 50)
                    self.get_prediction(show_menu=False)
                    input("\n  按回车继续...")
                elif user_input == 'U':
                    if self.data_manager.delete_last_result():
                        self.visualizer.print_success("已撤销上一局")
                    else:
                        self.visualizer.print_warning("没有可撤销的数据")
                else:
                    # 尝试添加结果
                    normalized = normalize_input(user_input)
                    
                    if normalized and len(normalized) == 1:
                        actual_result = normalized[0]
                        
                        # 先更新上一次预测的结果
                        if self.last_ensemble is not None:
                            self.last_ensemble.update_prediction_result(actual_result)
                            
                            # 打印上一局预测的准确率和连胜/连败记录
                            if self.last_ensemble.history.predictions:
                                last_pred = self.last_ensemble.history.predictions[-1]
                                # 只有当预测非中性(和局)时计算准确率
                                if not last_pred.get('neutral', False):
                                    from colorama import Fore, Style
                                    if last_pred['correct']:
                                        print(f"  {Fore.GREEN}🎯 预测正确!{Style.RESET_ALL}")
                                    else:
                                        print(f"  {Fore.RED}❌ 预测错误!{Style.RESET_ALL}")
                                        
                                    stats = self.last_ensemble.history.get_statistics()
                                    session_acc = stats.get('session_accuracy', 0) * 100
                                    session_valid = stats.get('session_valid', 0)
                                    session_correct = stats.get('session_correct', 0)
                                    cons_win = stats.get('consecutive_corrects', 0)
                                    cons_loss = stats.get('consecutive_errors', 0)
                                    
                                    if cons_win > 0:
                                        streak_msg = f"{Fore.GREEN}连对: {cons_win}{Style.RESET_ALL}"
                                    elif cons_loss > 0:
                                        streak_msg = f"{Fore.RED}连错: {cons_loss}{Style.RESET_ALL}"
                                    else:
                                        streak_msg = "无连胜败"
                                        
                                    print(f"  📊 会话胜率: {session_acc:.1f}% ({session_correct}/{session_valid}) | {streak_msg}")

                        # 然后添加新结果
                        if self.data_manager.add_result(actual_result):
                            result_cn = {'B': '庄', 'P': '闲', 'T': '和'}[actual_result]
                            print(f"\n  ✓ 已记录: {result_cn}")
                            
                            # 每次输入后自动显示预测（如果数据足够）
                            current = self.data_manager.get_current_shoe()
                            if len(current) >= 5:
                                print("\n  " + "─" * 50)
                                print(f"  📊 已有 {len(current)} 局数据，预测下一局...")
                                print("  " + "─" * 50)
                                self._show_quick_prediction()
                            elif len(current) < 5:
                                print(f"  ℹ️  已记录 {len(current)} 局，至少需要5局才能开始预测")
                        else:
                            self.visualizer.print_error("无效的输入")
                    else:
                        self.visualizer.print_error("请输入单个结果 (B/P/T)")
            
            except KeyboardInterrupt:
                print("\n")
                break
            except Exception as e:
                self.visualizer.print_error(f"发生错误: {e}")
    
    def batch_import_mode(self):
        """批量导入模式"""
        self.visualizer.print_header("批量导入数据")
        
        print("\n  支持的格式:")
        print("    - BBPPTBPPBPBP")
        print("    - 庄庄闲闲和庄闲闲庄闲庄闲")
        print("    - B,B,P,P,T,B,P,P,B,P,B,P")
        print("\n  输入数据（输入空行结束）:")
        
        lines = []
        while True:
            try:
                line = input("  ")
                if not line.strip():
                    break
                lines.append(line)
            except KeyboardInterrupt:
                print("\n")
                return
        
        if not lines:
            self.visualizer.print_warning("未输入任何数据")
            return
        
        text = ''.join(lines)
        success, count = self.data_manager.add_batch(text)
        
        if success:
            self.visualizer.print_success(f"成功导入 {count} 局数据")
            
            # 自动显示预测和分析
            if count >= 5:
                print("\n  正在分析...")
                
                # 显示当前数据概览
                current = self.data_manager.get_current_shoe()
                self.visualizer.print_recent_results(current, count=min(20, count))
                self.visualizer.print_road_map(current)
                self.visualizer.print_statistics(current)
                
                # 显示预测
                print("\n  " + "─" * 50)
                self.get_prediction(show_menu=False)
                
                input("\n  按回车返回主菜单...")
            else:
                self.visualizer.print_warning(f"数据太少（{count}局），至少需要5局才能预测")
        else:
            self.visualizer.print_error("导入失败，请检查数据格式")
    
    def view_current_data(self, show_menu=True):
        """查看当前数据"""
        if show_menu:
            self.visualizer.print_header("当前数据")
        
        current = self.data_manager.get_current_shoe()
        
        if not current:
            self.visualizer.print_warning("当前靴牌暂无数据")
            return
        
        # 显示最近结果
        self.visualizer.print_recent_results(current)
        
        # 显示路单
        self.visualizer.print_road_map(current)
        
        # 显示统计
        self.visualizer.print_statistics(current)
        
        if show_menu:
            input("\n  按回车继续...")
    
    def get_prediction(self, show_menu=True):
        """获取预测"""
        if show_menu:
            self.visualizer.print_header("获取预测")
        
        current = self.data_manager.get_current_shoe()
        all_data = self.data_manager.get_all_data()
        
        if len(current) < 5:
            self.visualizer.print_warning("数据太少（至少需要5局），预测可能不准确")
        
        if not all_data:
            self.visualizer.print_error("没有任何数据，无法预测")
            return
        
        # 优化：复用ensemble实例（避免重复训练）
        history = self.data_manager.all_history
        shoes = self.data_manager.get_shoes()
        
        need_recreate = (
            self.last_ensemble is None or
            len(shoes) != getattr(self, '_last_shoes_count', 0) or
            len(history) != getattr(self, '_last_history_count', 0)
        )
        
        if need_recreate:
            ensemble = EnsemblePredictor(history, current, shoes, self.prediction_history)
            self.last_ensemble = ensemble
            self._last_shoes_count = len(shoes)
            self._last_history_count = len(history)
        else:
            # Audit#B修复：使用 update_data 深层同步所有子模型数据
            ensemble = self.last_ensemble
            ensemble.update_data(history, current)

        # 获取预测
        print("\n  正在分析...")
        prediction = ensemble.predict_next()
        
        # 显示预测
        self.visualizer.print_prediction(prediction)
        
        # 显示靴牌分析（阶段3新增）
        if 'shoe_analysis' in prediction:
            self.visualizer.print_shoe_analysis(prediction['shoe_analysis'])
        
        # 显示异常检测（2025-10-31新增）
        if 'anomaly_detection' in prediction:
            self.visualizer.print_anomaly_detection(prediction['anomaly_detection'])
        
        # 显示投注建议
        suggestion = ensemble.get_betting_suggestion(bankroll=1000, risk_level='medium')
        self.visualizer.print_betting_suggestion(suggestion)
        
        # 询问是否查看详情
        if show_menu:
            show_detail = input("\n  是否查看各模型预测详情? (y/n): ").strip().lower()
            if show_detail == 'y':
                if 'individual_predictions' in prediction:
                    self.visualizer.print_individual_predictions(
                        prediction['individual_predictions']
                    )
            
            input("\n  按回车继续...")
    
    def view_detailed_analysis(self):
        """查看详细分析"""
        self.visualizer.print_header("详细分析报告")
        
        all_data = self.data_manager.get_all_data()
        
        if not all_data:
            self.visualizer.print_error("没有任何数据")
            return
        
        current = self.data_manager.get_current_shoe()
        history = self.data_manager.all_history
        shoes = self.data_manager.get_shoes()
        
        # 优化：复用ensemble实例
        need_recreate = (
            self.last_ensemble is None or
            len(shoes) != getattr(self, '_last_shoes_count', 0) or
            len(history) != getattr(self, '_last_history_count', 0)
        )
        
        if need_recreate:
            ensemble = EnsemblePredictor(history, current, shoes, self.prediction_history)
            self.last_ensemble = ensemble
            self._last_shoes_count = len(shoes)
            self._last_history_count = len(history)
        else:
            # Audit#B修复：使用 update_data 深层同步所有子模型数据
            ensemble = self.last_ensemble
            ensemble.update_data(history, current)

        print("\n  正在生成详细分析...")
        
        # 显示完整分析
        self.view_current_data(show_menu=False)
        
        # 获取预测
        self.get_prediction(show_menu=False)
        
        input("\n  按回车继续...")
    
    def start_new_shoe(self):
        """开始新靴牌"""
        self.visualizer.print_header("开始新靴牌")
        
        current = self.data_manager.get_current_shoe()
        
        if not current:
            self.visualizer.print_info("当前靴牌无数据，无需保存")
            return
        
        print(f"\n  当前靴牌有 {len(current)} 局数据")
        save = input("  是否保存当前靴牌到历史? (y/n): ").strip().lower()
        
        if save == 'y':
            name = input("  靴牌名称（留空自动生成）: ").strip()
            name = name if name else None
            
            if self.data_manager.save_current_shoe(name):
                self.visualizer.print_success("已保存当前靴牌到历史")
                self._clear_analysis_cache()  # 清空缓存
            else:
                self.visualizer.print_error("保存失败")
        
        self.data_manager.clear_current_shoe()
        self.visualizer.print_success("已开始新靴牌")
    
    def data_management(self):
        """数据管理"""
        self.visualizer.print_header("数据管理")
        
        options = [
            "导出当前靴牌数据",
            "从文件导入",
            "查看历史统计",
            "清空当前靴牌",
            "返回主菜单"
        ]
        
        self.visualizer.print_menu(options)
        
        choice = input("\n  请选择 (1-5): ").strip()
        
        if choice == '1':
            self.export_data()
        elif choice == '2':
            self.import_from_file()
        elif choice == '3':
            self.view_history_stats()
        elif choice == '4':
            self.clear_current()
        elif choice == '5':
            return
        else:
            self.visualizer.print_error("无效的选择")
    
    def export_data(self):
        """导出数据"""
        current = self.data_manager.get_current_shoe()
        
        if not current:
            self.visualizer.print_warning("当前靴牌无数据")
            return
        
        filename = input("\n  输入文件名 (如: data.txt): ").strip()
        
        if not filename:
            filename = "baccarat_export.txt"
        
        if self.data_manager.export_current_shoe(filename):
            self.visualizer.print_success(f"已导出到 {filename}")
        else:
            self.visualizer.print_error("导出失败")
    
    def import_from_file(self):
        """从文件导入"""
        filename = input("\n  输入文件路径: ").strip()
        
        if not filename:
            self.visualizer.print_warning("未输入文件路径")
            return
        
        success, count = self.data_manager.import_from_file(filename)
        
        if success:
            self.visualizer.print_success(f"成功导入 {count} 局数据")
        else:
            self.visualizer.print_error("导入失败，请检查文件路径和格式")
    
    def view_history_stats(self):
        """查看历史统计"""
        stats = self.data_manager.get_statistics()
        
        self.visualizer.print_section("📊 历史统计")
        self.visualizer.print_data_stats(stats)
        
        all_data = self.data_manager.get_all_data()
        if all_data:
            self.visualizer.print_statistics(all_data)
        
        input("\n  按回车继续...")
    
    def clear_current(self):
        """清空当前靴牌"""
        current = self.data_manager.get_current_shoe()
        
        if not current:
            self.visualizer.print_warning("当前靴牌已是空的")
            return
        
        confirm = input(f"\n  确定要清空当前 {len(current)} 局数据吗? (y/n): ").strip().lower()
        
        if confirm == 'y':
            self.data_manager.clear_current_shoe()
            self.visualizer.print_success("已清空当前靴牌")
        else:
            self.visualizer.print_info("已取消")
    
    def exit_system(self):
        """退出系统"""
        current = self.data_manager.get_current_shoe()
        
        if current and len(current) > 0:
            print("\n  当前靴牌有未保存的数据")
            save = input("  是否保存? (y/n): ").strip().lower()
            
            if save == 'y':
                self.data_manager.save_current_shoe()
                self.visualizer.print_success("已保存数据")
                self._clear_analysis_cache()  # 清空缓存
        
        print("\n  感谢使用百家乐智能预测系统！")
        print("  请理性娱乐，控制风险。\n")
        
        self.running = False
        sys.exit(0)


    def run_backtest(self):
        """运行历史回测（Walk-Forward 严格数据隔离）"""
        print("\n" + "="*60)
        print("  历史回测模式")
        print("  使用已存储的历史靴牌数据，模拟逐局预测")
        print("  注意：此过程可能需要数分钟，请耐心等待")
        print("="*60)
        
        try:
            from backtester import WalkForwardBacktester
            bt = WalkForwardBacktester()
            
            if len(bt.shoes) < 6:
                self.visualizer.print_error(
                    f"历史数据不足（当前 {len(bt.shoes)} 靴牌，至少需要 6 靴牌）\n"
                    "  请先积累更多历史数据后再运行回测。"
                )
                return
            
            bt.run(warmup_shoes=5, min_rounds_before_predict=10, verbose=True)
            
        except ImportError as e:
            self.visualizer.print_error(f"加载回测模块失败: {e}")
        except Exception as e:
            self.visualizer.print_error(f"回测出错: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    try:
        app = BaccaratPredictor()
        app.run()
    except KeyboardInterrupt:
        print("\n\n  程序已退出\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n  发生错误: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

