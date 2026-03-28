"""
工具函数模块
"""

def normalize_input(text):
    """
    标准化输入数据，支持多种格式
    
    Args:
        text: 输入的文本数据
        
    Returns:
        标准化后的列表 ['B', 'P', 'T', ...]
    """
    # 移除空白字符
    text = text.strip().replace(' ', '').replace('\n', '').replace('\r', '')
    
    # 移除逗号
    text = text.replace(',', '')
    
    # 中文转换
    text = text.replace('庄', 'B').replace('闲', 'P').replace('和', 'T')
    text = text.replace('banker', 'B').replace('player', 'P').replace('tie', 'T')
    text = text.replace('BANKER', 'B').replace('PLAYER', 'P').replace('TIE', 'T')
    
    # 转换为大写
    text = text.upper()
    
    # 转换为列表
    result = []
    for char in text:
        if char in ['B', 'P', 'T']:
            result.append(char)
    
    return result


def result_to_chinese(result):
    """将结果转换为中文"""
    mapping = {'B': '庄', 'P': '闲', 'T': '和'}
    return mapping.get(result, result)


def result_to_number(result):
    """将结果转换为数字（用于ML模型）"""
    mapping = {'B': 1, 'P': 0, 'T': 2}
    return mapping.get(result, -1)


def number_to_result(number):
    """将数字转换为结果"""
    mapping = {1: 'B', 0: 'P', 2: 'T'}
    return mapping.get(number, 'P')


def calculate_statistics(data):
    """
    计算基础统计信息
    
    Args:
        data: 结果列表 ['B', 'P', 'T', ...]
        
    Returns:
        统计字典
    """
    if not data:
        return {
            'total': 0,
            'banker_count': 0,
            'player_count': 0,
            'tie_count': 0,
            'banker_rate': 0,
            'player_rate': 0,
            'tie_rate': 0
        }
    
    total = len(data)
    banker_count = data.count('B')
    player_count = data.count('P')
    tie_count = data.count('T')
    
    return {
        'total': total,
        'banker_count': banker_count,
        'player_count': player_count,
        'tie_count': tie_count,
        'banker_rate': banker_count / total * 100 if total > 0 else 0,
        'player_rate': player_count / total * 100 if total > 0 else 0,
        'tie_rate': tie_count / total * 100 if total > 0 else 0
    }


def get_current_streak(data):
    """
    获取当前连胜情况
    
    Args:
        data: 结果列表
        
    Returns:
        (连胜类型, 连胜次数)
    """
    if not data or len(data) == 0:
        return None, 0
    
    current = data[-1]
    count = 1
    
    for i in range(len(data) - 2, -1, -1):
        if data[i] == current:
            count += 1
        else:
            break
    
    return current, count


def get_max_streak(data, result_type):
    """
    获取最长连胜记录
    
    Args:
        data: 结果列表
        result_type: 'B', 'P', 或 'T'
        
    Returns:
        最长连胜次数
    """
    if not data:
        return 0
    
    max_streak = 0
    current_streak = 0
    
    for result in data:
        if result == result_type:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    return max_streak


def format_percentage_bar(percentage, width=20):
    """
    格式化百分比进度条
    
    Args:
        percentage: 百分比值 (0-100)
        width: 进度条宽度
        
    Returns:
        进度条字符串
    """
    filled = int(percentage / 100 * width)
    bar = '█' * filled + '░' * (width - filled)
    return f"{bar} {percentage:.1f}%"


def validate_data(data):
    """
    验证数据有效性
    
    Args:
        data: 结果列表
        
    Returns:
        (是否有效, 错误信息)
    """
    if not data:
        return False, "数据为空"
    
    if len(data) < 1:
        return False, "数据量不足"
    
    for item in data:
        if item not in ['B', 'P', 'T']:
            return False, f"无效的数据项: {item}"
    
    return True, "数据有效"

