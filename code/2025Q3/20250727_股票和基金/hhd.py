import akshare as ak
import pandas as pd
import os
import time
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_all_a_share_stocks():
    """获取所有A股股票列表"""
    try:
        logger.info("正在获取A股股票列表...")
        
        # 获取上海证券交易所A股股票列表
        sh_stocks = ak.stock_info_sh_name_code()
        sh_stocks['exchange'] = 'SH'
        
        # 获取深圳证券交易所A股股票列表
        sz_stocks = ak.stock_info_sz_name_code()
        sz_stocks['exchange'] = 'SZ'
        
        # 合并两个交易所的股票列表
        all_stocks = pd.concat([sh_stocks, sz_stocks], ignore_index=True)
        
        logger.info(f"成功获取 {len(all_stocks)} 只A股股票信息")
        return all_stocks
    
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        return None

def download_stock_data(stock_code, exchange, period="daily"):
    """下载单只股票的历史数据"""
    try:
        # 构造完整的股票代码（交易所代码 + 股票代码）
        full_code = f"{exchange}{stock_code}"
        
        logger.info(f"正在下载 {full_code} 的历史数据...")
        
        # 获取股票历史数据
        df = ak.stock_zh_a_hist(symbol=full_code, period=period, adjust="qfq")
        
        if df is not None and not df.empty:
            # 重命名列
            df = df.rename(columns={
                '日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low',
                '成交量': 'volume', '成交额': 'amount', '涨跌幅': 'pct_chg', '换手率': 'turnover_rate'
            })
            
            # 添加股票代码和交易所信息
            df['stock_code'] = stock_code
            df['exchange'] = exchange
            df['full_code'] = full_code
            
            logger.info(f"成功下载 {full_code} 的 {len(df)} 条历史数据")
            return df
        else:
            logger.warning(f"未找到 {full_code} 的历史数据")
            return None
            
    except Exception as e:
        logger.error(f"下载 {stock_code} 数据失败: {e}")
        return None

def save_stock_data(df, stock_code, exchange, output_dir="stock_data"):
    """保存股票数据到CSV文件"""
    try:
        # 创建输出目录（如果不存在）
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 按交易所创建子目录
        exchange_dir = os.path.join(output_dir, exchange)
        if not os.path.exists(exchange_dir):
            os.makedirs(exchange_dir)
        
        # 保存文件
        filename = f"{stock_code}.csv"
        filepath = os.path.join(exchange_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        logger.info(f"已保存 {stock_code} 数据到 {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"保存 {stock_code} 数据失败: {e}")
        return False

def download_all_stocks_data(output_dir="stock_data", max_stocks=None, delay=0.5):
    """下载所有A股股票数据"""
    logger.info("开始下载所有A股股票数据...")
    
    # 获取所有股票列表
    all_stocks = get_all_a_share_stocks()
    
    if all_stocks is None or all_stocks.empty:
        logger.error("无法获取股票列表，程序终止")
        return False
    
    # 限制下载数量（用于测试）
    if max_stocks is not None:
        all_stocks = all_stocks.head(max_stocks)
    
    total_stocks = len(all_stocks)
    successful_downloads = 0
    failed_downloads = 0
    
    logger.info(f"准备下载 {total_stocks} 只股票的数据")
    
    # 遍历所有股票并下载数据
    for index, row in all_stocks.iterrows():
        stock_code = row['code']
        exchange = row['exchange']
        stock_name = row['name']
        
        logger.info(f"处理进度: {index + 1}/{total_stocks} - {exchange}{stock_code} {stock_name}")
        
        # 下载股票数据
        stock_data = download_stock_data(stock_code, exchange)
        
        if stock_data is not None and not stock_data.empty:
            # 保存数据
            if save_stock_data(stock_data, stock_code, exchange, output_dir):
                successful_downloads += 1
            else:
                failed_downloads += 1
        else:
            failed_downloads += 1
        
        # 添加延迟，避免请求过于频繁
        time.sleep(delay)
    
    # 输出统计信息
    logger.info("=" * 50)
    logger.info("下载完成!")
    logger.info(f"成功下载: {successful_downloads} 只股票")
    logger.info(f"下载失败: {failed_downloads} 只股票")
    logger.info(f"成功率: {successful_downloads/total_stocks*100:.2f}%")
    logger.info("=" * 50)
    
    return successful_downloads > 0

def main():
    """主函数"""
    print("A股股票数据下载工具")
    print("=" * 50)
    
    # 设置参数
    output_directory = "stock_data"  # 数据保存目录
    max_download = None             # 最大下载数量（None表示下载全部，可用于测试）
    request_delay = 0.3             # 请求延迟（秒），避免过于频繁的请求
    
    # 开始下载
    success = download_all_stocks_data(
        output_dir=output_directory,
        max_stocks=max_download,
        delay=request_delay
    )
    
    if success:
        print(f"数据已保存到 {output_directory} 目录")
        print("按交易所分类:")
        print("- SH: 上海证券交易所")
        print("- SZ: 深圳证券交易所")
    else:
        print("下载过程中出现错误，请检查日志")

if __name__ == "__main__":
    main()