#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FXマルチ通貨ペア時間帯別トレード分析システム
このコード全体を1つのセルで実行してください
"""

import pandas as pd
import MetaTrader5 as mt5
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
import sys
warnings.filterwarnings('ignore')

# =============================================================================
# 設定パラメータ（ここを編集してカスタマイズ）
# =============================================================================

# 分析対象通貨ペア（メジャー通貨ペア＋クロス円＋主要クロスペア）
TARGET_SYMBOLS = [
    # メジャー通貨ペア
    'EURUSD',   # ユーロ/米ドル
    'USDJPY',   # 米ドル/日本円
    'GBPUSD',   # 英ポンド/米ドル
    'USDCHF',   # 米ドル/スイスフラン
    'AUDUSD',   # 豪ドル/米ドル
    'USDCAD',   # 米ドル/カナダドル
    'NZDUSD',   # NZドル/米ドル
    
    # クロス円
    'EURJPY',   # ユーロ/円
    'GBPJPY',   # ポンド/円
    'AUDJPY',   # 豪ドル/円
    'NZDJPY',   # NZドル/円
    'CADJPY',   # カナダドル/円
    'CHFJPY',   # スイスフラン/円
    
    # EUR クロス
    'EURGBP',   # ユーロ/ポンド
    'EURAUD',   # ユーロ/豪ドル
    'EURNZD',   # ユーロ/NZドル
    'EURCAD',   # ユーロ/カナダドル
    'EURCHF',   # ユーロ/スイスフラン
    
    # GBP クロス
    'GBPAUD',   # ポンド/豪ドル
    'GBPNZD',   # ポンド/NZドル
    'GBPCAD',   # ポンド/カナダドル
    'GBPCHF',   # ポンド/スイスフラン
    
    # AUD クロス
    'AUDNZD',   # 豪ドル/NZドル
    'AUDCAD',   # 豪ドル/カナダドル
    'AUDCHF',   # 豪ドル/スイスフラン
    
    # その他クロス
    'NZDCAD',   # NZドル/カナダドル
    'NZDCHF',   # NZドル/スイスフラン
    'CADCHF',   # カナダドル/スイスフラン
    
    # 貴金属
    'XAUUSD',   # 金/米ドル
]

# 英語表示モード（True: 英語, False: 日本語フォント自動検出）
FORCE_ENGLISH = True

# 分析パラメータ
DEFAULT_SPREAD_MULTIPLIER = 5.0  # 平均スプレッドの何倍まで許容するか
DATA_COUNT = 10000              # 取得するデータ数（5分足で約35日分）
MIN_TRADES = 20                 # 戦略として採用する最小トレード数
TIMEZONE_HOURS = 9              # タイムゾーン調整（日本時間 = GMT+9）

# 特定時間帯のテスト（Noneの場合は全時間帯を探索）
SPECIFIC_ENTRY_TIME = None      # 例: "17:50"
SPECIFIC_EXIT_TIME = None       # 例: "19:00"

# =============================================================================
# グローバル変数とフォント設定
# =============================================================================

use_japanese = False
WEEKDAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# 日本語フォント設定関数
def setup_japanese_font():
    """日本語フォントを設定"""
    global use_japanese, WEEKDAY_NAMES
    
    try:
        import matplotlib.font_manager as fm
        
        # Windows
        if 'win' in sys.platform:
            fonts = ['MS Gothic', 'MS Mincho', 'Yu Gothic', 'Yu Mincho', 'Meiryo']
            for font in fonts:
                if font in [f.name for f in fm.fontManager.ttflist]:
                    plt.rcParams['font.family'] = font
                    print(f"フォント設定: {font}")
                    use_japanese = True
                    WEEKDAY_NAMES = ['月', '火', '水', '木', '金', '土', '日']
                    return True
        
        # Mac
        elif 'darwin' in sys.platform:
            fonts = ['Hiragino Sans', 'Hiragino Mincho Pro', 'Osaka']
            for font in fonts:
                if font in [f.name for f in fm.fontManager.ttflist]:
                    plt.rcParams['font.family'] = font
                    print(f"フォント設定: {font}")
                    use_japanese = True
                    WEEKDAY_NAMES = ['月', '火', '水', '木', '金', '土', '日']
                    return True
        
        # Linux
        else:
            fonts = ['IPAGothic', 'IPAPGothic', 'TakaoGothic', 'Noto Sans CJK JP']
            for font in fonts:
                if font in [f.name for f in fm.fontManager.ttflist]:
                    plt.rcParams['font.family'] = font
                    print(f"フォント設定: {font}")
                    use_japanese = True
                    WEEKDAY_NAMES = ['月', '火', '水', '木', '金', '土', '日']
                    return True
        
        print("日本語フォントが見つかりません。英語表示にします。")
        return False
        
    except Exception as e:
        print(f"フォント設定エラー: {e}")
        return False

# フォント設定実行
if not FORCE_ENGLISH:
    setup_japanese_font()
else:
    print("英語表示モードが有効です")

plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# MT5接続と初期化
# =============================================================================

print("MT5に接続中...")
if not mt5.initialize():
    print("MT5への接続失敗。MT5が起動していることを確認してください。")
    quit()
print("MT5接続成功！")

# =============================================================================
# 関数定義
# =============================================================================

def detect_actual_symbols(target_symbols):
    """ブローカーで使用されている実際のシンボル名を検出"""
    actual_symbols = []
    all_symbols = mt5.symbols_get()
    
    if not all_symbols:
        print("利用可能なシンボルが取得できません")
        return []
    
    print("\nシンボル名を確認中...")
    
    for target in target_symbols:
        found = False
        
        # 完全一致を試す
        for symbol in all_symbols:
            if symbol.name.upper() == target.upper():
                actual_symbols.append(symbol.name)
                print(f"{target} → {symbol.name} ✓")
                found = True
                break
        
        if not found:
            # 部分一致を試す
            for symbol in all_symbols:
                if target.upper() in symbol.name.upper():
                    actual_symbols.append(symbol.name)
                    print(f"{target} → {symbol.name} ✓")
                    found = True
                    break
        
        if not found:
            print(f"{target} → 見つかりません ✗")
            # ゴールドの別名を試す
            if target == 'XAUUSD':
                for gold in ['GOLD', 'Gold', 'XAU']:
                    for symbol in all_symbols:
                        if gold.upper() in symbol.name.upper():
                            actual_symbols.append(symbol.name)
                            print(f"{target} → {symbol.name} ✓ (GOLD)")
                            found = True
                            break
                    if found:
                        break
    
    return actual_symbols

def fetch_data(symbol, count=DATA_COUNT):
    """データ取得とpip計算"""
    print(f"\n{symbol}のデータ取得中...")
    
    # シンボル情報取得
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"エラー: {symbol}の情報が取得できません")
        return None
    
    # シンボルを有効化
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"エラー: {symbol}を有効化できません")
            return None
    
    # データ取得
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, count)
    if rates is None:
        print(f"エラー: {symbol}のデータ取得失敗")
        return None
    
    # DataFrame作成
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s') + timedelta(hours=TIMEZONE_HOURS)
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['weekday'] = df['time'].dt.weekday
    df['date'] = df['time'].dt.date
    
    # pip値計算
    digits = symbol_info.digits
    if digits >= 3:
        pip_value = 10 ** (-(digits - 1))
    else:
        pip_value = 10 ** (-digits)
    
    # スプレッド変換
    if 'spread' not in df.columns:
        df['spread'] = 20  # デフォルト2pips
    
    if digits == 5 or digits == 3:
        df['spread_pips'] = df['spread'] / 10
    else:
        df['spread_pips'] = df['spread']
    
    # 属性保存
    df.attrs['symbol'] = symbol
    df.attrs['pip_value'] = pip_value
    df.attrs['digits'] = digits
    df.attrs['spread_mean'] = df['spread_pips'].mean()
    
    print(f"取得完了: {len(df)}件、期間: {df['time'].min()} ～ {df['time'].max()}")
    print(f"平均スプレッド: {df['spread_pips'].mean():.1f} pips")
    
    return df

def backtest_time_range(df, entry_h, entry_m, exit_h, exit_m, direction='long'):
    """特定時間帯のバックテスト"""
    # スプレッドフィルタ
    max_spread = df.attrs['spread_mean'] * DEFAULT_SPREAD_MULTIPLIER
    
    # エントリー・決済データ抽出
    entry_mask = (df['hour'] == entry_h) & (df['minute'] == entry_m) & (df['spread_pips'] <= max_spread)
    exit_mask = (df['hour'] == exit_h) & (df['minute'] == exit_m)
    
    entries = df[entry_mask][['date', 'open', 'spread_pips', 'weekday']].copy()
    exits = df[exit_mask][['date', 'close', 'spread_pips']].copy()
    
    # 同じ日のトレードをマージ
    trades = pd.merge(entries, exits, on='date', suffixes=('_entry', '_exit'))
    
    if len(trades) == 0:
        return None
    
    # 損益計算
    pip_value = df.attrs['pip_value']
    
    if direction == 'long':
        # ロング: Ask買い → Bid売り
        trades['entry_price'] = trades['open'] + trades['spread_pips_entry'] * pip_value
        trades['exit_price'] = trades['close']
        trades['profit'] = (trades['exit_price'] - trades['entry_price']) / trades['entry_price']
    else:
        # ショート: Bid売り → Ask買い
        trades['entry_price'] = trades['open']
        trades['exit_price'] = trades['close'] + trades['spread_pips_exit'] * pip_value
        trades['profit'] = (trades['entry_price'] - trades['exit_price']) / trades['entry_price']
    
    # 統計計算
    if len(trades) > 0:
        return {
            'count': len(trades),
            'win_rate': (trades['profit'] > 0).mean(),
            'avg_return': trades['profit'].mean(),
            'total_return': trades['profit'].sum(),
            'sharpe': trades['profit'].mean() / trades['profit'].std() if trades['profit'].std() > 0 else 0,
            'trades': trades
        }
    
    return None

def find_best_times(df, symbol):
    """最適な時間帯を探索"""
    print(f"\n{symbol}: 最適時間帯を探索中...")
    
    results = []
    total_tests = 0
    
    # 時間帯の組み合わせを生成
    for entry_h in range(0, 24):
        for entry_m in [0, 30]:  # 30分単位
            for hold_h in [1, 2, 3]:  # 1-3時間保有
                exit_h = (entry_h + hold_h) % 24
                exit_m = entry_m
                
                total_tests += 2  # ロングとショート
                
                # 進捗表示
                if total_tests % 100 == 0:
                    print(f"  テスト中... {total_tests}個の戦略を検証")
                
                # ロングテスト
                long_result = backtest_time_range(df, entry_h, entry_m, exit_h, exit_m, 'long')
                if long_result and long_result['count'] >= MIN_TRADES:
                    results.append({
                        'symbol': symbol,
                        'direction': 'long',
                        'entry': f"{entry_h:02d}:{entry_m:02d}",
                        'exit': f"{exit_h:02d}:{exit_m:02d}",
                        'count': long_result['count'],
                        'win_rate': long_result['win_rate'],
                        'avg_return': long_result['avg_return'],
                        'sharpe': long_result['sharpe']
                    })
                
                # ショートテスト
                short_result = backtest_time_range(df, entry_h, entry_m, exit_h, exit_m, 'short')
                if short_result and short_result['count'] >= MIN_TRADES:
                    results.append({
                        'symbol': symbol,
                        'direction': 'short',
                        'entry': f"{entry_h:02d}:{entry_m:02d}",
                        'exit': f"{exit_h:02d}:{exit_m:02d}",
                        'count': short_result['count'],
                        'win_rate': short_result['win_rate'],
                        'avg_return': short_result['avg_return'],
                        'sharpe': short_result['sharpe']
                    })
    
    print(f"  完了: {total_tests}個の戦略をテスト")
    
    # DataFrameに変換
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        # 利益が出る戦略のみ
        profitable = results_df[results_df['avg_return'] > 0].sort_values('sharpe', ascending=False)
        return profitable
    
    return pd.DataFrame()

def detailed_backtest(df, entry_h, entry_m, exit_h, exit_m, direction):
    """詳細なバックテストと曜日別分析"""
    result = backtest_time_range(df, entry_h, entry_m, exit_h, exit_m, direction)
    
    if not result:
        return None, None
    
    trades = result['trades']
    
    # 累積リターン計算
    trades['cumulative'] = (1 + trades['profit']).cumprod() - 1
    
    # 曜日別統計
    weekday_stats = trades.groupby('weekday').agg({
        'profit': ['mean', 'count', lambda x: (x > 0).mean()]
    })
    weekday_stats.columns = ['avg_return', 'count', 'win_rate']
    
    return trades, weekday_stats

def plot_results(trades, weekday_stats, symbol, entry_time, exit_time, direction):
    """結果の可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 累積リターン
    ax = axes[0, 0]
    ax.plot(trades['cumulative'] * 100, linewidth=2)
    ax.set_title(f'{symbol} {direction.upper()}: {entry_time} → {exit_time}')
    ax.set_ylabel('Cumulative Return %' if not use_japanese else '累積リターン %')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--')
    
    # リターン分布
    ax = axes[0, 1]
    profits = trades['profit'] * 100
    ax.hist(profits, bins=30, alpha=0.7)
    ax.axvline(x=profits.mean(), color='r', linestyle='-', linewidth=2)
    ax.set_title('Return Distribution' if not use_japanese else 'リターン分布')
    ax.set_xlabel('Return %' if not use_japanese else 'リターン %')
    
    # 曜日別パフォーマンス
    ax = axes[1, 0]
    weekday_returns = weekday_stats['avg_return'] * 100
    weekday_names = [WEEKDAY_NAMES[i] for i in weekday_returns.index]
    colors = ['green' if x > 0 else 'red' for x in weekday_returns.values]
    ax.bar(weekday_names, weekday_returns.values, color=colors, alpha=0.7)
    ax.set_title('Average Return by Weekday' if not use_japanese else '曜日別平均リターン')
    ax.set_ylabel('Average Return %' if not use_japanese else '平均リターン %')
    ax.axhline(y=0, color='black', linestyle='-')
    
    # サマリー
    ax = axes[1, 1]
    ax.axis('off')
    
    win_rate = (trades['profit'] > 0).mean()
    avg_return = trades['profit'].mean()
    total_return = trades['cumulative'].iloc[-1]
    sharpe = trades['profit'].mean() / trades['profit'].std() if trades['profit'].std() > 0 else 0
    
    summary_text = f"""
=== Performance Summary ===

Symbol: {symbol}
Direction: {direction.upper()}
Entry: {entry_time}
Exit: {exit_time}

Trades: {len(trades)}
Win Rate: {win_rate:.1%}
Avg Return: {avg_return:.4%}
Total Return: {total_return:.2%}
Sharpe Ratio: {sharpe:.3f}
"""
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# メイン処理
# =============================================================================

def main():
    """メイン分析処理"""
    # 指定された通貨ペアの実際のシンボル名を検出
    symbols = detect_actual_symbols(TARGET_SYMBOLS)
    
    if not symbols:
        print("\n指定された通貨ペアが見つかりません")
        print("ブローカーのシンボル名を確認してください")
        mt5.shutdown()
        return
    
    print(f"\n分析対象通貨ペア: {symbols}")
    
    # 特定時間帯のテスト
    if SPECIFIC_ENTRY_TIME and SPECIFIC_EXIT_TIME:
        print(f"\n特定時間帯のテスト: {SPECIFIC_ENTRY_TIME} → {SPECIFIC_EXIT_TIME}")
        entry_h, entry_m = map(int, SPECIFIC_ENTRY_TIME.split(':'))
        exit_h, exit_m = map(int, SPECIFIC_EXIT_TIME.split(':'))
        
        for symbol in symbols:
            df = fetch_data(symbol)
            if df is None:
                continue
            
            print(f"\n{symbol}:")
            
            # ロングテスト
            long_result = backtest_time_range(df, entry_h, entry_m, exit_h, exit_m, 'long')
            if long_result:
                print(f"  Long: Trades={long_result['count']}, WinRate={long_result['win_rate']:.1%}, AvgReturn={long_result['avg_return']:.4%}")
            
            # ショートテスト
            short_result = backtest_time_range(df, entry_h, entry_m, exit_h, exit_m, 'short')
            if short_result:
                print(f"  Short: Trades={short_result['count']}, WinRate={short_result['win_rate']:.1%}, AvgReturn={short_result['avg_return']:.4%}")
        
        mt5.shutdown()
        return
    
    # 全時間帯の探索
    all_results = []
    
    # 各通貨ペアを分析
    for symbol in symbols:
        # データ取得
        df = fetch_data(symbol)
        if df is None:
            continue
        
        # 最適時間帯探索
        profitable = find_best_times(df, symbol)
        
        if len(profitable) > 0:
            print(f"\n{symbol}: {len(profitable)}個の利益戦略を発見")
            all_results.append({
                'symbol': symbol,
                'data': df,
                'strategies': profitable
            })
        else:
            print(f"\n{symbol}: 利益が出る戦略が見つかりませんでした")
    
    # 全体の最良戦略を表示
    if all_results:
        print("\n" + "="*60)
        print("全通貨ペア 最良戦略 TOP10")
        print("="*60)
        
        # 全戦略を統合
        all_strategies = []
        for result in all_results:
            all_strategies.extend(result['strategies'].to_dict('records'))
        
        # シャープレシオでソート
        all_strategies_df = pd.DataFrame(all_strategies)
        top_strategies = all_strategies_df.sort_values('sharpe', ascending=False).head(10)
        
        print("\n")
        # 表形式で見やすく表示
        display_df = top_strategies[['symbol', 'direction', 'entry', 'exit', 'count', 'win_rate', 'avg_return', 'sharpe']].copy()
        display_df['win_rate'] = (display_df['win_rate'] * 100).round(1)
        display_df['avg_return'] = (display_df['avg_return'] * 100).round(4)
        display_df['sharpe'] = display_df['sharpe'].round(3)
        
        display_df.columns = ['Symbol', 'Direction', 'Entry', 'Exit', 'Trades', 'Win%', 'AvgReturn%', 'Sharpe']
        
        print(display_df.to_string(index=False))
        
        # TOP3の詳細分析
        print("\n" + "="*60)
        print("TOP3戦略の詳細分析")
        print("="*60)
        
        for i in range(min(3, len(top_strategies))):
            strategy = top_strategies.iloc[i]
            
            # 該当するデータを探す
            df = None
            for result in all_results:
                if result['symbol'] == strategy['symbol']:
                    df = result['data']
                    break
            
            if df is None:
                continue
            
            # 詳細バックテスト
            entry_h, entry_m = map(int, strategy['entry'].split(':'))
            exit_h, exit_m = map(int, strategy['exit'].split(':'))
            
            print(f"\n【{i+1}位】 {strategy['symbol']} {strategy['direction'].upper()}: {strategy['entry']} → {strategy['exit']}")
            print(f"Sharpe Ratio: {strategy['sharpe']:.3f}, Avg Return: {strategy['avg_return']:.4%}")
            
            trades, weekday_stats = detailed_backtest(df, entry_h, entry_m, exit_h, exit_m, strategy['direction'])
            
            if trades is not None:
                plot_results(trades, weekday_stats, strategy['symbol'], strategy['entry'], strategy['exit'], strategy['direction'])
    
    else:
        print("\n利益が出る戦略が見つかりませんでした")
        print("以下を確認してください:")
        print("1. 十分なヒストリカルデータがあるか")
        print(f"2. スプレッドフィルタが厳しすぎないか（現在: 平均の{DEFAULT_SPREAD_MULTIPLIER}倍）")
        print(f"3. 最小トレード数の条件（{MIN_TRADES}回）が厳しすぎないか")
    
    # MT5終了
    mt5.shutdown()
    print("\n分析完了！")

# =============================================================================
# 実行
# =============================================================================

if __name__ == "__main__":
    main()
