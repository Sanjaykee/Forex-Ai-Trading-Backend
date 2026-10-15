import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from app.backtest.engine import run_backtest

pairs = ['EURUSD','GBPUSD','USDJPY','USDCHF','AUDUSD','NZDUSD','USDCAD','EURJPY','GBPJPY']
days  = 15
rr    = 3.0

grand_sig = grand_w = grand_l = 0
grand_pips = 0.0

for tf in ['M15', 'M3']:
    print(f'\n========== {tf} | Last {days} days | RR 1:{rr} ==========')
    tf_sig = tf_w = tf_l = 0
    tf_pips = 0.0
    for p in pairs:
        try:
            r = run_backtest(symbol=p, timeframe=tf, days=days, rr_ratio=rr)
            sig = r.get('total_signals', 0)
            w   = r.get('wins', 0)
            l   = r.get('losses', 0)
            wr  = r.get('win_rate', 0.0)
            np_ = r.get('net_pips', 0.0)
            fc  = r.get('filter_counts', {})
            pts = r.get('patterns', {})
            if sig == 0:
                dom = max(fc, key=fc.get) if fc else 'none'
                print(f'  {p:<8} 0 signals  block={dom}  fc={fc}')
            else:
                pt_str = ' | '.join(f"{k}:{v['wins']}/{v['total']}" for k,v in pts.items())
                print(f'  {p:<8} sig={sig:>2}  w={w}  l={l}  wr={wr:>5.1f}%  pips={np_:>+7.1f}  [{pt_str}]')
                tf_sig  += sig; tf_w += w; tf_l += l; tf_pips += np_
        except Exception as e:
            print(f'  {p:<8} ERROR: {e}')
    tf_wr = round(tf_w / tf_sig * 100, 1) if tf_sig else 0
    print(f'  {"SUBTOTAL":<8} sig={tf_sig:>2}  w={tf_w}  l={tf_l}  wr={tf_wr:>5.1f}%  pips={tf_pips:>+7.1f}')
    grand_sig += tf_sig; grand_w += tf_w; grand_l += tf_l; grand_pips += tf_pips

grand_wr = round(grand_w / grand_sig * 100, 1) if grand_sig else 0
print(f'\n========== GRAND TOTAL ==========')
print(f'  Signals={grand_sig}  Wins={grand_w}  Losses={grand_l}  WR={grand_wr}%  NetPips={grand_pips:+.1f}')
