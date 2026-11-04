"""
RIPTIDE AUTO, paper trading the index timing strategy (part of Ocean Outlays)

Run once each trading day, best between about 1:45 and 2:45pm Chicago.
Before 2:50 it places "market on close" orders, which Alpaca fills by
itself at exactly the 3:00 close, so you can run it before class and walk
away. After 2:50 it falls back to buying or selling right away. It works out
whether Riptide should hold SPY and QQQ tomorrow, prints the calls, and
buys or sells in your Alpaca PAPER account to match. Then it shows how
Riptide's paper trades have done so far.

It reads its signals from SPY and uses the "always on 1x" setup that won
upgrade round 3.
    uptrend (SPY above its 200 day average)
        signal day   UPRO, 3x
        quiet day    SPY, 1x, so it doesn't miss steady bull markets
    downtrend
        signal day   SPY, 1x, the crash brake
        quiet day    cash
It only ever touches those two funds, and only up to RIPTIDE_DOLLARS, so it
won't disturb TIDE or anything else in the account. It needs no memory
of past runs to decide, so missing a day is fine, it just catches up.

PAPER ONLY, the address below is the paper account. Leave it that way.

New notebook or new cell. Add your Alpaca paper keys and run it daily.
"""

import contextlib
import io
import json
import os
import sys
import traceback
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# On the iPad, paste your keys here. On GitHub, leave these alone and add them as
# repository secrets instead, so they never appear in the code.
ALPACA_KEY = os.environ.get("ALPACA_KEY", "PASTE_PAPER_KEY_ID")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "PASTE_PAPER_SECRET")
TRADE_URL = "https://paper-api.alpaca.markets"   # paper account, do not change for now
RIPTIDE_DOLLARS = 1000      # pretend Riptide has this much, split evenly between the funds
SIGNAL = "SPY"              # signals are read from this
UPTREND_FUND = "UPRO"       # traded in an uptrend. UPRO is 3x, SSO is 2x
DOWNTREND_FUND = "SPY"      # traded in a downtrend, the crash switch brake
RSI_BUY = 10
LOG_FILE = os.environ.get("RIPTIDE_LOG", "riptide_auto_log.csv")

# phone alerts, optional, through the free ntfy app (ntfy.sh), no account needed.
# Install ntfy on your phone, tap +, and subscribe to the same topic name as below.
# Anyone who knows the topic name can read it, so make it long and random.
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "riptide-PASTE-A-LONG-RANDOM-NAME")
ALERTS_ON = bool(os.environ.get("NTFY_TOPIC")) or False   # on GitHub this turns on by itself
ALERT_EVERY_RUN = True         # True also alerts on quiet days with no trades

HOLIDAYS = set(pd.to_datetime("""2026-01-01 2026-01-19 2026-02-16 2026-04-03 2026-05-25 2026-06-19
2026-07-03 2026-09-07 2026-11-26 2026-12-25 2027-01-01 2027-01-18 2027-02-15 2027-03-26
2027-05-31 2027-06-18 2027-07-05 2027-09-06 2027-11-25 2027-12-24""".split()))


def http(method, url, body=None, alpaca=True):
    headers = {"User-Agent": "Mozilla/5.0"}
    if alpaca:
        headers.update({"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET,
                        "Content-Type": "application/json"})
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429:
                time.sleep(15)
                continue
            raise RuntimeError(f"{url[:50]} said {e.code} {e.read().decode()[:200]}")
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 4:
                raise
            time.sleep(8 * (attempt + 1))
    raise RuntimeError("kept being told to slow down, try again in a minute")


def prices(sym):
    """Daily closes from Yahoo. During market hours the last row is today's live price."""
    now = int(time.time())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?period1={now - 3 * 365 * 86400}&period2={now}&interval=1d")
    res = http("GET", url, alpaca=False)["chart"]["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s").normalize()
    c = pd.Series(res["indicators"]["quote"][0]["close"], index=idx).dropna()
    c = c[~c.index.duplicated(keep="last")]
    if len(c) < 300 or pd.Series(c.index).diff().dt.days.median() > 3:
        raise RuntimeError("Yahoo didn't send daily prices, try again in a minute")
    return c


TRADING_DAYS = None   # filled from Alpaca's market calendar, real holidays and early closes
CLOSE_TIMES = {}      # day -> closing time in New York minutes, 960 normally, 780 on early closes


def load_calendar(today):
    """Pulls the official market calendar. Falls back to the typed holiday list if it can't."""
    global TRADING_DAYS
    start = (today - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = (today + pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    try:
        cal = http("GET", f"{TRADE_URL}/v2/calendar?start={start}&end={end}") or []
        days = set()
        for d in cal:
            day = pd.Timestamp(d["date"])
            days.add(day)
            h, m = d["close"].split(":")[:2]
            CLOSE_TIMES[day] = int(h) * 60 + int(m)
        if days:
            TRADING_DAYS = days
    except Exception as e:
        print(f"Couldn't load the market calendar ({e}), using the built in holiday list.\n")


def is_trading_day(d):
    if TRADING_DAYS is not None and min(TRADING_DAYS) <= d <= max(TRADING_DAYS):
        return d in TRADING_DAYS
    return d.weekday() < 5 and d not in HOLIDAYS


def next_trading_day(d):
    d = d + pd.Timedelta(days=1)
    while not is_trading_day(d):
        d += pd.Timedelta(days=1)
    return d


def ny_to_chicago(minutes):
    h, m = divmod(minutes - 60, 60)
    return f"{(h - 1) % 12 + 1}:{m:02d}{'pm' if h >= 12 else 'am'}"


def dip_state(c):
    """Replays the dip bounce rule through today's price. True means hold tomorrow."""
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    rsi2 = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    sma200, sma5 = c.rolling(200).mean(), c.rolling(5).mean()
    inside = False
    for i in range(len(c)):
        if not inside and c.iloc[i] > sma200.iloc[i] and rsi2.iloc[i] < RSI_BUY:
            inside = True
        elif inside and c.iloc[i] > sma5.iloc[i]:
            inside = False
    return inside, rsi2.iloc[-1]


def calendar_reasons(today):
    tomorrow = next_trading_day(today)
    after = next_trading_day(tomorrow)
    reasons = []
    if after.month != tomorrow.month:
        reasons.append("turn of month, last day")
    first_days, d = [], tomorrow.replace(day=1) - pd.Timedelta(days=1)
    while len(first_days) < 3:
        d = next_trading_day(d)
        first_days.append(d)
    if tomorrow in first_days:
        reasons.append("turn of month, early days")
    nxt_weekday = tomorrow + pd.Timedelta(days=1)
    while nxt_weekday.weekday() >= 5:
        nxt_weekday += pd.Timedelta(days=1)
    if not is_trading_day(nxt_weekday):
        reasons.append("pre holiday")
    return tomorrow, reasons


def log(rows):
    pd.DataFrame(rows).to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)


def record():
    if not os.path.exists(LOG_FILE):
        return
    lg = pd.read_csv(LOG_FILE)
    trades = lg[lg.action.isin(["buy", "sell"])]
    results, open_buy = [], {}
    for _, r in trades.iterrows():
        if r.action == "buy":
            open_buy[r.sym] = r
        elif r.sym in open_buy:
            b = open_buy.pop(r.sym)
            results.append(r.price / b.price - 1)
    if results:
        res = pd.Series(results)
        print(f"\nRiptide paper record, {len(res)} finished trades, win {(res > 0).mean():.0%}, "
              f"average {res.mean():+.2%} per trade, total about {(1 + res).prod() - 1:+.1%}")
    if open_buy:
        print("Currently holding " + ", ".join(open_buy))


def main():
    clock = http("GET", f"{TRADE_URL}/v2/clock")
    now_ny = pd.Timestamp(clock["timestamp"][:19])
    today = now_ny.normalize()
    load_calendar(today)
    close_today = CLOSE_TIMES.get(today, 16 * 60)
    if os.environ.get("GITHUB_ACTIONS"):
        # GitHub runs Riptide several times a day so it never misses, only one acts
        now_min = now_ny.hour * 60 + now_ny.minute
        if clock.get("is_open") and close_today - now_min > 90:
            print("Too early in the day for this run, a later scheduled run will handle it.")
            return []
        if os.path.exists(LOG_FILE) and f"{today:%Y-%m-%d}" in open(LOG_FILE).read():
            print("Riptide already ran today, nothing to do.")
            return []
        if not clock.get("is_open"):
            return []
    upcoming = [d for d in sorted(CLOSE_TIMES) if today <= d <= today + pd.Timedelta(days=7)
                and CLOSE_TIMES[d] < 16 * 60]
    for d in upcoming:
        when = "TODAY" if d == today else f"{d:%a %b %d}"
        print(f"EARLY CLOSE {when}, the market closes at {ny_to_chicago(CLOSE_TIMES[d])} Chicago. "
              f"Run Riptide by {ny_to_chicago(CLOSE_TIMES[d] - 30)} that day.")
        if d == next_trading_day(today):
            send_alert("Riptide, early close tomorrow",
                       f"The market closes at {ny_to_chicago(CLOSE_TIMES[d])} Chicago on {d:%a %b %d}. "
                       f"Run Riptide by {ny_to_chicago(CLOSE_TIMES[d] - 30)}.")
    if upcoming:
        print()
    if not clock.get("is_open"):
        if is_trading_day(today) and now_ny.hour * 60 + now_ny.minute >= close_today:
            print(f"The market already closed today at {ny_to_chicago(close_today)} Chicago, so Riptide "
                  "couldn't trade. It will catch up on the next run.")
        else:
            print("The market is closed. Run this between about 1:45 and 2:45pm Chicago on a trading day.")
        record()
        return
    minutes = now_ny.hour * 60 + now_ny.minute            # New York time
    if close_today == 16 * 60 and minutes < 14 * 60 + 45:
        print("Heads up, it's early. Dip calls are most accurate after about 1:45pm Chicago.\n")
    on_close = minutes < close_today - 10                  # Alpaca takes close orders until 10 minutes before the close
    open_orders = http("GET", f"{TRADE_URL}/v2/orders?status=open") or []
    pending = {o["symbol"] for o in open_orders}

    tomorrow, reasons_cal = calendar_reasons(today)
    held = {p["symbol"]: p for p in (http("GET", f"{TRADE_URL}/v2/positions") or [])}
    print(f"RIPTIDE AUTO {today:%a %b %d}, calls for {tomorrow:%a %b %d}\n")

    c = prices(SIGNAL)
    dip, rsi = dip_state(c)
    sma200 = c.rolling(200).mean()
    uptrend = c.iloc[-1] > sma200.iloc[-1]
    was_up = c.iloc[-2] > sma200.iloc[-2]
    gap200 = c.iloc[-1] / sma200.iloc[-1] - 1
    if uptrend != was_up:
        if uptrend:
            msg = (f"SPY is back above its 200 day average ({gap200:+.1%}). The brake is off, "
                   "Riptide returns to 3x on signal days and 1x on quiet days.")
            send_alert("Riptide, uptrend is back", msg)
        else:
            msg = (f"SPY just fell below its 200 day average ({gap200:+.1%}). This is Riptide's crash "
                   "signal. The brake is on, so it drops to 1x on signal days and cash on quiet days. "
                   "Downtrends often mean a bigger selloff, but some are false alarms.")
            send_alert("Riptide CRASH SIGNAL, trend broke", msg)
        print(f"TREND CHANGE  {msg}\n")
    elif uptrend and gap200 < 0.02:
        print(f"Heads up, SPY is only {gap200:.1%} above its 200 day average, close to the crash signal.\n")
    reasons = list(reasons_cal) + (["dip bounce"] if dip else [])
    if uptrend:
        target = UPTREND_FUND if reasons else DOWNTREND_FUND
    else:
        target = DOWNTREND_FUND if reasons else None
    trend_txt = "uptrend" if uptrend else "downtrend, brake on"
    rsi_txt = f"{rsi:.0f}" if np.isfinite(rsi) else "high"

    if target and reasons:
        lev = {"UPRO": 3, "SSO": 2}.get(target, 1)
        print(f"{target:<4} UP     {lev}x through {tomorrow:%a}, typical day about {0.12 * lev:+.2f}%   "
              f"({', '.join(reasons)}; {trend_txt})")
    elif target:
        print(f"{target:<4} HOLD   1x base position through {tomorrow:%a}   "
              f"(quiet day in an {trend_txt}, 2 day RSI {rsi_txt})")
    else:
        print(f"SKIP        cash tomorrow   (quiet day in a {trend_txt}, 2 day RSI {rsi_txt})")

    rows = []
    for sym in dict.fromkeys([UPTREND_FUND, DOWNTREND_FUND]):
        want = sym == target
        have = sym in held and float(held[sym]["qty"]) > 0
        if want == have:
            if want:
                rows.append({"day": f"{today:%Y-%m-%d}", "sym": sym, "action": "hold",
                             "price": float(prices(sym).iloc[-1]), "why": "; ".join(reasons)})
            continue
        if sym in pending:
            print(f"       an order for {sym} is already waiting for the close, nothing more to do")
            continue
        price = float(prices(sym).iloc[-1])
        if want:
            shares = int(RIPTIDE_DOLLARS // price)
            if on_close and shares >= 1:
                http("POST", f"{TRADE_URL}/v2/orders", {"symbol": sym, "qty": str(shares), "side": "buy",
                                                        "type": "market", "time_in_force": "cls"})
                print(f"       will buy {shares} share{'s' if shares > 1 else ''} of {sym} at today's close")
            else:
                http("POST", f"{TRADE_URL}/v2/orders", {"symbol": sym, "notional": f"{RIPTIDE_DOLLARS:.2f}",
                                                        "side": "buy", "type": "market", "time_in_force": "day"})
                print(f"       bought about ${RIPTIDE_DOLLARS:.0f} of {sym} now")
            rows.append({"day": f"{today:%Y-%m-%d}", "sym": sym, "action": "buy", "price": price,
                         "why": "; ".join(reasons)})
        else:
            qty = float(held[sym]["qty"])
            if on_close and qty == int(qty):
                http("POST", f"{TRADE_URL}/v2/orders", {"symbol": sym, "qty": str(int(qty)), "side": "sell",
                                                        "type": "market", "time_in_force": "cls"})
                print(f"       will sell all {int(qty)} share{'s' if qty > 1 else ''} of {sym} at today's close")
            else:
                http("DELETE", f"{TRADE_URL}/v2/positions/{sym}")
                print(f"       sold all {sym} now")
            rows.append({"day": f"{today:%Y-%m-%d}", "sym": sym, "action": "sell", "price": price, "why": ""})
    if not rows:
        rows.append({"day": f"{today:%Y-%m-%d}", "sym": "", "action": "cash", "price": np.nan, "why": ""})
    log(rows)
    record()
    return rows


class Tee(io.StringIO):
    """Prints as normal and keeps a copy for the email."""
    def __init__(self, out):
        super().__init__()
        self.out = out

    def write(self, text):
        self.out.write(text)
        return super().write(text)


def send_alert(subject, body):
    if not ALERTS_ON:
        return
    if "PASTE" in NTFY_TOPIC:
        print("\nAlerts are on but not set up yet, put your own topic name in NTFY_TOPIC.")
        return
    req = urllib.request.Request(f"https://ntfy.sh/{NTFY_TOPIC}", data=body[-3500:].encode("utf-8"),
                                 method="POST", headers={"Title": subject, "Tags": "ocean"})
    try:
        with urllib.request.urlopen(req, timeout=30):
            pass
        print("\nPhone alert sent.")
    except Exception as e:
        print(f"\nCouldn't send the phone alert ({e}). The trades above still went through.")


def run():
    buf = Tee(sys.stdout)
    try:
        with contextlib.redirect_stdout(buf):
            rows = main() or []
    except Exception:
        send_alert("Riptide ERROR, check it", buf.getvalue() + "\n\n" + traceback.format_exc())
        raise
    trades = [r for r in rows if r.get("action") in ("buy", "sell")]
    if trades:
        what = ", ".join(f"{r['action'].upper()} {r['sym']}" for r in trades)
        send_alert(f"Riptide {what}", buf.getvalue())
    elif ALERT_EVERY_RUN and rows:
        send_alert("Riptide, no trades today", buf.getvalue())


run()
