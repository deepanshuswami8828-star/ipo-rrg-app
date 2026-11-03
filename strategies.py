"""
strategies.py — RRG Trading Strategy Engine.
Strategy 1: Daily + Weekly RRG Confirmation.

BUY RULE:  (Daily in {Leading, Improving}) AND (Weekly in {Leading, Improving})
SELL RULE: (Daily in {Lagging, Weakening}) OR  (Weekly in {Lagging, Weakening})
Otherwise an open position is held; never a second buy while open.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

import config as cfg
from make_rrg_pvrinox import (
    calc_rs_ratio, calc_rs_momentum, quadrant,
    RS_PRESMOOTH, RS_SHORT, RS_LONG, MOM_SHORT, MOM_LONG
)

UP = {"Improving", "Leading"}
DOWN = {"Weakening", "Lagging"}
# same warm-up as the charts (config.py), so strategies use exactly the readings the app shows
WARMUP_DAILY = cfg.WARMUP_DAILY
WARMUP_WEEKLY = cfg.WARMUP_WEEKLY

# Strategy Registry Metadata
STRATEGY_REGISTRY = {
    1: {
        "id": 1,
        "label": "Strategy 1",
        "weekly_only": False,
        "name": "Daily + Weekly RRG Confirmation",
        "tag": "Multi-Timeframe",
        "short_desc": "Buy only when BOTH Daily & Weekly are Leading/Improving; Sell when EITHER drops into Weakening/Lagging.",
        "full_desc": "Strict dual-timeframe confirmation strategy. Buys only when BOTH Daily and Weekly RRG are in Leading or Improving. Exits immediately if EITHER timeframe falls into Weakening or Lagging. Held otherwise. Every buy, and every sell triggered by the RRG, happens on the next trading day after the signal, at the average of that day's open and close.",
        "entry_rule": "Buy when Daily ∈ {Leading, Improving} AND Weekly ∈ {Leading, Improving} - bought on the next trading day at the average of its open and close.",
        "exit_rule": "Sell when Daily ∈ {Lagging, Weakening} OR Weekly ∈ {Lagging, Weakening} - sold on the next trading day at the average of its open and close.",
        "best_market": "Sustained intermediate-term bull trends with confirmed weekly institutional participation.",
        "risk_watch": "Susceptible to quick exits if either timeframe stumbles; capital preservation prioritized.",
    },
    2: {
        "id": 2,
        "label": "Strategy 2 · Version 1",
        "weekly_only": True,
        "name": "Weekly RRG Only (Version 1)",
        "tag": "Weekly Only",
        "short_desc": "Buy when the Weekly RRG is Leading/Improving; Sell when it drops into Weakening/Lagging. Daily is ignored.",
        "full_desc": "Single-timeframe weekly strategy. Every trading day it reads the Weekly RRG as it stands at that day's close (week so far): buys on the first day the Weekly is in Leading or Improving and sells on the first day it is in Weakening or Lagging. The Daily RRG is never used. Buys and sells on the next trading day at the average of its open and close.",
        "entry_rule": "Buy on the first day the Weekly (as of that day) is in {Leading, Improving} - bought on the next trading day at the average of its open and close.",
        "exit_rule": "Sell on the first day the Weekly (as of that day) is in {Weakening, Lagging} - sold on the next trading day at the average of its open and close.",
        "best_market": "Long, steady trends where a stock outperforms NIFTY 50 for months at a time.",
        "risk_watch": "The weekly reading of an unfinished week can change before the week ends, so a signal can flip back; there is no stop-loss.",
    },
    # Strategy 2, version 2 (internal key 3; shown as "Strategy 2 · Version 2")
    3: {
        "id": 3,
        "label": "Strategy 2 · Version 2",
        "weekly_only": True,
        "name": "Weekly RRG Only (Version 2) — Sell when RS-Mom falls below 101",
        "tag": "Weekly Only",
        "short_desc": "Buy when Weekly enters Leading/Improving; Sell when Weekly RS-Momentum falls back below 101 after being Leading at 101+, or Weekly drops into Weakening/Lagging. Daily is ignored.",
        "full_desc": "Weekly-only strategy with an early exit. Reads the Weekly RRG every trading day (week so far). Buys when the Weekly RRG is in Leading or Improving. Once the trade has had a week in Leading with RS-Momentum at 101 or more, it sells on the first day the Weekly RS-Momentum falls below 101 - whether the stock is still Leading or has slipped into Improving - instead of waiting for Weakening. It also sells if the Weekly lands in Weakening or Lagging. After an early exit it buys again only after the Weekly has dropped into Weakening/Lagging and come back up. The Daily RRG is never used. Every buy, and every sell triggered by the RRG, happens on the next trading day after the signal, at the average of that day's open and close.",
        "entry_rule": "Buy on the first day the Weekly (as of that day) is in {Leading, Improving} (after an early exit: only once the Weekly has been in Weakening/Lagging and comes back up) - bought on the next trading day at the average of its open and close.",
        "exit_rule": "Sell on the first day the Weekly RS-Momentum falls below 101 after the trade has been Leading at 101+, or the Weekly is in {Weakening, Lagging} - sold on the next trading day at the average of its open and close.",
        "best_market": "Trends that roll over gradually, where momentum fades before the stock turns down.",
        "risk_watch": "Can exit a strong trend on a brief momentum dip and must then wait for a full down-and-up cycle to re-enter.",
    },
    # Strategy 2, version 3 (internal key 4; shown as "Strategy 2 · Version 3")
    4: {
        "id": 4,
        "label": "Strategy 2 · Version 3",
        "weekly_only": True,
        "name": "Weekly RRG Only (Version 3) — 10% Stop-Loss",
        "tag": "Weekly Only",
        "short_desc": "Buy when Weekly is Leading/Improving; Sell when Weekly drops into Weakening/Lagging, or on ANY day's close 10% or more below your buy price (stop-loss). Daily RRG is ignored.",
        "full_desc": "Version 1 with a protective stop-loss. Buys and sells on the Weekly RRG only, read every trading day (week so far): buys on the first day the Weekly is in Leading or Improving and sells on the first day it is in Weakening or Lagging. In addition, the price is checked at every day's close: the first close at or below 90% of the buy price sells the position that day and books the loss. After a stop-loss sale it buys again only after the Weekly has dropped into Weakening/Lagging and come back up. The Daily RRG is never used - the daily check looks only at the price. Every buy, and every sell triggered by the RRG, happens on the next trading day after the signal, at the average of that day's open and close.",
        "entry_rule": "Buy on the first day the Weekly (as of that day) is in {Leading, Improving} (after a stop-loss: only once the Weekly has been in Weakening/Lagging and comes back up) - bought on the next trading day at the average of its open and close.",
        "exit_rule": "Sell on the first day the Weekly (as of that day) is in {Weakening, Lagging} - sold on the next trading day at the average of its open and close; or at ANY day's close that is ≤ 90% of the buy price (10% stop-loss, sold at that close).",
        "best_market": "Trending stocks where the weekly RRG works well but you want a hard limit on any single loss.",
        "risk_watch": "A gap-down below the stop on one day books a loss a little bigger than 10%; a stopped-out stock that recovers is only re-bought after a weekly down-and-up cycle.",
    },
}


# ---- RRG & Weekly As-Of Calculations -----------------------------------------
def compute_rrg(df_aligned: pd.DataFrame, stock_name: str, tf: str = "Daily") -> Tuple[pd.DataFrame, bool]:
    """Computes RRG daily or weekly series from aligned closes."""
    bench_name = cfg.BENCHMARK_NAME
    closes = df_aligned.copy()
    if tf == "Weekly":
        closes = closes.groupby(closes.index.to_period("W-FRI")).tail(1)

    raw_len = len(closes)
    warmup_val = WARMUP_WEEKLY if tf == "Weekly" else WARMUP_DAILY
    warmup = min(warmup_val, raw_len // 3)
    if raw_len < warmup + 10:
        return pd.DataFrame(), False

    rr = calc_rs_ratio(closes[stock_name], closes[bench_name])
    rm = calc_rs_momentum(rr)
    df = pd.DataFrame({"RS_Ratio": rr, "RS_Momentum": rm, "Close": closes[stock_name]}).dropna()
    df = df.iloc[warmup:]
    df["Quadrant"] = [quadrant(a, b) for a, b in zip(df["RS_Ratio"], df["RS_Momentum"])]

    is_sufficient = len(df) >= 60 if tf == "Weekly" else len(df) >= 30
    return df, is_sufficient


def weekly_asof(closes: pd.DataFrame, stock_name: str, bench_name: str = cfg.BENCHMARK_NAME) -> pd.DataFrame:
    """
    Weekly RS-Ratio / RS-Momentum as they stood on each trading day (no look-ahead).
    """
    week = closes.index.to_period("W-FRI")
    wk = closes.groupby(week).tail(1)
    if len(wk) < 15:
        return pd.DataFrame()

    rs_w = pd.Series((wk[stock_name] / wk[bench_name]).to_numpy())
    sm = rs_w.rolling(RS_PRESMOOTH, min_periods=1).mean()
    e_fast = sm.ewm(span=RS_SHORT, adjust=False).mean().to_numpy()
    e_slow = sm.ewm(span=RS_LONG, adjust=False).mean().to_numpy()
    rr_w = pd.Series(100.0 * e_fast / e_slow)
    m_fast = rr_w.ewm(span=MOM_SHORT, adjust=False).mean().to_numpy()
    m_slow = rr_w.ewm(span=MOM_LONG, adjust=False).mean().to_numpy()
    prior = rs_w.rolling(RS_PRESMOOTH - 1, min_periods=1).sum().to_numpy()

    pos = pd.Index(wk.index.to_period("W-FRI")).get_indexer(week)
    prev = pos - 1
    warmup = min(WARMUP_WEEKLY, len(wk) // 3)
    ok = prev >= warmup
    if not ok.any():
        return pd.DataFrame()

    prev_ok = prev[ok]
    d_idx = closes.index[ok]
    rs_d = (closes[stock_name] / closes[bench_name]).to_numpy()[ok]

    def step(x, state, span):
        a = 2.0 / (span + 1.0)
        return a * x + (1 - a) * state

    sm_d = (prior[prev_ok] + rs_d) / RS_PRESMOOTH
    rr_d = 100.0 * step(sm_d, e_fast[prev_ok], RS_SHORT) / step(sm_d, e_slow[prev_ok], RS_LONG)
    rm_d = 100.0 * step(rr_d, m_fast[prev_ok], MOM_SHORT) / step(rr_d, m_slow[prev_ok], MOM_LONG)
    out = pd.DataFrame({"RS_Ratio": rr_d, "RS_Momentum": rm_d, "Close": closes[stock_name].loc[d_idx]}, index=d_idx)
    out["Quadrant"] = [quadrant(r, m) for r, m in zip(rr_d, rm_d)]
    return out


# ---- Entry rule (all strategies) ----------------------------------------------
# A buy signal appears at a day's close. "next_avg" (default, realistic): buy on the NEXT trading day
# at the average of that day's open and close; exit checks start the day after the buy.
# "close": the original rule, buy at the signal day's own close.
ENTRY_NEXT_AVG, ENTRY_CLOSE = "next_avg", "close"
DEFAULT_ENTRY = ENTRY_NEXT_AVG
ENTRY_TEXT = "Buys on the next trading day after the signal, at the average of that day's open and close."

# Exits triggered by the RRG (weekly quadrant / momentum, and Strategy 1's daily quadrant) are also known only
# at a day's close: "next_avg" (default) sells on the NEXT trading day at the average of its open and close.
# Price exits (stop-loss at the close, take-profit at its target) still fill the same day. "close": original rule.
DEFAULT_RRG_EXIT = ENTRY_NEXT_AVG
RRG_EXIT_TEXT = ("A weekly RRG exit sells on the next trading day after the signal, at the average of that day's "
                 "open and close; stop-loss and take-profit exits fill the same day.")


def _fill_px(open_px, close_px) -> float:
    """Next-day entry price: average of the day's open and close (close alone if the open is missing)."""
    return (open_px + close_px) / 2.0 if open_px is not None and open_px == open_px else close_px


def _open_of(opens: Optional[pd.Series], d) -> Optional[float]:
    return opens.get(d, np.nan) if opens is not None else np.nan


def _sell_next_day(pos: Dict, pending_exit: Dict, d, open_px, close_px) -> Dict:
    """Closed trade for an RRG exit signalled the previous day: sold today at the average of open and close."""
    fields = {k: v for k, v in pending_exit.items() if k != "armed_after"}
    return dict(pos, sell=d, sell_px=_fill_px(open_px, close_px), open=False, **fields)


# ---- Signal Generator (Strategy 1: Daily + Weekly Confirmation) -------------
def gen_strat_1(daily: pd.DataFrame, asof: pd.DataFrame, start: pd.Timestamp, cost_pct: float = 0.12,
                opens: Optional[pd.Series] = None, entry: str = DEFAULT_ENTRY,
                rrg_exit: str = DEFAULT_RRG_EXIT) -> Tuple[List[Dict], List[Dict]]:
    """Strategy 1: Daily + Weekly RRG Confirmation.

    BUY  (no open position): Daily in {Leading, Improving} AND Weekly in {Leading, Improving}.
    SELL (open position):    Daily in {Weakening, Lagging} OR  Weekly in {Weakening, Lagging}.
    Otherwise an open position is held. Weekly is the as-of reading for that day (no look-ahead);
    a day without a weekly reading neither buys nor sells. Buys per `entry` and sells per `rrg_exit`
    (default for both: next trading day at the average of its open and close, see ENTRY_NEXT_AVG).
    """
    q = daily["Quadrant"]
    prev_q = q.shift(1)
    w_q = asof["Quadrant"].reindex(daily.index)
    close = daily["Close"]
    c = cost_pct / 100.0

    trades, skipped, pos, pending, pending_exit = [], [], None, None, None
    for d in daily.index[daily.index >= start]:
        if pending_exit is not None:     # next-day RRG exit: sell today
            trades.append(_sell_next_day(pos, pending_exit, d, _open_of(opens, d), close[d]))
            pos, pending_exit = None, None
            continue
        if pending is not None:          # next-day entry: fill today, exits are checked from tomorrow
            pos = dict(pending, buy=d, buy_px=_fill_px(_open_of(opens, d), close[d]))
            pending = None
            continue
        dq, wq = q[d], w_q[d]
        if pd.isna(wq):
            continue
        if pos is None:
            if dq in UP and wq in UP:
                sig = dict(signal=d, weekly=wq, frm=prev_q[d], into=dq, daily_at_buy=dq, weekly_at_buy=wq)
                if entry == ENTRY_CLOSE:
                    pos = dict(sig, buy=d, buy_px=close[d])
                else:
                    pending = sig
            elif dq in UP and prev_q[d] not in UP and pd.notna(prev_q[d]):
                skipped.append(dict(date=d, weekly=wq, frm=prev_q[d], into=dq))
        elif dq in DOWN or wq in DOWN:
            who = [n for n, x in (("Daily", dq), ("Weekly", wq)) if x in DOWN]
            reason = " & ".join(f"{n} → {x}" for n, x in (("Daily", dq), ("Weekly", wq)) if x in DOWN)
            if len(who) == 1:
                other = "Weekly" if who[0] == "Daily" else "Daily"
                reason += f" ({other} {wq if other == 'Weekly' else dq})"
            fields = dict(exit_signal=d, exit_into=dq, exit_reason=reason, daily_at_sell=dq, weekly_at_sell=wq)
            if rrg_exit == ENTRY_CLOSE:
                trades.append(dict(pos, sell=d, sell_px=close[d], open=False, **fields))
                pos = None
            else:
                pending_exit = fields
    if pos is not None:
        trades.append(dict(pos, sell=None, sell_px=close.iloc[-1], exit_into=None, open=True,
                           exit_reason="Open Position", exit_signal_pending=(pending_exit or {}).get("exit_signal"),
                           daily_at_sell=daily["Quadrant"].iloc[-1],
                           weekly_at_sell=w_q.iloc[-1] if len(w_q) else ""))
    for t in trades:
        t["ret"] = 100.0 * ((t["sell_px"] * (1 - c)) / (t["buy_px"] * (1 + c)) - 1)
        end = t["sell"] if t["sell"] is not None else daily.index[-1]
        t["held"] = int(((daily.index > t["buy"]) & (daily.index <= end)).sum())
    return trades, skipped


# ---- Signal Generators (Strategy 2: Weekly RRG only) ------------------------
# All Strategy 2 versions read the Weekly RRG on EVERY trading day: the weekly reading as it
# stands at that day's close (week so far, `weekly_asof`). That uses completed daily closes
# only - no look-ahead - and acts on the first day a signal is valid instead of waiting for
# the week's last session. Fills are at that day's close, like every strategy in the app.
def _weekly_days(daily: pd.DataFrame, asof: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """Daily rows (from `start`) that have a weekly reading, with that day's weekly values."""
    w = asof[["Quadrant", "RS_Momentum"]].reindex(daily.index)
    return w[(w.index >= start) & w["Quadrant"].notna()]


def _finish(trades, pos, daily, days, cost_pct, extra_open=None, pending_exit=None):
    """Close-out an open position at the latest close and add return / held sessions.
    pending_exit: an RRG exit signalled on the last day, to be sold the next trading day."""
    c = cost_pct / 100.0
    close = daily["Close"]
    if pos is not None:
        trades.append(dict(pos, sell=None, sell_px=close.iloc[-1], exit_into=None, open=True,
                           exit_reason="Open Position", daily_at_sell="",
                           exit_signal_pending=(pending_exit or {}).get("exit_signal"),
                           weekly_at_sell=days["Quadrant"].iloc[-1] if len(days) else "",
                           **(extra_open or {})))
    for t in trades:
        t["ret"] = 100.0 * ((t["sell_px"] * (1 - c)) / (t["buy_px"] * (1 + c)) - 1)
        end = t["sell"] if t["sell"] is not None else daily.index[-1]
        t["held"] = int(daily.index.searchsorted(end, side="right") - daily.index.searchsorted(t["buy"], side="right"))
    return trades, []


def gen_strat_2(daily: pd.DataFrame, asof: pd.DataFrame, start: pd.Timestamp, cost_pct: float = 0.12,
                opens: Optional[pd.Series] = None, entry: str = DEFAULT_ENTRY,
                rrg_exit: str = DEFAULT_RRG_EXIT) -> Tuple[List[Dict], List[Dict]]:
    """Strategy 2 (version 1): Weekly RRG only.

    Every trading day, on the weekly reading as of that day's close:
      BUY  (no open position): Weekly in {Leading, Improving}
      SELL (open position):    Weekly in {Weakening, Lagging}
    The Daily RRG is not used. Buys per `entry` and sells per `rrg_exit` (default for both: next
    trading day at the average of its open and close).
    """
    days = _weekly_days(daily, asof, start)
    prev_w = asof["Quadrant"].reindex(daily.index).shift(1)
    close = daily["Close"]
    trades, pos, pending, pending_exit = [], None, None, None
    for d, r in days.iterrows():
        wq = r["Quadrant"]
        if pending_exit is not None:
            trades.append(_sell_next_day(pos, pending_exit, d, _open_of(opens, d), close[d]))
            pos, pending_exit = None, None
            continue
        if pending is not None:
            pos = dict(pending, buy=d, buy_px=_fill_px(_open_of(opens, d), close[d]))
            pending = None
            continue
        if pos is None and wq in UP:
            sig = dict(signal=d, weekly=wq, frm=prev_w[d], into=wq, daily_at_buy="", weekly_at_buy=wq)
            if entry == ENTRY_CLOSE:
                pos = dict(sig, buy=d, buy_px=close[d])
            else:
                pending = sig
        elif pos is not None and wq in DOWN:
            fields = dict(exit_signal=d, exit_into=wq, exit_reason=f"Weekly → {wq}", daily_at_sell="", weekly_at_sell=wq)
            if rrg_exit == ENTRY_CLOSE:
                trades.append(dict(pos, sell=d, sell_px=close[d], open=False, **fields))
                pos = None
            else:
                pending_exit = fields
    return _finish(trades, pos, daily, days, cost_pct, pending_exit=pending_exit)


MOM_EXIT_V2 = 101.0   # Strategy 2 version 2: early exit level for weekly RS-Momentum in Leading


def gen_strat_2_v2(daily: pd.DataFrame, asof: pd.DataFrame, start: pd.Timestamp, cost_pct: float = 0.12,
                   opens: Optional[pd.Series] = None, entry: str = DEFAULT_ENTRY,
                   rrg_exit: str = DEFAULT_RRG_EXIT) -> Tuple[List[Dict], List[Dict]]:
    """Strategy 2, version 2: Weekly RRG only, sell when momentum falls through 101.

    Every trading day, on the weekly reading as of that day's close:
      SELL (open position): the trade has been Leading with RS-Momentum >= 101, and
                            RS-Momentum is now below 101 (in any quadrant)     -> early exit
                            or Weekly in {Weakening, Lagging}                  -> safety net
      BUY  (flat):          Weekly in {Leading, Improving}, provided the weekly has been in
                            Weakening/Lagging since the last early exit (fresh entry only).
    The Daily RRG is not used. Buys per `entry` and sells per `rrg_exit` (default for both: next
    trading day at the average of its open and close); both exits are weekly RRG exits.
    """
    days = _weekly_days(daily, asof, start)
    prev_w = asof["Quadrant"].reindex(daily.index).shift(1)
    close = daily["Close"]
    trades, pos, armed, peaked, pending, pending_exit = [], None, True, False, None, None
    for d, r in days.iterrows():
        wq, wm = r["Quadrant"], r["RS_Momentum"]
        if pending_exit is not None:
            armed = pending_exit["armed_after"]
            trades.append(_sell_next_day(pos, pending_exit, d, _open_of(opens, d), close[d]))
            pos, pending_exit = None, None
            continue
        if pending is not None:
            pos = dict(pending, buy=d, buy_px=_fill_px(_open_of(opens, d), close[d]))
            pending = None
            peaked = wq == "Leading" and wm >= MOM_EXIT_V2
            continue
        if pos is not None:
            if peaked and wm < MOM_EXIT_V2 and wq not in DOWN:
                reason = f"Weekly RS-Momentum fell below {MOM_EXIT_V2:.0f} ({wm:.2f}, {wq})"
            elif wq in DOWN:
                reason = f"Weekly → {wq}" + (f" (RS-Momentum {wm:.2f})" if peaked else "")
            else:
                if wq == "Leading" and wm >= MOM_EXIT_V2:
                    peaked = True
                continue
            fields = dict(exit_signal=d, exit_into=wq, exit_reason=reason, daily_at_sell="", weekly_at_sell=wq,
                          weekly_mom_at_sell=wm)
            if rrg_exit == ENTRY_CLOSE:
                trades.append(dict(pos, sell=d, sell_px=close[d], open=False, **fields))
                pos = None
                armed = wq in DOWN      # an early exit waits for a full down-and-up cycle
            else:
                pending_exit = dict(fields, armed_after=wq in DOWN)
            continue
        if wq in DOWN:
            armed = True
        elif armed and wq in UP:
            sig = dict(signal=d, weekly=wq, frm=prev_w[d], into=wq,
                       daily_at_buy="", weekly_at_buy=wq, weekly_mom_at_buy=wm)
            if entry == ENTRY_CLOSE:
                pos = dict(sig, buy=d, buy_px=close[d])
                peaked = wq == "Leading" and wm >= MOM_EXIT_V2
            else:
                pending = sig
    return _finish(trades, pos, daily, days, cost_pct, pending_exit=pending_exit)


STOP_LOSS_V3 = 10.0   # Strategy 2 version 3: stop-loss, % below the buy price


def gen_strat_2_v3(daily: pd.DataFrame, asof: pd.DataFrame, start: pd.Timestamp, cost_pct: float = 0.12,
                   take_profit_pct: Optional[float] = None, highs: Optional[pd.Series] = None,
                   stop_loss_pct: float = STOP_LOSS_V3, opens: Optional[pd.Series] = None,
                   entry: str = DEFAULT_ENTRY, rrg_exit: str = DEFAULT_RRG_EXIT) -> Tuple[List[Dict], List[Dict]]:
    """Strategy 2, version 3: Weekly RRG only (version 1) plus a 10% stop-loss on price
    (stop_loss_pct; 10 unless an analysis asks for another level).

    Every trading day's close:
      SELL (open position): close <= 90% of the buy price               -> stop-loss, loss booked
                            or Weekly (as of that day) in {Weakening, Lagging}
      BUY  (flat):          Weekly (as of that day) in {Leading, Improving}, provided the weekly
                            has been in Weakening/Lagging since the last stop-loss (fresh entry).
    The Daily RRG is never used - the daily check looks only at the price. Buys per `entry`; a weekly
    exit sells per `rrg_exit` (default for both: next trading day at the average of its open and close);
    the stop-loss sells at that day's close. Stop-loss and take-profit are measured from the buy price
    and checked from the day after the buy.

    Optional take-profit (take_profit_pct with daily highs): from the day after the buy, the first day
    whose high reaches the buy price + take_profit_pct sells that day at exactly that price. Like a
    stop-loss, it then waits for the weekly to drop into Weakening/Lagging and come back up before buying.
    """
    tp_highs = highs if take_profit_pct is not None else None
    return v3_trades(v3_inputs(daily, asof, start, tp_highs, opens), cost_pct, take_profit_pct, stop_loss_pct,
                     entry, rrg_exit)


def v3_inputs(daily: pd.DataFrame, asof: pd.DataFrame, start: pd.Timestamp,
              highs: Optional[pd.Series] = None, opens: Optional[pd.Series] = None) -> Dict[str, Any]:
    """Day-by-day arrays Strategy 2 v3 walks through. Built once per stock, they can be reused for any
    stop-loss / take-profit setting (the Tags analysis runs hundreds of settings per stock)."""
    days = _weekly_days(daily, asof, start)
    dates = days.index
    return {
        "daily": daily, "days": days, "dates": dates,
        "quads": days["Quadrant"].to_numpy(),
        "closes": daily["Close"].reindex(dates).to_numpy(),
        "prevs": asof["Quadrant"].reindex(daily.index).shift(1).reindex(dates).to_numpy(),
        "highs": highs.reindex(dates).to_numpy() if highs is not None else None,
        "opens": opens.reindex(dates).to_numpy() if opens is not None else np.full(len(dates), np.nan),
    }


def v3_trades(inp: Dict[str, Any], cost_pct: float = 0.12, take_profit_pct: Optional[float] = None,
              stop_loss_pct: float = STOP_LOSS_V3, entry: str = DEFAULT_ENTRY,
              rrg_exit: str = DEFAULT_RRG_EXIT) -> Tuple[List[Dict], List[Dict]]:
    """Strategy 2 v3 rules (see gen_strat_2_v3) on inputs from v3_inputs."""
    trades, _, _ = v3_run(inp, cost_pct, take_profit_pct, stop_loss_pct, entry, rrg_exit)
    return trades, []


def v3_run(inp: Dict[str, Any], cost_pct: float = 0.12, take_profit_pct: Optional[float] = None,
           stop_loss_pct: float = STOP_LOSS_V3, entry: str = DEFAULT_ENTRY,
           rrg_exit: str = DEFAULT_RRG_EXIT) -> Tuple[List[Dict], Optional[Dict], Optional[Dict]]:
    """Strategy 2 v3 rules on inputs from v3_inputs. Returns (trades, buy signal still to be filled, exit
    signal still to be sold): the last two are signals from the last day, executed on the next trading day."""
    dates, quads, closes, prevs = inp["dates"], inp["quads"], inp["closes"], inp["prevs"]
    use_tp = take_profit_pct is not None and inp["highs"] is not None
    day_highs, day_opens = inp["highs"], inp["opens"]
    trades, pos, armed, pending, pending_exit = [], None, True, None, None
    for i, d in enumerate(dates):
        wq, px = quads[i], closes[i]
        if pending_exit is not None:     # next-day weekly exit: sell today
            trades.append(_sell_next_day(pos, pending_exit, d, day_opens[i], px))
            pos, pending_exit, armed = None, None, True
            continue
        if pending is not None:          # next-day entry: fill today, exits are checked from tomorrow
            pos = dict(pending, buy=d, buy_px=_fill_px(day_opens[i], px))
            pending = None
            continue
        if pos is not None:
            stop_px = pos["buy_px"] * (1 - stop_loss_pct / 100.0)
            if use_tp:
                tp_px = pos["buy_px"] * (1 + take_profit_pct / 100.0)
                hi = day_highs[i]
                if hi == hi and hi >= tp_px:          # hi == hi: not NaN
                    trades.append(dict(pos, sell=d, sell_px=tp_px, exit_into=wq, open=False,
                                       exit_reason=f"Take-profit {take_profit_pct:g}%: high ₹{hi:.2f} ≥ ₹{tp_px:.2f}",
                                       daily_at_sell="", weekly_at_sell=wq, stop_px=stop_px))
                    pos = None
                    armed = wq in DOWN      # like a stop-loss: re-buy only after a weekly down-and-up
                    continue
            if px <= stop_px:
                reason = f"Stop-loss {stop_loss_pct:g}%: close ₹{px:.2f} ≤ ₹{stop_px:.2f} (Weekly {wq})"
            elif wq in DOWN:
                reason = f"Weekly → {wq}"
                if rrg_exit != ENTRY_CLOSE:   # weekly exit: sell on the next trading day
                    pending_exit = dict(exit_signal=d, exit_into=wq, exit_reason=reason, daily_at_sell="",
                                        weekly_at_sell=wq, stop_px=stop_px)
                    continue
            else:
                continue
            trades.append(dict(pos, sell=d, sell_px=px, exit_into=wq, open=False, exit_reason=reason,
                               daily_at_sell="", weekly_at_sell=wq, stop_px=stop_px))
            pos = None
            armed = wq in DOWN          # after a stop-loss in an up week, wait for down-and-up
            continue
        if wq in DOWN:
            armed = True
        elif armed and wq in UP:
            sig = dict(signal=d, weekly=wq, frm=prevs[i], into=wq, daily_at_buy="", weekly_at_buy=wq)
            if entry == ENTRY_CLOSE:
                pos = dict(sig, buy=d, buy_px=px)
            else:
                pending = sig
    stop_open = {"stop_px": pos["buy_px"] * (1 - stop_loss_pct / 100.0)} if pos is not None else None
    trades, _ = _finish(trades, pos, inp["daily"], inp["days"], cost_pct, stop_open, pending_exit)
    return trades, pending, pending_exit


SIGNALS = {1: gen_strat_1, 2: gen_strat_2, 3: gen_strat_2_v2, 4: gen_strat_2_v3}


# ---- Tag dashboard: Strategy 2 version 3 profitability per period ------------
TAG_PERIODS = [
    ("Period 1", pd.Timestamp("2022-10-01"), pd.Timestamp("2024-09-30")),
    ("Period 2", pd.Timestamp("2024-10-01"), pd.Timestamp("2026-09-30")),
]
TAG_DATA_FROM = pd.Timestamp("2022-04-30")   # a ticker counts only if its data starts by April 2022
TAG_TRADE_AMOUNT = 100000.0                  # standard investment per trade (Rs)
TAG_COST_PCT = 0.0                           # charges per buy / sell (%)
TAG_MAX_ROWS = 50                            # sample tickers shown per period table
TAG_STOP_LOSSES = [6.0, 8.0, 10.0, 12.0, 14.0]   # stop-loss levels compared, one take-profit chart each


def period_trades(trades: List[Dict], start: pd.Timestamp, end: pd.Timestamp) -> List[Dict]:
    """Closed trades bought and sold inside [start, end]; a trade spanning two periods counts in neither."""
    return [t for t in trades if t.get("sell") is not None and t["buy"] >= start and t["sell"] <= end]


def trade_profit_pct(t: Dict, amount: float = TAG_TRADE_AMOUNT, cost_pct: float = TAG_COST_PCT) -> float:
    """Profit % of one trade with a fixed investment: whole shares bought at the buy close, sold at the sell close."""
    c = cost_pct / 100.0
    shares = int(amount // (t["buy_px"] * (1 + c)))
    if shares == 0:
        return 0.0
    paid = shares * t["buy_px"] * (1 + c)
    got = shares * t["sell_px"] * (1 - c)
    return 100.0 * (got - paid) / paid


# take-profit thresholds (%) analysed on the Tags page: 1-20 in steps of 1, then 25-100 in steps of 5
TAKE_PROFIT_THRESHOLDS = list(range(1, 21)) + list(range(25, 101, 5))


def period_report(trades_by_ticker: Dict[str, List[Dict]], start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, Any]:
    """Tag-level report for one period: per-ticker summaries and the simple average of daily profit %.

    trades_by_ticker holds the qualifying tickers only. The average skips tickers without a closed trade.
    """
    rows = {t: ticker_period_summary(period_trades(tr, start, end)) for t, tr in trades_by_ticker.items()}
    daily = [s["daily_profit"] for s in rows.values() if s["daily_profit"] is not None]
    n_trades = sum(s["trades"] for s in rows.values())
    return {
        "rows": rows,
        "tickers": len(rows),
        "trades": n_trades,
        "avg_days_held": sum(s["days_sum"] for s in rows.values()) / n_trades if n_trades else None,
        "avg_daily_profit": sum(daily) / len(daily) if daily else None,
        "tickers_with_trades": len(daily),
    }


def ticker_period_summary(trades: List[Dict]) -> Dict[str, Any]:
    """Per-ticker row of the tag dashboard for one period's closed trades."""
    profits = [trade_profit_pct(t) for t in trades]
    days = [int((t["sell"] - t["buy"]).days) for t in trades]
    total_days = sum(days)
    return {
        "trades": len(trades),
        "profits": profits,
        "days": days,
        "profit_sum": sum(profits),
        "days_sum": total_days,
        "daily_profit": sum(profits) / total_days if total_days > 0 else None,
    }


# ---- Execution Simulator -----------------------------------------------------
def simulate_single_strategy(daily: pd.DataFrame, raw_trades: List[Dict], start: pd.Timestamp,
                             capital: float = 10000.0, cost_pct: float = 0.12) -> Tuple[List[Dict], float, pd.Series, pd.Series, int]:
    """
    Simulates realistic cash and share execution for Strategy 1.
    Ensures whole shares, transaction costs, mark-to-market daily portfolio value.
    """
    c = cost_pct / 100.0
    close = daily["Close"][daily.index >= start]
    cash = float(capital)
    book = []
    shares_by_day = pd.Series(0, index=close.index)

    for t in raw_trades:
        px_in = t["buy_px"]
        shares = int(cash // (px_in * (1 + c)))
        paid = shares * px_in * (1 + c)
        px_out = t["sell_px"]
        got = shares * px_out * (1 - c)
        charges = (shares * px_in * c) + (shares * px_out * c)
        cash_before = cash
        cash = cash - paid + got
        end = t["sell"] if t["sell"] is not None else close.index[-1]
        shares_by_day[(close.index >= t["buy"]) & (close.index < end)] = shares
        if t["sell"] is None:
            shares_by_day[close.index == end] = shares

        entry_dict = dict(t)
        entry_dict.update({
            "shares": shares,
            "paid": paid,
            "got": got,
            "pnl": got - paid,
            "pnl_pct": 100.0 * (got - paid) / paid if paid > 0 else 0.0,
            "charges": charges,
            "cash_before": cash_before,
            "cash_after": cash,
        })
        book.append(entry_dict)

    cash_by_day = pd.Series(float(capital), index=close.index)
    for b in book:
        end = b["sell"] if b["sell"] is not None else close.index[-1]
        inside = (close.index >= b["buy"]) & (close.index < end)
        cash_by_day[inside] = b["cash_before"] - b["paid"]
        cash_by_day[close.index >= end] = b["cash_after"]
        if b["sell"] is None:
            cash_by_day[close.index == end] = b["cash_before"] - b["paid"]

    value = cash_by_day + shares_by_day * close * (1 - c)
    if book and book[-1]["sell"] is None:
        value.iloc[-1] = book[-1]["cash_after"]

    # Buy & Hold Benchmark
    bh_shares = int(capital // (close.iloc[0] * (1 + c)))
    bh_left = capital - bh_shares * close.iloc[0] * (1 + c)
    bh_value = bh_left + bh_shares * close * (1 - c)

    return book, cash, value, bh_value, bh_shares


# ---- Strategy Runner ---------------------------------------------------------
def run_strategy(strat_id: int, daily: pd.DataFrame, asof: pd.DataFrame, start: pd.Timestamp,
                 capital: float = 10000.0, cost_pct: float = 0.12,
                 stocks_rrg: Optional[Dict] = None, bench_clean: Optional[pd.DataFrame] = None,
                 opens: Optional[pd.Series] = None, entry: str = DEFAULT_ENTRY,
                 rrg_exit: str = DEFAULT_RRG_EXIT) -> Tuple[List[Dict], float, pd.Series, pd.Series, int, Dict]:
    """Runs a strategy by id and returns (book, cash, value, bh_value, bh_shares, metrics)."""
    raw, _ = SIGNALS[strat_id](daily, asof, start, cost_pct, opens=opens, entry=entry, rrg_exit=rrg_exit)
    book, cash, val, bh_val, bh_sh = simulate_single_strategy(daily, raw, start, capital, cost_pct)
    metrics = compute_metrics(book, val, bh_val, capital, start)
    return book, cash, val, bh_val, bh_sh, metrics


# ---- Metrics Calculation -----------------------------------------------------
def compute_metrics(book: List[Dict], value: pd.Series, bh_value: pd.Series,
                    capital: float, start: pd.Timestamp) -> Dict[str, Any]:
    """Calculates full quantitative performance metrics for the strategy."""
    final_val = float(value.iloc[-1])
    net_pnl = final_val - capital
    return_pct = 100.0 * net_pnl / capital

    days = max(1, (value.index[-1] - start).days)
    years = days / 365.25
    if final_val > 0 and years > 0:
        cagr = 100.0 * ((final_val / capital) ** (1.0 / years) - 1.0)
    else:
        cagr = -100.0 if final_val <= 0 else 0.0

    peak = value.cummax()
    dd_rupees = peak - value
    max_dd_rupees = float(dd_rupees.max())
    dd_pct = 100.0 * dd_rupees / peak
    max_dd_pct = float(dd_pct.max())

    closed = [b for b in book if b.get("sell") is not None]
    wins = [b for b in closed if b.get("pnl", 0.0) > 0]
    losses = [b for b in closed if b.get("pnl", 0.0) <= 0]
    win_rate = (100.0 * len(wins) / len(closed)) if closed else 0.0

    gross_profit = sum(b.get("pnl", 0.0) for b in wins)
    gross_loss = abs(sum(b.get("pnl", 0.0) for b in losses))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = 999.0
    else:
        profit_factor = 0.0

    avg_return = float(np.mean([b["pnl_pct"] for b in closed])) if closed else 0.0
    avg_hold = float(np.mean([b["held"] for b in closed if "held" in b])) if closed else 0.0
    total_charges = sum(b.get("charges", 0.0) for b in book)

    bh_final = float(bh_value.iloc[-1])
    bh_pnl = bh_final - capital
    bh_return_pct = 100.0 * bh_pnl / capital
    alpha = return_pct - bh_return_pct

    return {
        "capital": capital,
        "final_value": final_val,
        "net_pnl": net_pnl,
        "return_pct": return_pct,
        "cagr_pct": cagr,
        "max_dd_rupees": max_dd_rupees,
        "max_dd_pct": max_dd_pct,
        "total_trades": len(closed) + (1 if book and book[-1].get("sell") is None else 0),
        "closed_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_pct": avg_return,
        "avg_holding_days": avg_hold,
        "total_charges": total_charges,
        "bh_final": bh_final,
        "bh_return_pct": bh_return_pct,
        "alpha_pct": alpha,
    }


def explain_pnl(book: List[Dict], capital: float, final_val: float, cost_pct: float,
                stock_name: str, strategy_name: str) -> Dict[str, Any]:
    """
    Computes mathematical and market attribution explaining why the strategy
    produced a profit or a loss. Returns both English and Hindi explanations.
    """
    closed = [b for b in book if b.get("sell") is not None]
    total_trades = len(closed)
    net_pnl = final_val - capital
    net_pnl_pct = 100.0 * net_pnl / capital if capital > 0 else 0.0

    winning_trades = [b for b in closed if b.get("pnl", 0.0) > 0]
    losing_trades = [b for b in closed if b.get("pnl", 0.0) < 0]

    gross_profit = sum(b.get("pnl", 0.0) for b in winning_trades)
    gross_loss = abs(sum(b.get("pnl", 0.0) for b in losing_trades))
    total_charges = sum(b.get("charges", 0.0) for b in book)

    win_rate = 100.0 * len(winning_trades) / total_trades if total_trades > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.9 if gross_profit > 0 else 0.0)

    avg_win_pct = float(np.mean([b["pnl_pct"] for b in winning_trades])) if winning_trades else 0.0
    avg_loss_pct = float(np.mean([b["pnl_pct"] for b in losing_trades])) if losing_trades else 0.0

    is_profit = net_pnl >= 0
    bullets_en = []
    bullets_hi = []

    if total_trades == 0:
        headline = "No Trades Executed (Capital Preserved in Cash)"
        summary_en = f"The strategy found no qualifying entry signals for {stock_name} during this backtest window. All ₹{capital:,.0f} remained safely in cash with ₹0 charges paid."
        summary_hi = f"Is period me {stock_name} ke liye koi entry signal trigger nahi hua. Sara paisa cash me safe raha aur ₹0 charges lage."
    elif is_profit:
        headline = f"Strategy Generated Profit: +₹{net_pnl:,.0f} ({net_pnl_pct:+.1f}%)"
        summary_en = f"{stock_name} generated a net profit of ₹{net_pnl:,.0f} ({net_pnl_pct:+.1f}%) across {total_trades} completed trades, turning ₹{capital:,.0f} into ₹{final_val:,.0f}."
        summary_hi = f"{stock_name} ne ₹{net_pnl:,.0f} ({net_pnl_pct:+.1f}%) ka profit banaya. ₹{capital:,.0f} ki shuruat badhkar ₹{final_val:,.0f} ho gayi."

        if avg_win_pct > abs(avg_loss_pct):
            bullets_en.append(f"Strong Win/Loss Asymmetry: Winning trades averaged {avg_win_pct:+.1f}%, far outpacing the average losing trade of {avg_loss_pct:+.1f}%.")
            bullets_hi.append(f"Munafa bada, nuksan chhota: Jeetne wali trades ka average profit (+{avg_win_pct:.1f}%) haarne wali trades ke loss ({avg_loss_pct:.1f}%) se kafi bada raha.")

        if profit_factor >= 1.2:
            bullets_en.append(f"High Profit Factor ({profit_factor:.2f}): For every ₹1.00 lost across losing trades, the strategy extracted ₹{profit_factor:.2f} on winning trades.")
            bullets_hi.append(f"Profit Factor {profit_factor:.2f}: Har ₹1 ke nuksan par strategy ne ₹{profit_factor:.2f} ka gross munafa kamaya.")

        bullets_en.append(f"Dual-Timeframe Alignment: Both Daily and Weekly timeframes confirmed the uptrend, capturing powerful multi-week momentum moves.")
        bullets_hi.append(f"Daily aur Weekly dono me tezi: Dono timeframes ne uptrend confirm kiya, jisse mazboot tezi ka pura fayda mila.")

        bullets_en.append(f"Disciplined Risk Management: Fast exits when either Daily or Weekly entered Weakening or Lagging protected gains from bear reversals.")
        bullets_hi.append(f"Samay par surakshit exit: Jaise hi Daily ya Weekly kamzor hua, turant exit karke munafa surakshit kiya gaya.")
    else:
        headline = f"Strategy Incurred Loss: -₹{abs(net_pnl):,.0f} ({net_pnl_pct:+.1f}%)"
        summary_en = f"{stock_name} experienced a net drawdown of -₹{abs(net_pnl):,.0f} ({net_pnl_pct:+.1f}%) across {total_trades} completed trades, reducing ₹{capital:,.0f} to ₹{final_val:,.0f}."
        summary_hi = f"{stock_name} me -₹{abs(net_pnl):,.0f} ({net_pnl_pct:+.1f}%) ka nuksan hua. ₹{capital:,.0f} ghatkar ₹{final_val:,.0f} reh gaye."

        if len(losing_trades) > len(winning_trades):
            bullets_en.append(f"Win Rate Drag: {len(losing_trades)} losing trades outnumbered {len(winning_trades)} winning trades (Win Rate: {win_rate:.0f}%).")
            bullets_hi.append(f"Loss trades ki sankhya zyada: {len(losing_trades)} trades me loss hua aur sirf {len(winning_trades)} me profit (Win rate: {win_rate:.0f}%).")

        bullets_en.append("Choppy / False Breakout: Stock entered Improving/Leading triggering buys, but momentum faded quickly, causing early stops.")
        bullets_hi.append("Whipsaw / False Breakout: Dono timeframes me tezi aane par khareeda, lekin momentum sustain nahi hua aur exit trigger ho gaya.")

        if total_charges > 0.01 * capital:
            bullets_en.append(f"Transaction Cost Friction: High turnover generated ₹{total_charges:,.0f} in charges ({100 * total_charges / capital:.1f}% of starting capital).")
            bullets_hi.append(f"Brokerage aur charges ka bojh: Baar-baar buy/sell karne se ₹{total_charges:,.0f} charges lage.")

    return {
        "is_profit": is_profit,
        "net_pnl": net_pnl,
        "net_pnl_pct": net_pnl_pct,
        "headline": headline,
        "summary_en": summary_en,
        "summary_hi": summary_hi,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "total_charges": total_charges,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "bullets_en": bullets_en,
        "bullets_hi": bullets_hi,
    }


def trade_outcome_reason(trade: Dict, cost_pct: float = 0.12) -> str:
    """Explains why an individual trade resulted in profit or loss."""
    if trade.get("sell") is None:
        pnl_pct = trade.get("pnl_pct", 0.0)
        sym = "🟢" if pnl_pct >= 0 else "🔴"
        return f"⏳ Holding Active ({sym} {pnl_pct:+.1f}% mark-to-market)"

    pnl = trade.get("pnl", 0.0)
    pnl_pct = trade.get("pnl_pct", 0.0)
    reason = trade.get("exit_reason", "")

    if pnl > 0:
        return f"✅ Profit (+{pnl_pct:.1f}%): Trend ran well; exited on {reason}"
    elif pnl < 0:
        return f"❌ Loss ({pnl_pct:.1f}%): Exited on {reason} to prevent larger decline"
    else:
        return "⚖️ Flat (0.0%): Exited near breakeven"
