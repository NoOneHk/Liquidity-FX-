import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import traceback
import io
import uuid
import random
import math
from datetime import datetime, timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from anthropic import Anthropic

# =============================================================================
# PAGE CONFIG + GLOBAL STYLING
# =============================================================================

st.set_page_config(
    page_title="AI Trading Journal & Backtester Pro",
    page_icon="📈",
    layout="centered",

    initial_sidebar_state="expanded",
)

CUSTOM_CSS = ""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================================================================
# DATABASE LAYER (SQLite) — extended schema
# =============================================================================

DB_PATH = "trading_journal.db"


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS backtests (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            strategy_name TEXT,
            strategy_rules TEXT,
            generated_code TEXT,
            asset_name TEXT,
            data_source TEXT,
            starting_balance REAL,
            win_rate REAL,
            total_pnl REAL,
            max_drawdown REAL,
            profit_factor REAL,
            total_trades INTEGER,
            avg_win REAL,
            avg_loss REAL,
            achieved_rr REAL,
            target_rr REAL,
            risk_type TEXT,
            risk_value REAL,
            suggested_position_size REAL,
            equity_curve TEXT,
            drawdown_curve TEXT,
            trades TEXT,
            closed_trades TEXT,
            notes TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_backtest_to_journal(record: dict):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO backtests (
            id, created_at, strategy_name, strategy_rules, generated_code,
            asset_name, data_source, starting_balance, win_rate, total_pnl,
            max_drawdown, profit_factor, total_trades, avg_win, avg_loss,
            achieved_rr, target_rr, risk_type, risk_value, suggested_position_size,
            equity_curve, drawdown_curve, trades, closed_trades, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record["id"], record["created_at"], record["strategy_name"],
            record["strategy_rules"], record["generated_code"], record["asset_name"],
            record["data_source"], record["starting_balance"], record["win_rate"],
            record["total_pnl"], record["max_drawdown"], record["profit_factor"],
            record["total_trades"], record["avg_win"], record["avg_loss"],
            record["achieved_rr"], record["target_rr"], record["risk_type"],
            record["risk_value"], record["suggested_position_size"],
            json.dumps(record["equity_curve"]), json.dumps(record["drawdown_curve"]),
            json.dumps(record["trades"]), json.dumps(record["closed_trades"]),
            record.get("notes", ""),
        ),
    )
    conn.commit()
    conn.close()


def load_journal_entries():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM backtests ORDER BY created_at DESC", conn)
    conn.close()
    return df


def update_notes(entry_id: str, notes: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE backtests SET notes = ? WHERE id = ?", (notes, entry_id))
    conn.commit()
    conn.close()


def delete_entry(entry_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM backtests WHERE id = ?", (entry_id,))
    conn.commit()
    conn.close()


init_db()

# =============================================================================
# CLAUDE API INTEGRATION — PLAIN ENGLISH -> PANDAS STRATEGY CODE
# =============================================================================

STRATEGY_SYSTEM_PROMPT = """You are an expert quantitative Python developer.
You convert plain-English trading strategy rules into a single, complete,
runnable Python function that backtests the strategy on OHLCV data using
pandas and numpy only.

STRICT OUTPUT RULES:
- Output ONLY raw Python code. No markdown fences, no explanations, no comments
  like "implement here". Every line must be real, working code.
- The code must define exactly one function with this exact signature:

def run_strategy(df):
    ...
    return metrics_dict, equity_curve_list, trades_list, closed_trades_list

- `df` is a pandas DataFrame already sorted by date ascending with columns:
  'DateTime' (pandas Timestamp), 'Open', 'High', 'Low', 'Close', 'Volume',
  and a float column 'starting_balance' broadcast to every row (read it via
  df['starting_balance'].iloc[0]).
- MULTI-TIMEFRAME SUPPORT: if the plain-English rules reference a higher
  timeframe trend filter (e.g. "check the 1-hour trend while trading the
  5-minute chart"), build the higher timeframe series inside run_strategy by
  doing: htf = df.set_index('DateTime')[['Open','High','Low','Close']].resample('1H').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
  then compute your trend condition on `htf` and reindex/forward-fill it back
  onto the original `df` index using pd.merge_asof or .reindex(method='ffill')
  before combining it with your lower-timeframe entry signal. Never assume
  a fixed bar interval; infer it from consecutive df['DateTime'] differences
  if you need it.
- Inside the function you must:
  1. Compute a trading signal column (1 = long/buy, -1 = short/sell, 0 = flat)
     based strictly on the user's plain-English rules, using vectorized
     pandas/numpy operations or simple loops. Do not write import statements;
     pd and np are already available in scope.
  2. Simulate trades sequentially bar by bar: enter/exit positions according
     to signal changes, apply the position to the next bar's return, and
     track a running account balance starting from df['starting_balance'].iloc[0].
  3. Every time a position is opened or closed, append an entry to
     trades_list as a dict:
       {"index": <int row index in df>, "datetime": <str of df['DateTime']>,
        "action": "buy" or "sell", "price": <float execution price>}
     Use "buy" for opening a long or closing/covering a short.
     Use "sell" for opening a short or closing/exiting a long.
  4. Every time a round-trip position is fully closed, append an entry to
     closed_trades_list as a dict:
       {"entry_index": <int>, "exit_index": <int>, "entry_price": <float>,
        "exit_price": <float>, "direction": "long" or "short", "pnl": <float>}
  5. Compute metrics_dict with EXACTLY these keys (all floats, never None,
     never NaN — use 0.0 if undefined): 'win_rate' (0-100 percentage),
     'total_pnl' (final balance minus starting balance), 'max_drawdown'
     (positive percentage 0-100), 'profit_factor' (gross profit / gross
     loss, 0.0 if gross loss is 0), 'total_trades' (count of closed trades,
     cast to float).
  6. Build equity_curve_list as a plain Python list of floats representing
     the account balance at every bar (same length as df).
  7. Return exactly: return metrics_dict, equity_curve_list, trades_list, closed_trades_list
- Never use exec, eval, os, sys, open, requests, socket, subprocess, or any
  file/network/system access.
- Never define any other top-level function or class, only run_strategy.
- Handle edge cases (empty df, no signals) so the function never raises and
  returns empty lists / zeroed metrics instead.
- Do not print anything.
"""


def generate_strategy_code(api_key: str, plain_english_rules: str, model: str = "claude-sonnet-5") -> str:
    """
    Calls the Anthropic API to translate plain-English trading rules into a
    complete, executable Python pandas backtesting function.
    """
    client = Anthropic(api_key=api_key)

    user_prompt = (
        "Convert the following plain-English trading strategy rules into the "
        "run_strategy(df) function as specified in your system instructions.\n\n"
        "STRATEGY RULES:\n"
        f"{plain_english_rules.strip()}\n\n"
        "Remember: output ONLY the raw Python code for the function, nothing else."
    )

    response = client.messages.create(
        model=model,
        max_tokens=4500,
        temperature=0,
        system=STRATEGY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw_text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
    code = raw_text.strip()

    if code.startswith("```"):
        lines = code.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines).strip()

    return code


# =============================================================================
# SAFE EXECUTION SANDBOX FOR THE GENERATED CODE
# =============================================================================

FORBIDDEN_TOKENS = [
    "import os", "import sys", "subprocess", "socket", "shutil",
    "open(", "__import__", "eval(", "exec(", "requests", "urllib",
    "os.system", "os.popen", "input(", "compile(", "globals(", "locals(",
    "__class__", "__bases__", "__subclasses__",
]


def validate_generated_code(code: str):
    lowered = code.lower()
    violations = [tok for tok in FORBIDDEN_TOKENS if tok.lower() in lowered]
    if "def run_strategy" not in code:
        violations.append("missing run_strategy definition")
    return violations


def run_generated_strategy(code: str, df: pd.DataFrame, starting_balance: float):
    """
    Executes the AI-generated run_strategy(df) function inside a restricted
    namespace. Returns (metrics_dict, equity_curve_list, trades_list,
    closed_trades_list, error_message).
    """
    violations = validate_generated_code(code)
    if violations:
        return None, None, None, None, f"Generated code rejected by safety validator: {violations}"

    safe_builtins = {
        "range": range, "len": len, "float": float, "int": int, "str": str,
        "bool": bool, "list": list, "dict": dict, "tuple": tuple, "set": set,
        "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
        "enumerate": enumerate, "zip": zip, "sorted": sorted, "reversed": reversed,
        "any": any, "all": all, "isinstance": isinstance, "Exception": Exception,
        "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
        "IndexError": IndexError, "ZeroDivisionError": ZeroDivisionError,
        "True": True, "False": False, "None": None,
    }
    sandbox_globals = {"__builtins__": safe_builtins, "pd": pd, "np": np}
    sandbox_locals = {}

    try:
        exec(code, sandbox_globals, sandbox_locals)
    except Exception:
        return None, None, None, None, f"Error compiling generated code:\n{traceback.format_exc()}"

    run_strategy_fn = sandbox_locals.get("run_strategy") or sandbox_globals.get("run_strategy")
    if run_strategy_fn is None:
        return None, None, None, None, "Generated code did not define a run_strategy function."

    work_df = df.copy()
    work_df["starting_balance"] = float(starting_balance)

    try:
        result = run_strategy_fn(work_df)
    except Exception:
        return None, None, None, None, f"Error executing run_strategy(df):\n{traceback.format_exc()}"

    if not isinstance(result, tuple) or len(result) != 4:
        return None, None, None, None, (
            "run_strategy(df) must return exactly 4 values: "
            "metrics_dict, equity_curve_list, trades_list, closed_trades_list."
        )

    metrics_dict, equity_curve_list, trades_list, closed_trades_list = result

    required_keys = {"win_rate", "total_pnl", "max_drawdown", "profit_factor", "total_trades"}
    if not isinstance(metrics_dict, dict) or not required_keys.issubset(metrics_dict.keys()):
        return None, None, None, None, f"metrics_dict missing required keys. Got: {metrics_dict}"

    if not isinstance(equity_curve_list, (list, tuple, np.ndarray, pd.Series)):
        return None, None, None, None, "equity_curve_list must be a list-like of floats."

    equity_curve_list = [float(x) for x in equity_curve_list]

    for k in required_keys:
        try:
            metrics_dict[k] = float(metrics_dict[k])
        except (TypeError, ValueError):
            metrics_dict[k] = 0.0
        if np.isnan(metrics_dict[k]) or np.isinf(metrics_dict[k]):
            metrics_dict[k] = 0.0

    clean_trades = []
    if isinstance(trades_list, (list, tuple)):
        for t in trades_list:
            if not isinstance(t, dict):
                continue
            try:
                clean_trades.append({
                    "index": int(t.get("index", 0)),
                    "datetime": str(t.get("datetime", "")),
                    "action": str(t.get("action", "")).lower(),
                    "price": float(t.get("price", 0.0)),
                })
            except (TypeError, ValueError):
                continue

    clean_closed = []
    if isinstance(closed_trades_list, (list, tuple)):
        for t in closed_trades_list:
            if not isinstance(t, dict):
                continue
            try:
                clean_closed.append({
                    "entry_index": int(t.get("entry_index", 0)),
                    "exit_index": int(t.get("exit_index", 0)),
                    "entry_price": float(t.get("entry_price", 0.0)),
                    "exit_price": float(t.get("exit_price", 0.0)),
                    "direction": str(t.get("direction", "long")).lower(),
                    "pnl": float(t.get("pnl", 0.0)),
                })
            except (TypeError, ValueError):
                continue

    return metrics_dict, equity_curve_list, clean_trades, clean_closed, None


# =============================================================================
# RISK MANAGEMENT / POSITION SIZING CALCULATIONS
# =============================================================================

def compute_drawdown_series(equity_curve):
    if not equity_curve:
        return []
    drawdowns = []
    peak = equity_curve[0]
    for v in equity_curve:
        peak = max(peak, v)
        dd = ((peak - v) / peak * 100.0) if peak > 0 else 0.0
        drawdowns.append(round(dd, 4))
    return drawdowns


def compute_risk_metrics(closed_trades, risk_type: str, risk_value: float, starting_balance: float, target_rr: float):
    """
    Derives average win/loss, achieved risk:reward, and a suggested position
    size from the strategy's historical closed trades and the user's risk
    settings. This is an estimate based on average historical stop distance,
    not a guarantee of future results.
    """
    wins = [t["pnl"] for t in closed_trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in closed_trades if t["pnl"] <= 0]

    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean([abs(x) for x in losses])) if losses else 0.0
    achieved_rr = (avg_win / avg_loss) if avg_loss > 0 else 0.0

    stop_distances = [
        abs(t["entry_price"] - t["exit_price"])
        for t in closed_trades
        if t["pnl"] <= 0 and t["entry_price"] != t["exit_price"]
    ]
    avg_stop_distance = float(np.mean(stop_distances)) if stop_distances else 0.0

    if risk_type == "Percent of Balance":
        risk_amount = starting_balance * (risk_value / 100.0)
    else:
        risk_amount = risk_value

    if avg_stop_distance > 0:
        suggested_position_size = risk_amount / avg_stop_distance
    else:
        suggested_position_size = 0.0

    return {
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "achieved_rr": round(achieved_rr, 4),
        "risk_amount": round(risk_amount, 2),
        "avg_stop_distance": round(avg_stop_distance, 6),
        "suggested_position_size": round(suggested_position_size, 4),
        "target_rr": float(target_rr),
        "rr_gap": round(achieved_rr - target_rr, 4),
    }


# =============================================================================
# CSV VALIDATION / LOADING
# =============================================================================

REQUIRED_COLUMNS = ["DateTime", "Open", "High", "Low", "Close", "Volume"]


def load_and_validate_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, f"Could not read CSV: {e}"

    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, f"CSV is missing required columns: {missing}. Found columns: {list(df.columns)}"

    try:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
    except Exception as e:
        return None, f"Could not parse DateTime column: {e}"

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df.sort_values("DateTime").reset_index(drop=True)

    if len(df) < 5:
        return None, "CSV must contain at least 5 valid rows of OHLC data."
