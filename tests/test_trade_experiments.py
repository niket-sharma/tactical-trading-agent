from tests.test_core import make_price_df
from trading_agent.trade_engine import TradeSystemConfig
from trading_agent.trade_engine.experiments import run_param_sweep


def test_run_param_sweep_produces_metrics():
    prices = make_price_df(80)
    cfg = TradeSystemConfig()
    param_grid = {
        "rsi_long_threshold": [50.0],
        "rsi_short_threshold": [55.0],
        "pullback_max_pct": [0.05],
        "stop_loss_pct": [0.10],
        "take_profit_max_pct": [0.15],
        "trailing_stop_pct": [0.07],
        "max_position_fraction": [0.20],
        "typical_position_fraction": [0.10],
    }

    results = run_param_sweep(prices, cfg, param_grid=param_grid)

    assert len(results) == 1
    expected_cols = {"cagr", "sharpe", "max_dd", "daily_vol", "trades", "final_equity"}
    assert expected_cols.issubset(set(results.columns))
