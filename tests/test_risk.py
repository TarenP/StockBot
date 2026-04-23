from types import SimpleNamespace

from broker.risk import PortfolioRiskEngine


def test_start_session_resets_daily_loss_when_session_key_changes():
    portfolio = SimpleNamespace(equity=95.0, cash=95.0, positions={})
    risk = PortfolioRiskEngine(max_daily_loss=0.04, max_drawdown=0.50)

    risk.start_session(100.0, session_key="2024-01-02")
    status_before_reset, _ = risk.check_portfolio_health(portfolio)

    risk.start_session(95.0, session_key="2024-01-03")
    status_after_reset, _ = risk.check_portfolio_health(portfolio)

    assert status_before_reset == "halt"
    assert status_after_reset == "ok"


def test_risk_engine_regime_tracking_does_not_change_cash_floor():
    portfolio = SimpleNamespace(equity=100.0, cash=5.0, positions={})
    risk = PortfolioRiskEngine(cash_floor=0.05, max_gross_exposure=0.95)

    risk.set_market_regime(0)
    risk_on_allowed, _ = risk.check_pre_trade(alloc_value=1.0, portfolio=portfolio)

    risk.set_market_regime(3)
    risk_off_allowed, _ = risk.check_pre_trade(alloc_value=1.0, portfolio=portfolio)

    assert risk_on_allowed is False
    assert risk_off_allowed is False


def test_risk_engine_vol_target_is_stable_across_regimes():
    history = [100.0, 102.0, 99.0, 103.0, 98.0, 104.0, 100.0]
    risk = PortfolioRiskEngine(
        target_volatility=0.15,
        vol_lookback=5,
        equity_history_getter=lambda: history,
    )

    risk.set_market_regime(0)
    risk_on_alloc = risk.vol_scale_allocation(100.0)

    risk.set_market_regime(3)
    risk_off_alloc = risk.vol_scale_allocation(100.0)

    assert risk_on_alloc == risk_off_alloc
