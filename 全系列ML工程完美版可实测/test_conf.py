from backtester import WalkForwardBacktester

print("Running backtest for ALL models (including ML) to check baseline performance...")
bt = WalkForwardBacktester()
bt.run(warmup_shoes=5, min_rounds_before_predict=10, verbose=True, enabled_models=None)
