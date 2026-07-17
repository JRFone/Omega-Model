from stock_model.native_benchmark import NativeBenchmarkSettings, run_native_benchmark


def test_native_benchmark_reports_parity():
    result = run_native_benchmark(NativeBenchmarkSettings(candidates=40, years=20, repeats=1, seed=31))
    assert result["valid_comparisons"] > 0
    assert result["parity_pass"]
    assert result["python_seconds"] >= 0
    assert result["native_seconds"] >= 0
