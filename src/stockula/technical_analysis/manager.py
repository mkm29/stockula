"""Manager for coordinating technical analysis strategies."""

from typing import Any

import pandas as pd

from stockula.config import StockulaConfig
from stockula.interfaces import IDataFetcher, ILoggingManager

from .indicators import TechnicalIndicators


class TechnicalAnalysisManager:
    """Manages different technical analysis strategies and provides unified interface."""

    def __init__(self, data_fetcher: IDataFetcher, logging_manager: ILoggingManager):
        """Initialize TechnicalAnalysisManager.

        Args:
            data_fetcher: Data fetcher instance
            logging_manager: Logging manager instance
        """
        self.data_fetcher = data_fetcher
        self.logger = logging_manager

        # Predefined indicator groups for different analysis strategies
        self.indicator_groups = {
            "basic": ["sma", "ema", "rsi", "volume"],
            "momentum": ["rsi", "macd", "stoch", "adx", "cci", "williams_r"],
            "trend": ["sma", "ema", "macd", "adx", "ichimoku"],
            "volatility": ["bbands", "atr", "stoch"],
            "volume": ["obv", "volume"],
            "comprehensive": [
                "sma",
                "ema",
                "rsi",
                "macd",
                "bbands",
                "stoch",
                "atr",
                "adx",
                "williams_r",
                "cci",
                "obv",
                "ichimoku",
            ],
        }

        # Default parameters for indicators
        self.default_params = {
            "sma": {"period": 20},
            "ema": {"period": 20},
            "rsi": {"period": 14},
            "macd": {"period_fast": 12, "period_slow": 26, "signal": 9},
            "bbands": {"period": 20, "std": 2},
            "stoch": {"period": 14},
            "atr": {"period": 14},
            "adx": {"period": 14},
            "williams_r": {"period": 14},
            "cci": {"period": 20},
            "obv": {},
            "ichimoku": {"tenkan": 9, "kijun": 26, "senkou": 52},
        }

    def analyze_symbol(
        self,
        symbol: str,
        config: StockulaConfig,
        analysis_type: str = "comprehensive",
        custom_indicators: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Analyze a single symbol with specified indicators."""
        try:
            data = self._fetch_data(symbol, config, start_date, end_date)
            if data.empty:
                return {"ticker": symbol, "error": "No data available"}

            ta = TechnicalIndicators(data)
            indicators = self._select_indicators(analysis_type, custom_indicators)
            ta_config = config.technical_analysis

            results = {
                "ticker": symbol,
                "current_price": data["Close"].iloc[-1],
                "analysis_type": analysis_type,
                "indicators": self._calculate_indicators(ta, indicators, ta_config, symbol),
            }
            results["summary"] = self._generate_analysis_summary(results["indicators"], data)
            return results

        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {str(e)}")
            return {"ticker": symbol, "error": str(e)}

    def _fetch_data(
        self, symbol: str, config: StockulaConfig, start_date: str | None, end_date: str | None
    ) -> pd.DataFrame:
        start = start_date or config.data.start_date
        end = end_date or config.data.end_date
        if hasattr(start, "strftime"):
            start = start.strftime("%Y-%m-%d")  # type: ignore[union-attr]
        if hasattr(end, "strftime"):
            end = end.strftime("%Y-%m-%d")  # type: ignore[union-attr]
        return self.data_fetcher.get_stock_data(symbol, start=start, end=end)

    def _select_indicators(self, analysis_type: str, custom_indicators: list[str] | None) -> list[str]:
        if custom_indicators:
            return custom_indicators
        return self.indicator_groups.get(analysis_type, self.indicator_groups["comprehensive"])

    def _calculate_indicators(
        self,
        ta: TechnicalIndicators,
        indicators: list[str],
        ta_config: Any,
        symbol: str,
    ) -> dict[str, Any]:
        results = {}
        for indicator in indicators:
            if indicator == "volume":
                results["volume"] = self._format_volume(ta.data)
            elif hasattr(ta, indicator):
                results[indicator] = self._safe_calculate_indicator(ta, indicator, ta_config, symbol)
        return results

    def _format_volume(self, data: pd.DataFrame) -> dict[str, Any]:
        return {
            "current": data["Volume"].iloc[-1],
            "average": data["Volume"].mean(),
            "ratio": data["Volume"].iloc[-1] / data["Volume"].mean(),
        }

    def _safe_calculate_indicator(
        self,
        ta: TechnicalIndicators,
        indicator: str,
        ta_config: Any,
        symbol: str,
    ) -> dict[str, Any]:
        try:
            params = self._get_indicator_params(indicator, ta_config)
            indicator_func = getattr(ta, indicator)
            result = indicator_func(**params)
            return self._format_indicator_result(result)
        except Exception as e:
            self.logger.warning(f"Failed to calculate {indicator} for {symbol}: {str(e)}")
            return {"error": str(e)}

    def _format_indicator_result(self, result: Any) -> dict[str, Any]:
        if isinstance(result, pd.Series):
            return {
                "current": result.iloc[-1] if not result.empty else None,
                "values": result.to_dict() if len(result) <= 10 else None,
            }
        elif isinstance(result, pd.DataFrame):
            return {
                "current": {col: result[col].iloc[-1] for col in result.columns if not result.empty},
                "values": result.to_dict() if len(result) <= 10 else None,
            }
        return {"value": result}

    def analyze_multiple_symbols(
        self,
        symbols: list[str],
        config: StockulaConfig,
        analysis_type: str = "comprehensive",
        custom_indicators: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Analyze multiple symbols.

        Args:
            symbols: List of stock symbols
            config: Configuration object
            analysis_type: Type of analysis
            custom_indicators: Custom list of indicators

        Returns:
            Dictionary mapping symbols to their analysis results
        """
        results = {}
        for symbol in symbols:
            self.logger.info(f"Analyzing technical indicators for {symbol}")
            results[symbol] = self.analyze_symbol(symbol, config, analysis_type, custom_indicators)
        return results

    def quick_analysis(self, symbol: str, start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
        """Perform quick basic analysis with key indicators.

        Args:
            symbol: Stock symbol
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Dictionary with basic analysis results
        """
        try:
            data = self.data_fetcher.get_stock_data(symbol, start=start_date, end=end_date)
            if data.empty:
                return {"ticker": symbol, "error": "No data available"}

            ta = TechnicalIndicators(data)

            # Calculate only essential indicators
            sma20 = ta.sma(20)
            ema20 = ta.ema(20)
            rsi = ta.rsi(14)

            current_price = data["Close"].iloc[-1]

            return {
                "ticker": symbol,
                "current_price": current_price,
                "analysis_type": "quick",
                "sma20": sma20.iloc[-1] if not sma20.empty else None,
                "ema20": ema20.iloc[-1] if not ema20.empty else None,
                "rsi": rsi.iloc[-1] if not rsi.empty else None,
                "price_vs_sma20": (current_price - sma20.iloc[-1]) / sma20.iloc[-1] * 100 if not sma20.empty else None,
                "volume_ratio": data["Volume"].iloc[-1] / data["Volume"].mean(),
                "trend": self._determine_trend(data, sma20, ema20),
                "momentum": self._determine_momentum(rsi.iloc[-1] if not rsi.empty else None),
            }

        except Exception as e:
            self.logger.error(f"Error in quick analysis for {symbol}: {str(e)}")
            return {"ticker": symbol, "error": str(e)}

    def momentum_analysis(self, symbol: str, config: StockulaConfig) -> dict[str, Any]:
        """Perform momentum-focused analysis.

        Args:
            symbol: Stock symbol
            config: Configuration object

        Returns:
            Dictionary with momentum analysis results
        """
        return self.analyze_symbol(symbol, config, analysis_type="momentum")

    def trend_analysis(self, symbol: str, config: StockulaConfig) -> dict[str, Any]:
        """Perform trend-focused analysis.

        Args:
            symbol: Stock symbol
            config: Configuration object

        Returns:
            Dictionary with trend analysis results
        """
        return self.analyze_symbol(symbol, config, analysis_type="trend")

    def volatility_analysis(self, symbol: str, config: StockulaConfig) -> dict[str, Any]:
        """Perform volatility-focused analysis.

        Args:
            symbol: Stock symbol
            config: Configuration object

        Returns:
            Dictionary with volatility analysis results
        """
        return self.analyze_symbol(symbol, config, analysis_type="volatility")

    def get_indicator_groups(self) -> dict[str, list[str]]:
        """Get available indicator groups.

        Returns:
            Dictionary of indicator groups
        """
        return self.indicator_groups.copy()

    def get_available_indicators(self) -> list[str]:
        """Get all available indicators.

        Returns:
            List of available indicator names
        """
        # Get all unique indicators from all groups
        all_indicators = set()
        for indicators in self.indicator_groups.values():
            all_indicators.update(indicators)
        return sorted(all_indicators)

    def calculate_custom_indicators(
        self,
        symbol: str,
        indicators: dict[str, dict[str, Any]],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Calculate custom indicators with specific parameters.

        Args:
            symbol: Stock symbol
            indicators: Dictionary mapping indicator names to their parameters
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Dictionary with calculated indicators
        """
        try:
            data = self.data_fetcher.get_stock_data(symbol, start=start_date, end=end_date)
            if data.empty:
                return {"ticker": symbol, "error": "No data available"}

            ta = TechnicalIndicators(data)
            results = {"ticker": symbol, "current_price": data["Close"].iloc[-1], "indicators": {}}

            for indicator_name, params in indicators.items():
                results["indicators"][indicator_name] = self._calculate_single_custom_indicator(
                    ta, indicator_name, params
                )

            return results

        except Exception as e:
            self.logger.error(f"Error calculating custom indicators for {symbol}: {str(e)}")
            return {"ticker": symbol, "error": str(e)}

    def _calculate_single_custom_indicator(
        self, ta: TechnicalIndicators, indicator_name: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Helper to calculate a single custom indicator."""
        if not hasattr(ta, indicator_name):
            return {"error": f"Unknown indicator: {indicator_name}"}
        try:
            indicator_func = getattr(ta, indicator_name)
            result = indicator_func(**params)
            if isinstance(result, pd.Series):
                return {
                    "current": result.iloc[-1] if not result.empty else None,
                    "params": params,
                }
            elif isinstance(result, pd.DataFrame):
                return {
                    "current": {col: result[col].iloc[-1] for col in result.columns if not result.empty},
                    "params": params,
                }
            else:
                return {"value": result, "params": params}
        except Exception as e:
            return {"error": str(e), "params": params}

    def _get_indicator_params(self, indicator: str, ta_config: Any) -> dict[str, Any]:
        """Get parameters for an indicator from config or defaults.

        Args:
            indicator: Indicator name
            ta_config: Technical analysis configuration

        Returns:
            Dictionary of parameters
        """
        # Check if custom parameters are defined in config
        if hasattr(ta_config, indicator):
            config_params = getattr(ta_config, indicator)
            if isinstance(config_params, dict):
                return config_params

        # Use default parameters
        return self.default_params.get(indicator, {})

    def _generate_analysis_summary(self, indicators: dict[str, Any], data: pd.DataFrame) -> dict[str, Any]:
        """Generate a summary of the technical analysis.

        Args:
            indicators: Calculated indicators
            data: Price data

        Returns:
            Dictionary with analysis summary
        """
        summary: dict[str, Any] = {"signals": [], "strength": "neutral"}

        summary["signals"].extend(self._rsi_signals(indicators))
        summary["signals"].extend(self._macd_signals(indicators))
        summary["signals"].extend(self._price_vs_sma_signals(indicators, data))

        summary["strength"] = self._determine_strength(summary["signals"])
        return summary

    def _rsi_signals(self, indicators: dict[str, Any]) -> list[str]:
        signals = []
        rsi = indicators.get("rsi", {}).get("current")
        if rsi is not None:
            if rsi > 70:
                signals.append("RSI Overbought")
            elif rsi < 30:
                signals.append("RSI Oversold")
        return signals

    def _macd_signals(self, indicators: dict[str, Any]) -> list[str]:
        signals = []
        macd = indicators.get("macd", {}).get("current")
        if isinstance(macd, dict):
            macd_val = macd.get("MACD")
            macd_signal = macd.get("MACD_SIGNAL")
            if macd_val is not None and macd_signal is not None:
                if macd_val > macd_signal:
                    signals.append("MACD Bullish")
                else:
                    signals.append("MACD Bearish")
        return signals

    def _price_vs_sma_signals(self, indicators: dict[str, Any], data: pd.DataFrame) -> list[str]:
        signals = []
        current_price = data["Close"].iloc[-1]
        sma = indicators.get("sma", {}).get("current")
        if sma is not None:
            if current_price > sma:
                signals.append("Price above SMA")
            else:
                signals.append("Price below SMA")
        return signals

    def _determine_strength(self, signals: list[str]) -> str:
        bullish_signals = sum(
            1 for signal in signals if "Bullish" in signal or "above" in signal or "Oversold" in signal
        )
        bearish_signals = sum(
            1 for signal in signals if "Bearish" in signal or "below" in signal or "Overbought" in signal
        )
        if bullish_signals > bearish_signals:
            return "bullish"
        elif bearish_signals > bullish_signals:
            return "bearish"
        return "neutral"

    def _determine_trend(self, data: pd.DataFrame, sma: pd.Series, ema: pd.Series) -> str:
        """Determine the current trend.

        Args:
            data: Price data
            sma: Simple moving average
            ema: Exponential moving average

        Returns:
            Trend description
        """
        if sma.empty or ema.empty:
            return "unknown"

        current_price = data["Close"].iloc[-1]
        sma_value = sma.iloc[-1]
        ema_value = ema.iloc[-1]

        if current_price > sma_value and current_price > ema_value:
            return "uptrend"
        elif current_price < sma_value and current_price < ema_value:
            return "downtrend"
        else:
            return "sideways"

    def _determine_momentum(self, rsi_value: float | None) -> str:
        """Determine momentum based on RSI.

        Args:
            rsi_value: RSI value

        Returns:
            Momentum description
        """
        if rsi_value is None:
            return "unknown"

        if rsi_value > 70:
            return "overbought"
        elif rsi_value < 30:
            return "oversold"
        elif rsi_value > 50:
            return "bullish"
        else:
            return "bearish"
