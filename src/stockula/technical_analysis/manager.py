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
        """Analyze a single symbol with specified indicators.

        Args:
            symbol: Stock symbol to analyze
            config: Configuration object
            analysis_type: Type of analysis ('basic', 'momentum', 'trend', 'volatility', 'volume', 'comprehensive')
            custom_indicators: Custom list of indicators to calculate
            start_date: Optional start date for data
            end_date: Optional end date for data

        Returns:
            Dictionary with analysis results
        """
        try:
            # Fetch data
            # Convert date to string if needed
            start = start_date or config.data.start_date
            end = end_date or config.data.end_date
            if hasattr(start, "strftime"):
                start = start.strftime("%Y-%m-%d")  # type: ignore[union-attr]
            if hasattr(end, "strftime"):
                end = end.strftime("%Y-%m-%d")  # type: ignore[union-attr]
            data = self.data_fetcher.get_stock_data(symbol, start=start, end=end)

            if data.empty:
                return {"ticker": symbol, "error": "No data available"}

            # Create TechnicalIndicators instance
            ta = TechnicalIndicators(data)

            # Determine which indicators to calculate
            if custom_indicators:
                indicators = custom_indicators
            else:
                indicators = self.indicator_groups.get(analysis_type, self.indicator_groups["comprehensive"])

            # Calculate indicators based on configuration
            ta_config = config.technical_analysis
            results = {
                "ticker": symbol,
                "current_price": data["Close"].iloc[-1],
                "analysis_type": analysis_type,
                "indicators": {},
            }

            # Calculate each indicator
            for indicator in indicators:
                if indicator == "volume":
                    results["indicators"]["volume"] = {
                        "current": data["Volume"].iloc[-1],
                        "average": data["Volume"].mean(),
                        "ratio": data["Volume"].iloc[-1] / data["Volume"].mean(),
                    }
                elif hasattr(ta, indicator):
                    try:
                        # Get parameters from config or use defaults
                        params = self._get_indicator_params(indicator, ta_config)
                        indicator_func = getattr(ta, indicator)
                        result = indicator_func(**params)

                        # Format result based on type
                        if isinstance(result, pd.Series):
                            results["indicators"][indicator] = {
                                "current": result.iloc[-1] if not result.empty else None,
                                "values": result.to_dict() if len(result) <= 10 else None,
                            }
                        elif isinstance(result, pd.DataFrame):
                            results["indicators"][indicator] = {
                                "current": {col: result[col].iloc[-1] for col in result.columns if not result.empty},
                                "values": result.to_dict() if len(result) <= 10 else None,
                            }
                    except Exception as e:
                        self.logger.warning(f"Failed to calculate {indicator} for {symbol}: {str(e)}")
                        results["indicators"][indicator] = {"error": str(e)}

            # Add analysis summary
            results["summary"] = self._generate_analysis_summary(results["indicators"], data)

            return results

        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {str(e)}")
            return {"ticker": symbol, "error": str(e)}

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
                if hasattr(ta, indicator_name):
                    try:
                        indicator_func = getattr(ta, indicator_name)
                        result = indicator_func(**params)

                        if isinstance(result, pd.Series):
                            results["indicators"][indicator_name] = {
                                "current": result.iloc[-1] if not result.empty else None,
                                "params": params,
                            }
                        elif isinstance(result, pd.DataFrame):
                            results["indicators"][indicator_name] = {
                                "current": {col: result[col].iloc[-1] for col in result.columns if not result.empty},
                                "params": params,
                            }
                    except Exception as e:
                        results["indicators"][indicator_name] = {"error": str(e), "params": params}
                else:
                    results["indicators"][indicator_name] = {"error": f"Unknown indicator: {indicator_name}"}

            return results

        except Exception as e:
            self.logger.error(f"Error calculating custom indicators for {symbol}: {str(e)}")
            return {"ticker": symbol, "error": str(e)}

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

        # Check RSI for overbought/oversold
        if "rsi" in indicators and indicators["rsi"].get("current"):
            rsi_value = indicators["rsi"]["current"]
            if rsi_value > 70:
                summary["signals"].append("RSI Overbought")
            elif rsi_value < 30:
                summary["signals"].append("RSI Oversold")

        # Check MACD for crossovers
        if "macd" in indicators and indicators["macd"].get("current"):
            macd_current = indicators["macd"]["current"]
            if "MACD" in macd_current and "MACD_SIGNAL" in macd_current:
                if macd_current["MACD"] > macd_current["MACD_SIGNAL"]:
                    summary["signals"].append("MACD Bullish")
                else:
                    summary["signals"].append("MACD Bearish")

        # Check price vs moving averages
        current_price = data["Close"].iloc[-1]
        if "sma" in indicators and indicators["sma"].get("current"):
            sma_value = indicators["sma"]["current"]
            if current_price > sma_value:
                summary["signals"].append("Price above SMA")
            else:
                summary["signals"].append("Price below SMA")

        # Determine overall strength
        bullish_signals = sum(
            1 for signal in summary["signals"] if "Bullish" in signal or "above" in signal or "Oversold" in signal
        )
        bearish_signals = sum(
            1 for signal in summary["signals"] if "Bearish" in signal or "below" in signal or "Overbought" in signal
        )

        if bullish_signals > bearish_signals:
            summary["strength"] = "bullish"
        elif bearish_signals > bullish_signals:
            summary["strength"] = "bearish"

        return summary

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
