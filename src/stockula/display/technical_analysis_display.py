"""
Technical Analysis Display Service following SRP.
Single Responsibility: Displaying technical analysis results.
"""

from typing import Any, Dict

from rich.console import Console
from rich.table import Table

from ..utils import get_console


class TechnicalAnalysisDisplay:
    """Displays technical analysis results - Single Responsibility: TA Display."""

    def __init__(self, console: Console | None = None):
        """Initialize technical analysis display.

        Args:
            console: Rich console for output (optional)
        """
        self.console = get_console(console)

    def display_technical_analysis(self, results: Dict[str, Any]) -> None:
        """Display technical analysis results.

        Args:
            results: Technical analysis results dictionary
        """
        if not results or "indicators" not in results:
            self.console.print("[yellow]No technical analysis data to display[/yellow]")
            return

        indicators = results["indicators"]
        ticker = results.get("ticker", "Unknown")

        self.console.print(f"\n[bold cyan]Technical Analysis for {ticker}[/bold cyan]")

        # Create table for indicators
        table = Table(title="Technical Indicators", show_header=True, header_style="bold blue")
        table.add_column("Indicator", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        table.add_column("Signal", style="green")

        # Display indicators in organized manner
        self._add_moving_averages(table, indicators)
        self._add_momentum_indicators(table, indicators)
        self._add_volatility_indicators(table, indicators)
        self._add_volume_indicators(table, indicators)

        self.console.print(table)

        # Display any errors or warnings
        if results.get("error"):
            self.console.print(f"[red]Error: {results['error']}[/red]")

        if results.get("warnings"):
            for warning in results["warnings"]:
                self.console.print(f"[yellow]Warning: {warning}[/yellow]")

    def _add_moving_averages(self, table: Table, indicators: Dict[str, Any]) -> None:
        """Add moving average indicators to table.

        Args:
            table: Rich table to add to
            indicators: Indicators dictionary
        """
        # Simple Moving Averages
        for key, value in indicators.items():
            if key.startswith("SMA_"):
                period = key.split("_")[1]
                signal = self._determine_ma_signal(value, indicators.get("close", 0))
                table.add_row(f"SMA ({period})", f"{value:.2f}", signal)

        # Exponential Moving Averages
        for key, value in indicators.items():
            if key.startswith("EMA_"):
                period = key.split("_")[1]
                signal = self._determine_ma_signal(value, indicators.get("close", 0))
                table.add_row(f"EMA ({period})", f"{value:.2f}", signal)

    def _add_momentum_indicators(self, table: Table, indicators: Dict[str, Any]) -> None:
        """Add momentum indicators to table.

        Args:
            table: Rich table to add to
            indicators: Indicators dictionary
        """
        # RSI
        if "RSI" in indicators:
            rsi_value = indicators["RSI"]
            signal = self._determine_rsi_signal(rsi_value)
            table.add_row("RSI (14)", f"{rsi_value:.2f}", signal)

        # MACD
        if "MACD" in indicators:
            macd_data = indicators["MACD"]
            if isinstance(macd_data, dict):
                macd_line = macd_data.get("MACD", 0)
                signal_line = macd_data.get("Signal", 0)
                histogram = macd_data.get("Histogram", 0)
                signal = self._determine_macd_signal(macd_line, signal_line)

                table.add_row("MACD Line", f"{macd_line:.4f}", signal)
                table.add_row("MACD Signal", f"{signal_line:.4f}", "")
                table.add_row("MACD Histogram", f"{histogram:.4f}", "")

        # ADX
        if "ADX" in indicators:
            adx_value = indicators["ADX"]
            signal = self._determine_adx_signal(adx_value)
            table.add_row("ADX (14)", f"{adx_value:.2f}", signal)

    def _add_volatility_indicators(self, table: Table, indicators: Dict[str, Any]) -> None:
        """Add volatility indicators to table.

        Args:
            table: Rich table to add to
            indicators: Indicators dictionary
        """
        # Bollinger Bands
        if "BBands" in indicators:
            bbands_data = indicators["BBands"]
            if isinstance(bbands_data, dict):
                upper = bbands_data.get("upper", 0)
                middle = bbands_data.get("middle", 0)
                lower = bbands_data.get("lower", 0)
                current_price = indicators.get("close", 0)
                signal = self._determine_bbands_signal(current_price, upper, lower)

                table.add_row("BB Upper", f"{upper:.2f}", "")
                table.add_row("BB Middle", f"{middle:.2f}", "")
                table.add_row("BB Lower", f"{lower:.2f}", signal)

        # ATR
        if "ATR" in indicators:
            atr_value = indicators["ATR"]
            table.add_row("ATR (14)", f"{atr_value:.2f}", "Volatility")

    def _add_volume_indicators(self, table: Table, indicators: Dict[str, Any]) -> None:
        """Add volume indicators to table.

        Args:
            table: Rich table to add to
            indicators: Indicators dictionary
        """
        # OBV
        if "OBV" in indicators:
            obv_value = indicators["OBV"]
            table.add_row("OBV", f"{obv_value:,.0f}", "Volume Trend")

    def _determine_ma_signal(self, ma_value: float, current_price: float) -> str:
        """Determine moving average signal.

        Args:
            ma_value: Moving average value
            current_price: Current price

        Returns:
            Signal string
        """
        if current_price > ma_value:
            return "🟢 Bullish"
        elif current_price < ma_value:
            return "🔴 Bearish"
        else:
            return "⚪ Neutral"

    def _determine_rsi_signal(self, rsi_value: float) -> str:
        """Determine RSI signal.

        Args:
            rsi_value: RSI value

        Returns:
            Signal string
        """
        if rsi_value > 70:
            return "🔴 Overbought"
        elif rsi_value < 30:
            return "🟢 Oversold"
        else:
            return "⚪ Neutral"

    def _determine_macd_signal(self, macd_line: float, signal_line: float) -> str:
        """Determine MACD signal.

        Args:
            macd_line: MACD line value
            signal_line: Signal line value

        Returns:
            Signal string
        """
        if macd_line > signal_line:
            return "🟢 Bullish"
        elif macd_line < signal_line:
            return "🔴 Bearish"
        else:
            return "⚪ Neutral"

    def _determine_adx_signal(self, adx_value: float) -> str:
        """Determine ADX signal.

        Args:
            adx_value: ADX value

        Returns:
            Signal string
        """
        if adx_value > 25:
            return "🟢 Strong Trend"
        elif adx_value < 20:
            return "🔴 Weak Trend"
        else:
            return "⚪ Moderate"

    def _determine_bbands_signal(self, price: float, upper: float, lower: float) -> str:
        """Determine Bollinger Bands signal.

        Args:
            price: Current price
            upper: Upper band
            lower: Lower band

        Returns:
            Signal string
        """
        if price > upper:
            return "🔴 Overbought"
        elif price < lower:
            return "🟢 Oversold"
        else:
            return "⚪ Normal"
