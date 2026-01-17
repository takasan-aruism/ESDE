"""
ESDE Phase 8-9: Semantic Monitor
================================

TUIダッシュボードでESDEの「生命の鼓動」を可視化する。

Display Layout:
- Header: Version, Uptime, Ledger Seq
- Left Panel: Live Feed (Input, Target, Strategy, Result)
- Center Panel: Top Rigid/Volatile Rankings
- Right Panel: System Stats

Dependencies: rich (pip install rich)
"""

import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Rich imports with fallback
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Dummy classes for type hints
    Console = None
    Table = None
    Panel = None
    Layout = None
    Live = None
    Text = None
    box = None


# ==========================================
# Monitor State
# ==========================================
@dataclass
class MonitorState:
    """モニター状態"""
    # Timing
    start_time: float = field(default_factory=time.time)
    total_steps: int = 0
    
    # Last observation
    last_input: str = ""
    last_target_atom: Optional[str] = None
    last_rigidity: Optional[float] = None
    last_observations: int = 0
    last_strategy_mode: str = "neutral"
    last_temperature: float = 0.1
    last_formula: Optional[str] = None
    last_direction: str = "=>"
    last_alert: bool = False
    
    # Aggregates
    total_alerts: int = 0
    ledger_seq: int = -1
    index_size: int = 0
    
    # Direction counts
    direction_creative: int = 0
    direction_destructive: int = 0
    direction_neutral: int = 0
    
    # Rankings (atom_id -> (R, N))
    atom_stats: Dict[str, tuple] = field(default_factory=dict)
    
    def uptime(self) -> str:
        """稼働時間を文字列で返す"""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def direction_balance(self) -> Dict[str, float]:
        """方向性バランスを計算"""
        total = self.direction_creative + self.direction_destructive + self.direction_neutral
        if total == 0:
            return {"=>+": 0.0, "-|>": 0.0, "=>": 0.0}
        return {
            "=>+": self.direction_creative / total,
            "-|>": self.direction_destructive / total,
            "=>": self.direction_neutral / total,
        }
    
    def top_rigid(self, limit: int = 5) -> List[tuple]:
        """硬直度上位を返す"""
        items = [(aid, r, n) for aid, (r, n) in self.atom_stats.items() if r is not None]
        return sorted(items, key=lambda x: -x[1])[:limit]
    
    def top_volatile(self, limit: int = 5) -> List[tuple]:
        """流動度上位（低Rigidity）を返す"""
        items = [(aid, r, n) for aid, (r, n) in self.atom_stats.items() if r is not None and n >= 2]
        return sorted(items, key=lambda x: x[1])[:limit]


# ==========================================
# Semantic Monitor
# ==========================================
class SemanticMonitor:
    """
    ESDE Semantic Monitor - TUIダッシュボード
    
    Usage:
        monitor = SemanticMonitor()
        monitor.update(pipeline_result)
        monitor.render()  # または Live context内で使用
    """
    
    VERSION = "8.9.0"
    
    def __init__(self):
        self.state = MonitorState()
        self.console = Console() if RICH_AVAILABLE else None
    
    def update(self, result: Any) -> None:
        """
        PipelineResultからモニター状態を更新する。
        
        Args:
            result: PipelineResult (from pipeline.core_pipeline)
        """
        self.state.total_steps += 1
        
        # Last observation
        self.state.last_input = getattr(result, 'input_text', '')[:50]
        self.state.last_target_atom = getattr(result, 'target_atom', None)
        self.state.last_rigidity = getattr(result, 'rigidity', None)
        
        strategy = getattr(result, 'strategy', None)
        if strategy:
            self.state.last_strategy_mode = strategy.mode.value if hasattr(strategy.mode, 'value') else str(strategy.mode)
            self.state.last_temperature = strategy.temperature
        
        molecule = getattr(result, 'molecule', None)
        if molecule:
            self.state.last_formula = molecule.get('formula', '')[:30]
        
        ledger_entry = getattr(result, 'ledger_entry', None)
        if ledger_entry:
            self.state.last_direction = ledger_entry.get('direction', '=>')
            self.state.ledger_seq = ledger_entry.get('seq', -1)
            
            # Direction counts
            direction = ledger_entry.get('direction', '=>')
            if direction == '=>+':
                self.state.direction_creative += 1
            elif direction == '-|>':
                self.state.direction_destructive += 1
            else:
                self.state.direction_neutral += 1
        
        # Alert
        alert = getattr(result, 'alert', None)
        self.state.last_alert = alert is not None
        if alert:
            self.state.total_alerts += 1
        
        # Update atom stats from index if available
        # (この部分はIndexへの参照が必要な場合に拡張)
    
    def update_from_index(self, index: Any) -> None:
        """
        SemanticIndexからatom_statsを更新する。
        """
        if hasattr(index, 'atom_stats'):
            for atom_id, stats in index.atom_stats.items():
                if hasattr(stats, 'count_total') and hasattr(stats, 'formula_counts'):
                    n = stats.count_total
                    if n > 0 and stats.formula_counts:
                        r = max(stats.formula_counts.values()) / n
                    else:
                        r = None
                    self.state.atom_stats[atom_id] = (r, n)
        
        if hasattr(index, 'total_events'):
            self.state.index_size = len(getattr(index, 'atom_stats', {}))
    
    def render(self) -> None:
        """コンソールに現在の状態を描画する。"""
        if not RICH_AVAILABLE:
            self._render_plain()
            return
        
        self.console.clear()
        self.console.print(self._build_layout())
    
    def get_renderable(self):
        """Live用のrenderableオブジェクトを返す。"""
        if not RICH_AVAILABLE:
            return str(self)
        return self._build_layout()
    
    def _build_layout(self):
        """Richレイアウトを構築する。"""
        layout = Layout()
        
        # Header + Body
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body")
        )
        
        # Body: Left, Center, Right
        layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="center", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # Populate
        layout["header"].update(self._build_header())
        layout["left"].update(self._build_live_feed())
        layout["center"].update(self._build_rankings())
        layout["right"].update(self._build_stats())
        
        return layout
    
    def _build_header(self):
        """ヘッダーパネルを構築"""
        text = Text()
        text.append("ESDE ", style="bold cyan")
        text.append(f"v{self.VERSION}", style="dim")
        text.append(" │ ", style="dim")
        text.append(f"Uptime: {self.state.uptime()}", style="green")
        text.append(" │ ", style="dim")
        text.append(f"Steps: {self.state.total_steps}", style="yellow")
        text.append(" │ ", style="dim")
        text.append(f"Ledger Seq: {self.state.ledger_seq}", style="blue")
        text.append(" │ ", style="dim")
        alert_style = "bold red" if self.state.total_alerts > 0 else "dim"
        text.append(f"Alerts: {self.state.total_alerts}", style=alert_style)
        
        return Panel(text, box=box.DOUBLE)
    
    def _build_live_feed(self):
        """Live Feedパネルを構築"""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="cyan", width=12)
        table.add_column("Value")
        
        table.add_row("Input:", self.state.last_input or "-")
        table.add_row("Target:", self.state.last_target_atom or "-")
        
        # Rigidity with color coding
        if self.state.last_rigidity is not None:
            r = self.state.last_rigidity
            if r > 0.9:
                r_style = "bold red"
            elif r < 0.3:
                r_style = "bold blue"
            else:
                r_style = "green"
            r_text = Text(f"{r:.3f}", style=r_style)
        else:
            r_text = Text("-", style="dim")
        table.add_row("Rigidity:", r_text)
        
        # Strategy
        mode = self.state.last_strategy_mode
        if mode == "disruptive":
            mode_style = "bold red"
        elif mode == "stabilizing":
            mode_style = "bold blue"
        else:
            mode_style = "green"
        table.add_row("Strategy:", Text(f"{mode} (T={self.state.last_temperature})", style=mode_style))
        
        # Formula
        table.add_row("Formula:", self.state.last_formula or "-")
        
        # Direction
        table.add_row("Direction:", self.state.last_direction)
        
        # Alert indicator
        if self.state.last_alert:
            table.add_row("", Text("⚠ ALERT FIRED", style="bold red blink"))
        
        return Panel(table, title="[bold]Live Feed[/bold]", border_style="cyan")
    
    def _build_rankings(self):
        """Rankingsパネルを構築"""
        # Top Rigid
        rigid_table = Table(title="Top Rigid (R→1.0)", box=box.SIMPLE)
        rigid_table.add_column("Atom", style="cyan")
        rigid_table.add_column("R", justify="right")
        rigid_table.add_column("N", justify="right")
        
        for atom_id, r, n in self.state.top_rigid(5):
            rigid_table.add_row(atom_id[:20], f"{r:.3f}", str(n))
        
        if not self.state.top_rigid(5):
            rigid_table.add_row("-", "-", "-")
        
        # Top Volatile
        volatile_table = Table(title="Top Volatile (R→0.0)", box=box.SIMPLE)
        volatile_table.add_column("Atom", style="blue")
        volatile_table.add_column("R", justify="right")
        volatile_table.add_column("N", justify="right")
        
        for atom_id, r, n in self.state.top_volatile(5):
            volatile_table.add_row(atom_id[:20], f"{r:.3f}", str(n))
        
        if not self.state.top_volatile(5):
            volatile_table.add_row("-", "-", "-")
        
        from rich.columns import Columns
        return Panel(
            Columns([rigid_table, volatile_table]),
            title="[bold]Rankings[/bold]",
            border_style="yellow"
        )
    
    def _build_stats(self):
        """System Statsパネルを構築"""
        table = Table(show_header=False, box=None)
        table.add_column("Stat", style="dim")
        table.add_column("Value", justify="right")
        
        table.add_row("Index Size:", str(self.state.index_size))
        table.add_row("", "")
        
        # Direction Balance
        balance = self.state.direction_balance()
        table.add_row("[Direction Balance]", "")
        table.add_row("  =>+ (Creative):", f"{balance['=>+']:.1%}")
        table.add_row("  -|> (Destructive):", f"{balance['-|>']:.1%}")
        table.add_row("  =>  (Neutral):", f"{balance['=>']:.1%}")
        
        return Panel(table, title="[bold]Stats[/bold]", border_style="green")
    
    def _render_plain(self) -> None:
        """rich非依存のプレーンテキスト出力"""
        print("=" * 60)
        print(f"ESDE v{self.VERSION} | Uptime: {self.state.uptime()} | Steps: {self.state.total_steps}")
        print(f"Ledger Seq: {self.state.ledger_seq} | Alerts: {self.state.total_alerts}")
        print("-" * 60)
        print(f"Input: {self.state.last_input}")
        print(f"Target: {self.state.last_target_atom} (R={self.state.last_rigidity})")
        print(f"Strategy: {self.state.last_strategy_mode} (T={self.state.last_temperature})")
        print(f"Formula: {self.state.last_formula}")
        print(f"Direction: {self.state.last_direction}")
        if self.state.last_alert:
            print("⚠ ALERT FIRED")
        print("=" * 60)
    
    def __str__(self) -> str:
        return f"SemanticMonitor(steps={self.state.total_steps}, alerts={self.state.total_alerts})"


# ==========================================
# Test
# ==========================================
if __name__ == "__main__":
    print(f"Rich available: {RICH_AVAILABLE}")
    
    monitor = SemanticMonitor()
    
    # Mock result
    class MockResult:
        input_text = "I love you"
        target_atom = "EMO.love"
        rigidity = 0.75
        strategy = type('Strategy', (), {'mode': type('Mode', (), {'value': 'neutral'})(), 'temperature': 0.1})()
        molecule = {"formula": "aa_1 × aa_2"}
        ledger_entry = {"seq": 42, "direction": "=>+"}
        alert = None
    
    monitor.update(MockResult())
    monitor.render()
