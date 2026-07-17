from __future__ import annotations

from dataclasses import dataclass
from tkinter import Canvas, Listbox, Misc, Text, Tk
from tkinter import ttk


@dataclass(frozen=True)
class Palette:
    background: str
    panel: str
    raised: str
    header: str
    text: str
    muted: str
    field: str
    accent: str
    selected: str
    border: str
    warning: str
    danger: str
    success: str
    chart: str


PALETTES = {
    "dark": Palette(
        background="#0b1220",
        panel="#111827",
        raised="#172033",
        header="#07101e",
        text="#f8fafc",
        muted="#cbd5e1",
        field="#1f2937",
        accent="#38bdf8",
        selected="#164e63",
        border="#334155",
        warning="#f59e0b",
        danger="#ef4444",
        success="#22c55e",
        chart="#f8fafc",
    ),
    "light": Palette(
        background="#eef3f8",
        panel="#ffffff",
        raised="#f8fafc",
        header="#102a43",
        text="#172033",
        muted="#52606d",
        field="#ffffff",
        accent="#0369a1",
        selected="#dbeafe",
        border="#cbd5e1",
        warning="#b45309",
        danger="#b91c1c",
        success="#15803d",
        chart="#ffffff",
    ),
    "high contrast": Palette(
        background="#000000",
        panel="#000000",
        raised="#111111",
        header="#000000",
        text="#ffffff",
        muted="#ffffff",
        field="#000000",
        accent="#00ffff",
        selected="#003b46",
        border="#ffffff",
        warning="#ffff00",
        danger="#ff4d4d",
        success="#00ff66",
        chart="#ffffff",
    ),
}


class ThemeManager:
    """Applies one coherent theme to the shell and legacy Tk workspaces."""

    DENSITY_FONT = {"large text": 12, "comfortable": 10, "compact": 9}
    DENSITY_PAD = {"large text": 10, "comfortable": 7, "compact": 4}

    def __init__(self, root: Tk) -> None:
        self.root = root
        self.style = ttk.Style(root)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
        self.theme = "dark"
        self.density = "comfortable"

    @property
    def palette(self) -> Palette:
        return PALETTES[self.theme]

    def apply(self, theme: str, density: str) -> None:
        theme_key = theme.strip().lower()
        if theme_key == "system":
            theme_key = "dark"
        if theme_key not in PALETTES:
            theme_key = "dark"
        density_key = density.strip().lower()
        if density_key not in self.DENSITY_FONT:
            density_key = "comfortable"
        self.theme = theme_key
        self.density = density_key
        palette = self.palette
        font_size = self.DENSITY_FONT[density_key]
        padding = self.DENSITY_PAD[density_key]

        self.root.configure(background=palette.background)
        self.root.option_add("*Font", ("Segoe UI", font_size))

        common = {
            "background": palette.panel,
            "foreground": palette.text,
            "bordercolor": palette.border,
            "lightcolor": palette.border,
            "darkcolor": palette.border,
            "troughcolor": palette.background,
        }
        self.style.configure(".", **common)
        self.style.configure("TFrame", background=palette.panel)
        self.style.configure("Shell.TFrame", background=palette.background)
        self.style.configure("Header.TFrame", background=palette.header)
        self.style.configure("Sidebar.TFrame", background=palette.panel)
        self.style.configure("Card.TFrame", background=palette.raised, borderwidth=1, relief="solid")
        self.style.configure("TLabel", background=palette.panel, foreground=palette.text)
        self.style.configure("Header.TLabel", background=palette.header, foreground=palette.text)
        self.style.configure("Muted.TLabel", background=palette.panel, foreground=palette.muted)
        self.style.configure("CardTitle.TLabel", background=palette.raised, foreground=palette.text, font=("Segoe UI", font_size + 2, "bold"))
        self.style.configure("CardText.TLabel", background=palette.raised, foreground=palette.muted)
        self.style.configure("Status.TLabel", background=palette.raised, foreground=palette.text, padding=(padding + 3, padding))
        self.style.configure("TButton", background=palette.raised, foreground=palette.text, padding=(padding + 3, padding))
        self.style.map(
            "TButton",
            background=[("active", palette.selected), ("pressed", palette.accent)],
            foreground=[("disabled", palette.muted)],
        )
        self.style.configure("Accent.TButton", background=palette.accent, foreground="#07101e", font=("Segoe UI", font_size, "bold"))
        self.style.map("Accent.TButton", background=[("active", palette.success), ("pressed", palette.selected)])
        self.style.configure("Tool.TButton", padding=(padding, max(2, padding - 2)))
        self.style.configure("TLabelframe", background=palette.panel, foreground=palette.text, bordercolor=palette.border)
        self.style.configure("TLabelframe.Label", background=palette.panel, foreground=palette.text, font=("Segoe UI", font_size, "bold"))
        self.style.configure("TEntry", fieldbackground=palette.field, foreground=palette.text, insertcolor=palette.text)
        self.style.configure("TCombobox", fieldbackground=palette.field, foreground=palette.text, arrowcolor=palette.text)
        self.style.map("TCombobox", fieldbackground=[("readonly", palette.field)], foreground=[("readonly", palette.text)])
        self.style.configure("TSpinbox", fieldbackground=palette.field, foreground=palette.text, arrowcolor=palette.text)
        self.style.configure("TNotebook", background=palette.background, bordercolor=palette.border)
        self.style.configure("TNotebook.Tab", background=palette.raised, foreground=palette.muted, padding=(padding + 4, padding))
        self.style.map("TNotebook.Tab", background=[("selected", palette.selected)], foreground=[("selected", palette.text)])
        self.style.configure(
            "Treeview",
            background=palette.field,
            fieldbackground=palette.field,
            foreground=palette.text,
            rowheight=max(24, font_size * 2 + padding),
            bordercolor=palette.border,
        )
        self.style.map("Treeview", background=[("selected", palette.selected)], foreground=[("selected", palette.text)])
        self.style.configure("Treeview.Heading", background=palette.raised, foreground=palette.text, font=("Segoe UI", font_size, "bold"))
        self.style.configure("TSeparator", background=palette.border)

        # Legacy workspaces use these named styles. Re-map them after a workspace
        # is embedded so the selected global theme remains authoritative.
        for name in ("Launcher.TFrame", "App.TFrame"):
            self.style.configure(name, background=palette.background)
        for name in ("LauncherHeader.TFrame", "PDHeader.TFrame"):
            self.style.configure(name, background=palette.header)
        for name in ("LauncherTitle.TLabel", "PDTitle.TLabel", "HeaderTitle.TLabel"):
            self.style.configure(name, background=palette.header, foreground=palette.text)
        for name in ("LauncherSub.TLabel", "PDSub.TLabel", "HeaderSub.TLabel"):
            self.style.configure(name, background=palette.header, foreground=palette.muted)
        for name in ("LauncherStatus.TLabel", "PDStatus.TLabel"):
            self.style.configure(name, background=palette.raised, foreground=palette.text)
        for name in ("Workspace.TLabelframe",):
            self.style.configure(name, background=palette.raised, foreground=palette.text)
        for name in ("Workspace.TLabelframe.Label",):
            self.style.configure(name, background=palette.raised, foreground=palette.text)
        for name in ("WorkspaceText.TLabel", "SideText.TLabel", "CardSub.TLabel", "CardSubtitle.TLabel"):
            self.style.configure(name, background=palette.panel, foreground=palette.muted)
        for name in ("SideTitle.TLabel", "CardValue.TLabel", "Section.TLabel"):
            self.style.configure(name, background=palette.panel, foreground=palette.text)

        self._apply_tk_widgets(self.root)

    def _apply_tk_widgets(self, widget: Misc) -> None:
        palette = self.palette
        for child in widget.winfo_children():
            try:
                if isinstance(child, Text):
                    child.configure(
                        background=palette.field,
                        foreground=palette.text,
                        insertbackground=palette.text,
                        selectbackground=palette.selected,
                        selectforeground=palette.text,
                    )
                elif isinstance(child, Listbox):
                    child.configure(
                        background=palette.field,
                        foreground=palette.text,
                        selectbackground=palette.selected,
                        selectforeground=palette.text,
                        highlightbackground=palette.border,
                    )
                elif isinstance(child, Canvas):
                    role = getattr(child, "omega_role", "")
                    if role == "sidebar":
                        child.configure(background=palette.panel, highlightbackground=palette.border)
                    elif role == "controls":
                        child.configure(background=palette.panel, highlightbackground=palette.border)
                    else:
                        # Several legacy plots draw black text directly. A light
                        # chart surface preserves readability until those plots are
                        # migrated to the interactive Plotly renderer.
                        child.configure(background=palette.chart, highlightbackground=palette.border)
            except Exception:
                pass
            self._apply_tk_widgets(child)
