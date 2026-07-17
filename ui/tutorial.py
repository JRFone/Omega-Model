from __future__ import annotations

from dataclasses import dataclass
from tkinter import Frame, Label, LEFT, RIGHT, X, StringVar
from tkinter import ttk
from typing import Callable, Protocol


@dataclass(frozen=True)
class TutorialStep:
    title: str
    body: str
    action: str
    caution: str = ""
    requires_click: bool = True


@dataclass(frozen=True)
class TutorialTarget:
    """A live control that the learner must use before the tutorial advances."""

    widget: object
    instruction: str
    event: str = "<ButtonRelease-1>"
    bounds: Callable[[], tuple[int, int, int, int]] | None = None


class TutorialHost(Protocol):
    root: object

    def perform_tutorial_action(self, action: str, finished: Callable[[bool, str], None]) -> None: ...

    def prepare_tutorial_target(self, action: str) -> TutorialTarget | None: ...

    def verify_tutorial_action(self, action: str, finished: Callable[[bool, str], None]) -> None: ...


FIRST_MODEL_STEPS = (
    TutorialStep(
        "Welcome to Omega",
        "This guide uses the live Omega interface. Start by clicking the highlighted Home control. The guide will wait for the correct click before continuing.",
        "home",
        "A successful software demonstration is not proof that a real stock assessment is scientifically valid.",
    ),
    TutorialStep(
        "Choose a dataset",
        "Click the highlighted Dataset Library control. Original data stay unchanged; analyses use the active dataset and working results are stored separately.",
        "datasets",
    ),
    TutorialStep(
        "Load the beginner dataset",
        "Click the highlighted Load beginner dataset button. It contains catch, an abundance index, direct biomass observations, sector catches, recruitment multipliers, and composition examples.",
        "load_beginner",
    ),
    TutorialStep(
        "Open Integrated Assessment",
        "Click the highlighted Integrated Assessment control. The selected dataset is passed into the real age-structured assessment workspace in this window.",
        "integrated",
    ),
    TutorialStep(
        "Use a quick teaching fit",
        "This setup step has no single workspace control. Click Apply teaching setup in the guide to use a short fit with fixed biology. Formal work needs stronger convergence settings and repeated starts.",
        "configure_quick_fit",
        requires_click=False,
    ),
    TutorialStep(
        "Fit the model",
        "Click the highlighted Fit integrated model button. Omega runs the real fitting function and the guide waits for completion before advancing.",
        "run_fit",
        "Do not interpret a fitted curve until convergence, residuals, sensitivity, and data conflict have been checked.",
    ),
    TutorialStep(
        "Inspect biomass and fishing mortality",
        "Click the highlighted Biomass and F tab. Depletion is biomass relative to a modelled reference level, not a direct count of every fish.",
        "show_biomass",
    ),
    TutorialStep(
        "Inspect fit diagnostics",
        "Click the highlighted Fit Diagnostics tab. Diagnostics show how the observations were fitted and whether numerical warnings were recorded.",
        "show_diagnostics",
    ),
    TutorialStep(
        "Explore priority diagnostics",
        "Click Priority Diagnostics. Likelihood profiles, ASPM, and interval coverage test different weaknesses; no single diagnostic establishes that a model is correct.",
        "priority",
    ),
    TutorialStep(
        "Explore management strategy evaluation",
        "Click Biomass & MSE to inspect how management rules are tested against simulated operating truths and sources of error.",
        "mse",
        "The operating truth is simulated. It is not the unknown true biomass of a real stock.",
    ),
    TutorialStep(
        "Tutorial complete",
        "Click the highlighted Home control to finish. You can then revisit any workspace or use Detach when a second window is genuinely useful.",
        "home",
    ),
)


class TutorialController:
    """In-window automatic demonstration and required-click guided practice."""

    SPEED_MS = {"slow": 5000, "normal": 2800, "fast": 1200}

    def __init__(self, host: TutorialHost, parent) -> None:
        self.host = host
        self.parent = parent
        self.steps = FIRST_MODEL_STEPS
        self.index = 0
        self.automatic = False
        self.paused = True
        self.after_id: str | None = None
        self.spotlight_after_id: str | None = None
        self.target: TutorialTarget | None = None
        self.target_bind_id: str | None = None
        self.speed = StringVar(value="Normal")
        self.step_text = StringVar()
        self.title_text = StringVar()
        self.body_text = StringVar()
        self.caution_text = StringVar()
        self.status_text = StringVar(value="Ready")
        self._spotlight_parts: list[object] = []
        self.panel = self._build_panel(parent)
        self.hide()

    def _build_panel(self, parent):
        panel = ttk.Frame(parent, padding=14, style="Card.TFrame", relief="solid", borderwidth=2)
        top = ttk.Frame(panel, style="Card.TFrame")
        top.pack(fill=X)
        ttk.Label(top, textvariable=self.step_text, style="CardText.TLabel").pack(side=LEFT)
        ttk.Button(top, text="×", width=3, command=self.exit, style="Tool.TButton").pack(side=RIGHT)
        ttk.Label(panel, textvariable=self.title_text, style="CardTitle.TLabel", wraplength=450, justify="left").pack(anchor="w", pady=(8, 5))
        ttk.Label(panel, textvariable=self.body_text, style="CardText.TLabel", wraplength=450, justify="left").pack(anchor="w")
        ttk.Label(panel, textvariable=self.caution_text, style="CardText.TLabel", wraplength=450, justify="left").pack(anchor="w", pady=(8, 0))
        controls = ttk.Frame(panel, style="Card.TFrame")
        controls.pack(fill=X, pady=(12, 0))
        self.previous_button = ttk.Button(controls, text="Previous", command=self.previous)
        self.previous_button.pack(side=LEFT)
        self.pause_button = ttk.Button(controls, text="Pause", command=self.pause)
        self.pause_button.pack(side=LEFT, padx=4)
        self.resume_button = ttk.Button(controls, text="Watch for me", command=self.resume)
        self.resume_button.pack(side=LEFT)
        self.next_button = ttk.Button(controls, text="Next", command=self.next)
        self.next_button.pack(side=LEFT, padx=4)
        self.do_button = ttk.Button(controls, text="Do it for me", command=self.do_for_me)
        self.do_button.pack(side=LEFT)
        ttk.Combobox(controls, textvariable=self.speed, values=("Slow", "Normal", "Fast"), state="readonly", width=8).pack(side=RIGHT)
        ttk.Label(panel, textvariable=self.status_text, style="CardText.TLabel", wraplength=450).pack(anchor="w", pady=(8, 0))
        return panel

    def show(self) -> None:
        self.panel.place(relx=1.0, rely=1.0, x=-28, y=-54, anchor="se", width=520)
        self.panel.lift()
        self._render()

    def hide(self) -> None:
        self.panel.place_forget()

    def start(self, automatic: bool = True) -> None:
        self.cancel_pending()
        self._clear_target()
        self.index = 0
        self.automatic = automatic
        self.paused = not automatic
        self.status_text.set("Automatic demonstration running." if automatic else "Guided practice: click the highlighted control.")
        self.show()
        self._render()
        if automatic:
            self._schedule_current()
        else:
            self.parent.after(120, self._guide_current)

    def _render(self) -> None:
        step = self.steps[self.index]
        self.step_text.set(f"Step {self.index + 1} of {len(self.steps)}")
        self.title_text.set(step.title)
        self.body_text.set(step.body)
        self.caution_text.set(f"Important: {step.caution}" if step.caution else "")
        if self.automatic:
            self.next_button.configure(text="Next", state="normal")
        elif step.requires_click:
            self.next_button.configure(text="Click highlighted control", state="disabled")
        else:
            self.next_button.configure(text="Apply teaching setup", state="normal")

    def _schedule_current(self) -> None:
        self.cancel_pending()
        delay = self.SPEED_MS.get(self.speed.get().lower(), 2800)
        self.after_id = self.parent.after(delay, self._run_current_action)

    def _run_current_action(self) -> None:
        self.after_id = None
        if self.paused:
            return
        step = self.steps[self.index]
        self.status_text.set(f"Working: {step.title}")
        self.host.perform_tutorial_action(step.action, self._action_finished)

    def _action_finished(self, success: bool, detail: str) -> None:
        self.status_text.set(detail)
        if not success:
            self.paused = True
            return
        if self.automatic and not self.paused and self.index < len(self.steps) - 1:
            self.index += 1
            self._render()
            self._schedule_current()
        elif self.index == len(self.steps) - 1:
            self.paused = True
            self.status_text.set("Tutorial complete. You can take control of Omega now.")

    def _guide_current(self) -> None:
        self._clear_target()
        if self.automatic:
            return
        step = self.steps[self.index]
        self._render()
        if not step.requires_click:
            self.status_text.set("Click Apply teaching setup in this guide to continue.")
            self.panel.lift()
            return
        try:
            target = self.host.prepare_tutorial_target(step.action)
        except Exception as exc:
            self.status_text.set(f"The guide could not find this control: {exc}")
            return
        if target is None:
            self.status_text.set("This control is not available yet. Use Do it for me or go back one step.")
            return
        self.target = target
        try:
            self.target_bind_id = target.widget.bind(target.event, self._target_clicked, add="+")
        except Exception as exc:
            self.status_text.set(f"The highlighted control could not be activated: {exc}")
            self.target = None
            return
        self.status_text.set(target.instruction)
        self._draw_spotlight()

    def _target_clicked(self, _event=None) -> None:
        if self.automatic or self.target is None:
            return
        action = self.steps[self.index].action
        self.status_text.set("Checking that the highlighted action completed…")
        self.parent.after(160, lambda: self.host.verify_tutorial_action(action, self._guided_action_finished))

    def _guided_action_finished(self, success: bool, detail: str) -> None:
        self.status_text.set(detail)
        if not success:
            self._draw_spotlight()
            return
        self._clear_target()
        if self.index < len(self.steps) - 1:
            self.index += 1
            self._render()
            self.parent.after(180, self._guide_current)
        else:
            self.paused = True
            self.status_text.set("Tutorial complete. You used every highlighted control yourself.")

    def _draw_spotlight(self) -> None:
        self._destroy_spotlight_parts()
        target = self.target
        if target is None:
            return
        try:
            self.parent.update_idletasks()
            if target.bounds is not None:
                screen_x, screen_y, width, height = target.bounds()
            else:
                screen_x = target.widget.winfo_rootx()
                screen_y = target.widget.winfo_rooty()
                width = target.widget.winfo_width()
                height = target.widget.winfo_height()
            root_x = self.parent.winfo_rootx()
            root_y = self.parent.winfo_rooty()
            x = screen_x - root_x - 5
            y = screen_y - root_y - 5
            width = max(int(width) + 10, 24)
            height = max(int(height) + 10, 24)
        except Exception:
            return
        colour = "#facc15"
        thickness = 5
        parts = [
            Frame(self.parent, background=colour, highlightthickness=0),
            Frame(self.parent, background=colour, highlightthickness=0),
            Frame(self.parent, background=colour, highlightthickness=0),
            Frame(self.parent, background=colour, highlightthickness=0),
        ]
        parts[0].place(x=x, y=y, width=width, height=thickness)
        parts[1].place(x=x, y=y + height - thickness, width=width, height=thickness)
        parts[2].place(x=x, y=y, width=thickness, height=height)
        parts[3].place(x=x + width - thickness, y=y, width=thickness, height=height)
        pointer = Label(
            self.parent,
            text=f"CLICK HERE  •  {target.instruction}",
            background="#facc15",
            foreground="#111827",
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=5,
        )
        pointer.update_idletasks()
        pointer_y = y - 38 if y >= 45 else y + height + 8
        pointer.place(x=max(8, x), y=pointer_y)
        self._spotlight_parts = [*parts, pointer]
        for part in self._spotlight_parts:
            part.lift()
        self.panel.lift()

    def pause(self) -> None:
        self.paused = True
        self.cancel_pending()
        if self.automatic:
            self.status_text.set("Paused. The current workspace remains available.")
        else:
            self.status_text.set("Guided practice is waiting for the highlighted click.")

    def resume(self) -> None:
        self._clear_target()
        self.automatic = True
        self.paused = False
        self._render()
        self.status_text.set("Automatic demonstration resumed.")
        self._schedule_current()

    def previous(self) -> None:
        self.cancel_pending()
        self._clear_target()
        self.automatic = False
        self.paused = True
        if self.index > 0:
            self.index -= 1
        self._render()
        self.parent.after(120, self._guide_current)

    def next(self) -> None:
        step = self.steps[self.index]
        if not self.automatic and step.requires_click:
            self.status_text.set("Use the highlighted control. The guide advances only after the correct click.")
            return
        self._run_step_for_user()

    def do_for_me(self) -> None:
        self._run_step_for_user()

    def _run_step_for_user(self) -> None:
        self.cancel_pending()
        self._clear_target()
        self.automatic = False
        self.paused = False
        step = self.steps[self.index]
        self.status_text.set(f"Working: {step.title}")

        def completed(success: bool, detail: str) -> None:
            self.status_text.set(detail)
            self.paused = True
            if success and self.index < len(self.steps) - 1:
                self.index += 1
                self._render()
                self.parent.after(180, self._guide_current)
            elif success:
                self.status_text.set("Tutorial complete. You can take control of Omega now.")

        self.host.perform_tutorial_action(step.action, completed)

    def exit(self) -> None:
        self.pause()
        self._clear_target()
        self.hide()

    def _clear_target(self) -> None:
        if self.target is not None and self.target_bind_id:
            try:
                self.target.widget.unbind(self.target.event, self.target_bind_id)
            except Exception:
                pass
        self.target = None
        self.target_bind_id = None
        self._destroy_spotlight_parts()

    def _destroy_spotlight_parts(self) -> None:
        for part in self._spotlight_parts:
            try:
                part.destroy()
            except Exception:
                pass
        self._spotlight_parts = []
        if self.spotlight_after_id is not None:
            try:
                self.parent.after_cancel(self.spotlight_after_id)
            except Exception:
                pass
            self.spotlight_after_id = None

    def cancel_pending(self) -> None:
        if self.after_id is not None:
            try:
                self.parent.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
