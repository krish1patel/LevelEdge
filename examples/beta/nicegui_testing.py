from nicegui import ui

# --- Your backend class would be imported here ---
# from your_package import YourBackend
# backend = YourBackend()

# Track tabs: {name: {tab, panel, label_el}}
tab_counter = 0
tabs_map: dict = {}


def get_next_name() -> str:
    global tab_counter
    tab_counter += 1
    return f"Tab {tab_counter}"


def add_tab(label: str = None):
    label = label or get_next_name()

    with tab_bar:
        # Tab: label (double-click to rename) + close button, Chrome-style
        t = ui.tab(label).classes("browser-tab").style(
            "padding-right: 4px; min-width: 120px; max-width: 200px;"
        )
        with t:
            label_el = ui.label(label).classes("tab-label flex-grow cursor-pointer")
            label_el.on("dblclick", lambda tname=label: rename_tab(tname))
            (
                ui.button(icon="close", on_click=lambda tname=label: close_tab(tname))
                .props("flat round dense size=xs")
                .classes("tab-close")
                .style("opacity: 0.6; flex-shrink: 0;")
                .on("mousedown.stop", lambda: None)  # prevent tab switch on close click
            )

    with tab_panels:
        with ui.tab_panel(label) as panel:
            build_tab_content(label)

    tabs_map[label] = {"tab": t, "panel": panel, "label_el": label_el}
    tab_bar.set_value(label)  # switch to new tab


def close_tab(name: str):
    if len(tabs_map) <= 1:
        ui.notify("Can't close the last tab.", type="warning")
        return

    entry = tabs_map.pop(name)
    entry["tab"].delete()
    entry["panel"].delete()

    # Switch to last remaining tab
    remaining = list(tabs_map.keys())
    tab_bar.set_value(remaining[-1])


def rename_tab(old_name: str):
    async def apply_rename():
        new_name = input_field.value.strip()
        if not new_name or new_name == old_name:
            dialog.close()
            return
        if new_name in tabs_map:
            ui.notify("A tab with that name already exists.", type="negative")
            return

        entry = tabs_map.pop(old_name)
        entry["tab"]._props["name"] = new_name
        entry["tab"]._props["label"] = new_name
        entry["tab"].update()
        if "label_el" in entry:
            entry["label_el"].set_text(new_name)
        entry["panel"]._props["name"] = new_name
        entry["panel"].update()
        tabs_map[new_name] = entry
        tab_bar.set_value(new_name)
        dialog.close()

    with ui.dialog() as dialog, ui.card():
        ui.label("Rename Tab").classes("text-lg font-semibold mb-2")
        input_field = ui.input(value=old_name, placeholder="New tab name").classes("w-full")
        with ui.row().classes("justify-end mt-2 gap-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Rename", on_click=apply_rename).props("color=primary")
        input_field.on("keydown.enter", apply_rename)

    dialog.open()
    dialog.move(tab_panels)  # keep it scoped to the app


def build_tab_content(name: str):
    """Put your per-tab UI here, wiring up your backend class as needed."""
    with ui.column().classes("gap-4 p-4 w-full"):
        with ui.row().classes("items-center gap-2"):
            ui.label(f"Content for: {name}").classes("text-xl font-semibold")
            ui.button(
                "Rename Tab",
                icon="edit",
                on_click=lambda tname=name: rename_tab(tname)
            ).props("flat size=sm")

        ui.separator()

        # Example: wire your backend here
        # result = backend.get_data_for(name)
        ui.label("Your content goes here.").classes("text-gray-500")


# ── App Layout ──────────────────────────────────────────────────────────────

ui.add_head_html("""
<style>
  /* Chrome-like dynamic tab bar: scrollable when many tabs */
  .q-tabs__content { gap: 2px; overflow-x: auto; flex-wrap: nowrap; }
  .q-tab.browser-tab {
    border-radius: 8px 8px 0 0 !important;
    display: flex !important;
    align-items: center;
    gap: 4px;
  }
  .q-tab.browser-tab .tab-label { pointer-events: auto; }
  .q-tab--active { background: white !important; }
  .q-tab:not(.q-tab--active) { background: #e5e7eb; }
  .q-tab:not(.q-tab--active):hover { background: #d1d5db; }
  .q-tab .tab-close:hover { opacity: 1 !important; }
</style>
""")

with ui.column().classes("w-full h-screen gap-0"):
    # Top bar: tabs + add button
    with ui.row().classes("w-full items-center bg-gray-100 border-b px-2 pt-2 gap-0"):
        tab_bar = ui.tabs().props("align=left dense").classes("flex-1")

        # ➕ Add Tab button at end of tab row
        ui.button(icon="add", on_click=add_tab).props("flat round dense").style(
            "margin-bottom: 4px; margin-left: 4px;"
        )

    # Tab content panels
    tab_panels = ui.tab_panels(tab_bar).classes("w-full flex-1")

# Start with one tab (Chrome-style)
add_tab("New Tab")

# Ctrl+W / Cmd+W to close current tab
def handle_key(e):
    if e.args.get("key") == "w" and (e.args.get("ctrlKey") or e.args.get("metaKey")):
        current = tab_bar.value
        if current and current in tabs_map:
            close_tab(current)

ui.keyboard(on_key=handle_key)

ui.run(title="My App", dark=False)
