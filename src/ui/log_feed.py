"""
Live action log feed: shows real-time events and NPC actions.
"""

from nicegui import ui


def render_log_feed():
    """
    Renders a scrollable, auto-updating log of simulation events.
    
    Shows actions like:
    - NPC movements
    - Transactions
    - Guard patrols
    - Player actions
    - System events
    """
    with ui.card().classes('w-full h-96'):
        ui.label('📜 Live Action Log').classes('text-xl font-bold')
        
        # Log container with scroll
        log_container = ui.column().classes('w-full h-80 overflow-y-auto')
        
        with log_container:
            # Placeholder logs (will be replaced with real simulation events)
            ui.label('[00:01] Simulation started').classes('text-sm')
            ui.label('[00:02] Guard #1 patrolling to (5, 3)').classes('text-sm text-blue-600')
            ui.label('[00:03] Shopkeeper #1 restocked 10 potions').classes('text-sm text-green-600')
            ui.label('[00:04] Player spawned at (2, 2)').classes('text-sm text-orange-600')
            ui.label('[00:05] Guard #2 detected player at (3, 2)').classes('text-sm text-red-600')
        
        # Filter controls
        with ui.row():
            ui.checkbox('Show Guards', value=True)
            ui.checkbox('Show Shops', value=True)
            ui.checkbox('Show Player', value=True)
            ui.checkbox('Show System', value=True)
        
    return log_container  # Return reference for appending new logs
