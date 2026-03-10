"""
Map visualization component: renders the tile-based world grid with actors.
"""

from nicegui import ui


def render_map_view():
    """
    Renders the ASCII map grid showing tiles, NPCs, player, and items.
    
    This view will be updated in real-time as the simulation ticks.
    Uses a monospace font to preserve grid alignment.
    """
    with ui.card().classes('w-full'):
        ui.label('🗺️ World Map').classes('text-xl font-bold')
        
        # Map container with monospace font for ASCII grid
        map_label = ui.label('').classes('font-mono text-sm whitespace-pre')
        
        # Placeholder ASCII map (will be replaced with real simulation state)
        demo_map = """
########################################
#                                      #
#  @  [SHOP]   G                       #
#                                      #
#            [SHOP]                    #
#                       G              #
#                                      #
########################################
        """.strip()
        
        map_label.set_text(demo_map)
        
        # Legend
        with ui.row():
            ui.label('Legend:').classes('font-bold')
            ui.label('@ = Player')
            ui.label('G = Guard')
            ui.label('[SHOP] = Shop')
            ui.label('# = Wall')
        
    return map_label  # Return reference for live updates
