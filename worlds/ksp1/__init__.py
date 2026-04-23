from .world import KSP1World as KSP1World

from worlds.LauncherComponents import Component, Type, components, launch


def launch_tracker(*args):
    from .tracker_client import launch as tracker_main
    launch(tracker_main, name="KSP1TrackerClient", args=args)


components.append(Component(
    "KSP1 Tracker",
    func=launch_tracker,
    cli=True,
    component_type=Type.CLIENT,
    game_name="Kerbal Space Program 1",
    supports_uri=True,
    description="Track in-logic locations and inspect rocket designs.",
))
