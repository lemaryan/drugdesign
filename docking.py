"""Optional DOCKSTRING integration.

Connect this adapter to the DOCKSTRING version installed on Savio after testing
a single molecule manually. The main agent never pretends a docking run occurred
when this function is not configured.
"""


def dock_molecule(smiles):
    """Return real docking scores, e.g. {'docking_jak2': -8.1, 'docking_lck': -6.2}."""
    raise NotImplementedError(
        "Connect this function to the installed DOCKSTRING API before enabling docking."
    )
