from tile2net.grid.grid import file
if False:
    from .ingrid import InGrid

# todo: we seem to have forgotten to commit ingrid.file and have to implement this once again
class File(
    file.File
):
    instance: InGrid

