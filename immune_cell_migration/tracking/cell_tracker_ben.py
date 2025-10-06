import clickpoints
from pathlib import Path
from .tracking_functions import run_tracking


def track_cells(path_list):

    for path, _ in path_list:
        # iterate over all databases in all subfolders
        for database_name in Path(path).glob("*-*_pos*.cdb"):  # "**/*-*_pos*.cdb"
            print(database_name)
            print("Processing data:", database_name)
            # load database
            db = clickpoints.DataFile(str(database_name))
            Frames = db.getImages(layer=1).count()

            db.setMaskType(name='NK', color='#0000ff', index=1)

            run_tracking(path, db, Frames, start_frame=1)

            db.db.close()
        print("--------all done------")
