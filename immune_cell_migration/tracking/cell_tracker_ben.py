import clickpoints
from pathlib import Path
from joblib import Parallel, delayed
from .tracking_functions import run_tracking


def _track_one(path, database_name):
    print(database_name)
    print("Processing data:", database_name)
    db = clickpoints.DataFile(str(database_name))
    Frames = db.getImages(layer=1).count()
    db.setMaskType(name='NK', color='#0000ff', index=1)
    run_tracking(path, db, Frames, start_frame=1)
    try:
        db.db.close()
    except Exception:
        pass


def track_cells(path_list, n_jobs=1):
    """Track cells in every position database.

    ``n_jobs`` > 1 tracks that many databases in parallel (positions are
    independent; the tracker itself is single-process numpy). Default 1 keeps the
    original serial behavior.
    """
    tasks = [(path, str(database_name))
             for path, _ in path_list
             for database_name in Path(path).glob("*-*_pos*.cdb")]

    if n_jobs and n_jobs > 1 and len(tasks) > 1:
        Parallel(n_jobs=min(int(n_jobs), len(tasks)))(
            delayed(_track_one)(path, db) for path, db in tasks
        )
    else:
        for path, db in tasks:
            _track_one(path, db)
    print("--------all done------")
