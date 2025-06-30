import os
from joblib import load


def load_one(name):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, '..', 'models', 'Basic')

    model_path = os.path.join(models_dir, f'{name}_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Missing model: {model_path}")
    model, r2 = load(model_path)
    model.r2 = r2
    return model


def load_all():
    """
    Loads all basic models from the models/Basic directory relative to this script file.
    Returns all models in fixed order as separate variables.
    """

    pts_pg_model = load_one('pts_pg')
    ast_pg_model = load_one('ast_pg')
    blk_pg_model = load_one('blk_pg')
    reb_pg_model = load_one('reb_pg')
    gp_model = load_one('gp')
    gs_model = load_one('gs')
    fga_pg_model = load_one('fga_pg')
    height_model = load_one('height')
    fg3a_pg_model = load_one('fg3a_pg')
    fta_pg_model = load_one('fta_pg')
    tov_pg_model = load_one('tov_pg')
    min_pg_model = load_one('min_pg')
    ts_pct_model = load_one('ts_pct')

    return (
        pts_pg_model, ast_pg_model, blk_pg_model, reb_pg_model, gp_model, gs_model,
        fga_pg_model, height_model, fg3a_pg_model,
        fta_pg_model, tov_pg_model, min_pg_model, ts_pct_model
    )


if __name__ == '__main__':

    pts_pg_model, ast_pg_model, blk_pg_model, reb_pg_model, gp_model, gs_model, fga_pg_model, height_model, fg3a_pg_model, fta_pg_model, tov_pg_model, min_pg_model, ts_pct_model = load_all()

    print(pts_pg_model.r2)
