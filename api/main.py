from flask import Flask, request, jsonify
import pandas as pd
import sys
import os
import traceback

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


# Add the project root (Court Sense ai) to sys.path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))


app = Flask(__name__)


@app.route("/basic-predict", methods=['POST'])
def basic_predict():

    valid_models = ['pts_pg', 'ast_pg', 'blk_pg', 'reb_pg', 'gp', 'gs',
                    'fga_pg', 'height', 'fg3a_pg',
                    'fta_pg', 'tov_pg', 'min_pg', 'ts_pct']

    try:

        data = request.get_json()

        if not data or 'target' not in data or 'features' not in data:
            return jsonify({"error": "Missing 'target' or 'features' in the request body"}), 400

        target = data['target']
        features = data['features']

        if target not in valid_models:
            return jsonify({'error': "'target' was not a valid model"}), 400

        if target in features:
            return jsonify({
                "error": f"'{target}' should not be included in 'features'. It's the target, not an input."
            }), 400

        required_features = [
            'pts_pg', 'ast_pg', 'blk_pg', 'reb_pg', 'gp', 'gs', 'fga_pg', 'height',
            'bodyWeight', 'fg3a_pg', 'fta_pg', 'tov_pg', 'min_pg', 'ts_pct'
        ]

        unexpected_keys = [
            key for key in features if key not in required_features]

        if unexpected_keys:
            return jsonify({
                "error": f"Unexpected feature(s): {', '.join(unexpected_keys)}",
                "unexpected": unexpected_keys
            }), 400

        non_numeric = [
            key for key, val in features.items()
            if not isinstance(val, (int, float))
        ]

        if non_numeric:
            return jsonify({
                "error": f"Non-numeric values found for: {', '.join(non_numeric)}",
                "invalid": non_numeric
            }), 400

        missing_keys = [
            key for key in required_features
            if key != target and (key not in features or features[key] in [None, ""])
        ]

        if missing_keys:
            return jsonify({
                "error": f"Missing required feature(s): '{', '.join(missing_keys)}' ",
                'missing': missing_keys
            }), 400

        model = load_one(target)

        if target == 'blk_pg':
            input_order = [
                'reb_pg', 'gp', 'gs', 'pts_pg', 'ast_pg', 'fga_pg',
                'height', 'bodyWeight', 'fg3a_pg', 'fta_pg', 'tov_pg', 'min_pg', 'ts_pct'
            ]
        else:
            input_order = [
                'gp', 'gs', 'pts_pg', 'ast_pg', 'fga_pg', 'height',
                'bodyWeight', 'fg3a_pg', 'fta_pg', 'tov_pg', 'min_pg', 'ts_pct'
            ]

        input_features = [col for col in input_order if col != target]

        df = pd.DataFrame([[features[col]
                          for col in input_features]], columns=input_features)

        pred = float(model.predict(df)[0])

        return jsonify({
            'prediction': pred,
            'target': target
        })

    except Exception as e:
        tb_str = traceback.format_exc()  # get full traceback as a string
        print(tb_str)

        return jsonify({'error': str(e), 'traceback': tb_str}), 500


if __name__ == '__main__':
    app.run(debug=True)
