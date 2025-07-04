from flask import Flask, request, jsonify
import pandas as pd
import sys
import os
import traceback
import os
from joblib import load
import random

app = Flask(__name__)

import random

STAT_RESPONSE_TEMPLATES = {
    "pts_pg": {
        "high": [
            "Scoring machine! {val:.1f} points per game is elite.",
            "{val:.1f} PPG? This player is lighting it up!"
        ],
        "medium": [
            "{val:.1f} points per game, a reliable scorer.",
            "Solid scorer with {val:.1f} PPG."
        ],
        "low": [
            "Scoring is light, just {val:.1f} per game.",
            "{val:.1f} PPG might not move the needle."
        ]
    },
    "ast_pg": {
        "high": [
            "This player’s got vision, {val:.1f} assists per game!",
            "Playmaker alert: {val:.1f} APG."
        ],
        "medium": [
            "A decent distributor with {val:.1f} assists per game.",
            "{val:.1f} APG, moving the ball well."
        ],
        "low": [
            "Not much of a passer, only {val:.1f} APG.",
            "{val:.1f} assists per game, minimal playmaking."
        ]
    },
    "blk_pg": {
        "high": [
            "A defensive wall, {val:.1f} blocks per game!",
            "Shot blocker alert: {val:.1f} BPG."
        ],
        "medium": [
            "Respectable rim protection at {val:.1f} blocks.",
            "Averaging {val:.1f} BPG, holding it down."
        ],
        "low": [
            "Just {val:.1f} blocks, not much rim protection.",
            "This player doesn’t block much: {val:.1f} per game."
        ]
    },
    "reb_pg": {
        "high": [
            "Board beast! {val:.1f} rebounds a game.",
            "{val:.1f} RPG — always cleaning the glass."
        ],
        "medium": [
            "Decent rebounding with {val:.1f} per game.",
            "Solid rebounder: {val:.1f} RPG."
        ],
        "low": [
            "Light on rebounds — {val:.1f} per game.",
            "{val:.1f} RPG — might need more hustle."
        ]
    },
    "fga_pg": {
        "high": [
            "{val:.1f} shots a game — volume scorer!",
            "Letting it fly with {val:.1f} FGA per game."
        ],
        "medium": [
            "Takes a moderate {val:.1f} shots per game.",
            "{val:.1f} FGA — knows when to shoot."
        ],
        "low": [
            "Only {val:.1f} shots a game — selective shooter.",
            "Low volume: {val:.1f} FGA."
        ]
    },
    "fg3a_pg": {
        "high": [
            "Shooting from deep — {val:.1f} threes per game.",
            "{val:.1f} 3PA — living beyond the arc!"
        ],
        "medium": [
            "Comfortable from range, {val:.1f} 3PA.",
            "Shooting some threes: {val:.1f} per game."
        ],
        "low": [
            "Rarely takes threes, only {val:.1f} per game.",
            "Low 3-point volume: {val:.1f} 3PA."
        ]
    },
    "fta_pg": {
        "high": [
            "Gets to the line often, {val:.1f} FTA per game.",
            "{val:.1f} free throws per game, aggressive!"
        ],
        "medium": [
            "Draws some contact: {val:.1f} FTA.",
            "{val:.1f} free throws per game — decent pressure."
        ],
        "low": [
            "Doesn’t draw many fouls — {val:.1f} FTA.",
            "Low free throw attempts: {val:.1f}."
        ]
    },
    "tov_pg": {
        "high": [
            "Careless with the ball — {val:.1f} turnovers per game.",
            "{val:.1f} TOV per game — turnover-prone."
        ],
        "medium": [
            "Turns it over a bit: {val:.1f} per game.",
            "{val:.1f} TOV — manageable, but could be better."
        ],
        "low": [
            "Protects the ball well: {val:.1f} turnovers per game.",
            "Low turnover rate — just {val:.1f} TOV."
        ]
    },
    "min_pg": {
        "high": [
            "Heavy minutes: {val:.1f} per game.",
            "{val:.1f} MPG, a key player on the floor."
        ],
        "medium": [
            "Gets decent playing time: {val:.1f} minutes.",
            "{val:.1f} MPG, trusted rotation piece."
        ],
        "low": [
            "Limited minutes, {val:.1f} per game.",
            "{val:.1f} MPG — not always in the mix."
        ]
    }
}

def generate_natural_response(target: str, pred: float):
    thresholds = {
        "pts_pg": (20, 10),
        "ast_pg": (7, 3),
        "blk_pg": (1.5, 0.5),
        "reb_pg": (10, 5),
        "fga_pg": (15, 8),
        "fg3a_pg": (7, 3),
        "fta_pg": (6, 3),
        "tov_pg": (3.5, 1.5),
        "min_pg": (25, 17)
    }

    if target not in STAT_RESPONSE_TEMPLATES:
        return f"Prediction: {pred:.1f} {target.replace('_pg', '')} per game."

    high, medium = thresholds.get(target, (float('inf'), 0))

    if pred >= high:
        category = "high"
    elif pred >= medium:
        category = "medium"
    else:
        category = "low"

    sentence = random.choice(STAT_RESPONSE_TEMPLATES[target][category])
    return sentence.format(val=pred)



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

        input_order = [
            'reb_pg', 'gp', 'gs', 'pts_pg', 'ast_pg', 'fga_pg', 'height',
            'bodyWeight', 'fg3a_pg', 'fta_pg', 'tov_pg', 'min_pg', 'ts_pct'
        ]

        input_features = [col for col in input_order if col != target]

        df = pd.DataFrame([[features[col]
                          for col in input_features]], columns=input_features)

        pred = float(model.predict(df)[0])

        sentence = generate_natural_response(target, pred)

        return jsonify({
            'prediction': pred,
            'target': target,
            'sentence': sentence
        })

    except Exception as e:
        tb_str = traceback.format_exc()  # get full traceback as a string
        print(tb_str)

        return jsonify({'error': str(e), 'traceback': tb_str}), 500


if __name__ == '__main__':
    app.run(debug=True)
