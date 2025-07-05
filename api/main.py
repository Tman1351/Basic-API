from flask import Flask, request, jsonify
import pandas as pd
from sys import path
import os
import traceback
from joblib import load
from random import choice

app = Flask(__name__)

path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))

STAT_RESPONSE_TEMPLATES = {
    "pts_pg": {
        "high": [
            "Scoring machine! {val:.2f} points per game is elite.",
            "{val:.2f} PPG? This player is lighting it up!"
        ],
        "medium": [
            "{val:.2f} points per game, a reliable scorer.",
            "Solid scorer with {val:.2f} PPG."
        ],
        "low": [
            "Scoring is light, just {val:.2f} per game.",
            "{val:.2f} PPG might not move the needle."
        ]
    },
    "ast_pg": {
        "high": [
            "This player’s got vision, {val:.2f} assists per game!",
            "Playmaker alert: {val:.2f} APG."
        ],
        "medium": [
            "A decent distributor with {val:.2f} assists per game.",
            "{val:.2f} APG, moving the ball well."
        ],
        "low": [
            "Not much of a passer, only {val:.2f} APG.",
            "{val:.2f} assists per game, minimal playmaking."
        ]
    },
    "blk_pg": {
        "high": [
            "A defensive wall, {val:.2f} blocks per game!",
            "Shot blocker alert: {val:.2f} BPG."
        ],
        "medium": [
            "Respectable rim protection at {val:.2f} blocks.",
            "Averaging {val:.2f} BPG, holding it down."
        ],
        "low": [
            "Just {val:.2f} blocks, not much rim protection.",
            "This player doesn’t block much: {val:.2f} per game."
        ]
    },
    "reb_pg": {
        "high": [
            "Board beast! {val:.2f} rebounds a game.",
            "{val:.2f} RPG — always cleaning the glass."
        ],
        "medium": [
            "Decent rebounding with {val:.2f} per game.",
            "Solid rebounder: {val:.2f} RPG."
        ],
        "low": [
            "Light on rebounds — {val:.2f} per game.",
            "{val:.2f} RPG — might need more hustle."
        ]
    },
    "fga_pg": {
        "high": [
            "{val:.2f} shots a game — volume scorer!",
            "Letting it fly with {val:.2f} FGA per game."
        ],
        "medium": [
            "Takes a moderate {val:.2f} shots per game.",
            "{val:.2f} FGA — knows when to shoot."
        ],
        "low": [
            "Only {val:.2f} shots a game — selective shooter.",
            "Low volume: {val:.2f} FGA."
        ]
    },
    "fg3a_pg": {
        "high": [
            "Shooting from deep — {val:.2f} threes per game.",
            "{val:.2f} 3PA — living beyond the arc!"
        ],
        "medium": [
            "Comfortable from range, {val:.2f} 3PA.",
            "Shooting some threes: {val:.2f} per game."
        ],
        "low": [
            "Rarely takes threes, only {val:.2f} per game.",
            "Low 3-point volume: {val:.2f} 3PA."
        ]
    },
    "fta_pg": {
        "high": [
            "Gets to the line often, {val:.2f} FTA per game.",
            "{val:.2f} free throws per game, aggressive!"
        ],
        "medium": [
            "Draws some contact: {val:.2f} FTA.",
            "{val:.2f} free throws per game — decent pressure."
        ],
        "low": [
            "Doesn’t draw many fouls — {val:.2f} FTA.",
            "Low free throw attempts: {val:.2f}."
        ]
    },
    "tov_pg": {
        "high": [
            "Careless with the ball — {val:.2f} turnovers per game.",
            "{val:.2f} TOV per game — turnover-prone."
        ],
        "medium": [
            "Turns it over a bit: {val:.2f} per game.",
            "{val:.2f} TOV — manageable, but could be better."
        ],
        "low": [
            "Protects the ball well: {val:.2f} turnovers per game.",
            "Low turnover rate — just {val:.2f} TOV."
        ]
    },
    "min_pg": {
        "high": [
            "Heavy minutes: {val:.2f} per game.",
            "{val:.2f} MPG, a key player on the floor."
        ],
        "medium": [
            "Gets decent playing time: {val:.2f} minutes.",
            "{val:.2f} MPG, trusted rotation piece."
        ],
        "low": [
            "Limited minutes, {val:.2f} per game.",
            "{val:.2f} MPG — not always in the mix."
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

    zero_templates = [
        'This player makes no contribution in this stat... Like at all. 😓 Complete zero.', 'I Predicted... 0.',
        'I think 0.', "The answer issss... a  B I G  f a t  zero. 😔", 'They average nothing at all in that.', 'I predict 0️⃣. 😂', 'Dang. They would most likely average 0.'
    ]

    if target not in STAT_RESPONSE_TEMPLATES:
        return f"Prediction: {pred:.2f} {target.replace('_pg', '')} per game."

    high, medium = thresholds.get(target, (float('inf'), 0))

    if pred <= 0:
        return choice(zero_templates)
    elif pred >= high:
        category = "high"
    elif pred >= medium:
        category = "medium"
    else:
        category = "low"

    sentence = choice(STAT_RESPONSE_TEMPLATES[target][category])
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



@app.route("/basic-predict", methods=['POST'])
def basic_predict():

    valid_models = ['pts_pg', 'ast_pg', 'blk_pg', 'reb_pg', 'gp', 'gs',
                    'fga_pg', 'height', 'fg3a_pg',
                    'fta_pg', 'tov_pg', 'min_pg', 'ts_pct']
    
    valid_targets = ['ast_pg', 'blk_pg', 'fg3a_pg', 'fga_pg', 'fta_pg', 'min_pg', 'pts_pg', 'reb_pg', 'tov_pg']

    required_features = [
            'pts_pg', 'ast_pg', 'blk_pg', 'reb_pg', 'gp', 'gs', 'fga_pg', 'height',
            'bodyWeight', 'fg3a_pg', 'fta_pg', 'tov_pg', 'min_pg', 'ts_pct'
        ]

    try:

        data = request.get_json()

        if not data or 'target' not in data or 'features' not in data:
            return jsonify({"error": "Missing 'target' or 'features' in the request body"}), 400

        target = data['target']
        features = data['features']

        if target not in valid_models:
            return jsonify({'error': "'target' was not a valid model"}), 400
        elif target in features:
            return jsonify({
                "error": f"'{target}' should not be included in 'features'. It's the target, not an input."
            }), 400
        elif target not in valid_targets:
            return jsonify({'error': f"'{target}' is not allowed to be predicted as of now. Reasons may be because of low accuracy or unstability"}), 400

        
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
    
@app.route("/", methods=["GET"])
def index():
    return "CourtSense API is running."



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)


