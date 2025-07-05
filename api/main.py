from flask import Flask, request, jsonify, render_template
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
            "Based on the input, I project a high scoring output of {val:.2f} PPG.",
            "The model predicts a top-tier scoring average: {val:.2f} points per game.",
            "Expected to be a primary scorer with approximately {val:.2f} points each game."
        ],
        "medium": [
            "Prediction suggests a moderate scoring role at {val:.2f} PPG.",
            "{val:.2f} points per game is consistent with a solid offensive contributor.",
            "Likely to contribute reliably with {val:.2f} points on average."
        ],
        "low": [
            "The forecast indicates limited scoring impact at {val:.2f} points per game.",
            "Projected to score minimally, around {val:.2f} PPG.",
            "Not expected to be a major scorer, with an average of {val:.2f} points."
        ]
    },
    "ast_pg": {
        "high": [
            "Prediction points to advanced playmaking: {val:.2f} assists per game.",
            "{val:.2f} APG suggests a high-volume distributor.",
            "Expected to facilitate the offense efficiently with {val:.2f} assists."
        ],
        "medium": [
            "Model expects solid distribution skills: {val:.2f} APG.",
            "{val:.2f} assists per game implies dependable ball movement.",
            "A moderate level of playmaking is predicted at {val:.2f} APG."
        ],
        "low": [
            "Prediction indicates limited assists: {val:.2f} per game.",
            "Low assist average expected — approximately {val:.2f} per game.",
            "{val:.2f} APG implies minimal involvement in playmaking duties."
        ]
    },
    "blk_pg": {
        "high": [
            "Projected to be a strong rim protector with {val:.2f} blocks per game.",
            "Defensive impact expected to be high: {val:.2f} BPG.",
            "Model predicts consistent shot blocking at {val:.2f} blocks per game."
        ],
        "medium": [
            "Forecast suggests moderate rim protection: {val:.2f} BPG.",
            "{val:.2f} blocks per game indicates a respectable defensive presence.",
            "Expected to contest shots occasionally with {val:.2f} BPG."
        ],
        "low": [
            "Low block output anticipated at {val:.2f} per game.",
            "Minimal rim protection likely: {val:.2f} BPG.",
            "Blocking is not expected to be a key contribution — {val:.2f} per game."
        ]
    },
    "reb_pg": {
        "high": [
            "Projection suggests strong rebounding presence: {val:.2f} RPG.",
            "Expected to dominate the boards with {val:.2f} rebounds per game.",
            "{val:.2f} rebounds per game indicates high involvement in securing possessions."
        ],
        "medium": [
            "Forecast predicts a decent rebounding average: {val:.2f} RPG.",
            "Model indicates solid contribution on the glass at {val:.2f} rebounds.",
            "Predicted to rebound at a steady clip: {val:.2f} RPG."
        ],
        "low": [
            "Rebounding may be a weak area — {val:.2f} RPG forecasted.",
            "Low rebounding numbers expected at around {val:.2f} per game.",
            "{val:.2f} rebounds suggest limited impact on the boards."
        ]
    },
    "fga_pg": {
        "high": [
            "Model projects high shooting volume: {val:.2f} FGA per game.",
            "{val:.2f} shots per game expected — primary scoring option.",
            "Heavy usage projected with {val:.2f} field goal attempts."
        ],
        "medium": [
            "Expected to take a balanced number of shots: {val:.2f} FGA.",
            "Moderate field goal volume predicted at {val:.2f} attempts per game.",
            "The player is projected to take about {val:.2f} shots per game."
        ],
        "low": [
            "Shooting frequency is low — around {val:.2f} FGA per game.",
            "Minimal shot attempts forecasted at {val:.2f} per game.",
            "Not projected to shoot often — {val:.2f} attempts on average."
        ]
    },
    "fg3a_pg": {
        "high": [
            "High volume of three-point attempts projected: {val:.2f} per game.",
            "{val:.2f} 3PA suggests frequent perimeter shooting.",
            "Model indicates a strong focus beyond the arc — {val:.2f} threes attempted."
        ],
        "medium": [
            "Moderate three-point activity expected — {val:.2f} attempts per game.",
            "Prediction suggests balanced perimeter involvement: {val:.2f} 3PA.",
            "{val:.2f} three-point attempts per game is a healthy middle ground."
        ],
        "low": [
            "Low three-point volume expected at {val:.2f} per game.",
            "Not likely to take many threes — only {val:.2f} per game forecasted.",
            "{val:.2f} 3PA implies limited shooting from deep."
        ]
    },
    "fta_pg": {
        "high": [
            "Aggressive driving projected — {val:.2f} FTA per game.",
            "Player expected to draw a high number of fouls: {val:.2f} free throws.",
            "The model indicates frequent trips to the line at {val:.2f} attempts."
        ],
        "medium": [
            "Predicted to earn some free throws — around {val:.2f} per game.",
            "Moderate foul drawing expected at {val:.2f} FTA.",
            "{val:.2f} free throw attempts indicates decent paint pressure."
        ],
        "low": [
            "Low frequency of free throws — {val:.2f} FTA predicted.",
            "Minimal contact drawn — only {val:.2f} trips to the line per game.",
            "Free throw attempts are not expected to be high — just {val:.2f}."
        ]
    },
    "tov_pg": {
        "high": [
            "Model forecasts high turnover risk — {val:.2f} per game.",
            "Possession control could be a concern with {val:.2f} TOV.",
            "{val:.2f} turnovers projected, indicating loose ball security."
        ],
        "medium": [
            "Turnovers projected at a manageable level — {val:.2f} per game.",
            "Ball security is fair, with {val:.2f} TOV predicted.",
            "{val:.2f} turnovers suggest average handling under pressure."
        ],
        "low": [
            "Possession expected to be well-managed — {val:.2f} turnovers.",
            "Low turnover risk forecasted: {val:.2f} TOV per game.",
            "{val:.2f} turnovers shows strong ball control."
        ]
    },
    "min_pg": {
        "high": [
            "Model suggests heavy minutes — around {val:.2f} MPG.",
            "Player is expected to be a major rotation piece with {val:.2f} minutes.",
            "{val:.2f} minutes per game indicates a core role."
        ],
        "medium": [
            "Moderate playing time expected — about {val:.2f} minutes.",
            "Projection shows decent rotation usage: {val:.2f} MPG.",
            "{val:.2f} minutes suggests steady floor presence."
        ],
        "low": [
            "Limited playing time forecasted — only {val:.2f} minutes.",
            "{val:.2f} MPG suggests a smaller role in the rotation.",
            "Not expected to log many minutes — around {val:.2f} per game."
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
        "min_pg": (25, 11)
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
    return render_template('index.html')



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)


