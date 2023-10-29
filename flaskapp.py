from flask import Flask, render_template, request
from svd_model import best_model

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    key_mapping = {
        'A': 0,
        'A#': 1,
        'B': 2,
        'C': 3,
        'C#': 4,
        'D': 5,
        'D#': 6,
        'E': 7,
        'F': 8,
        'F#': 9,
        'G': 10,
        'G#': 11
    }

    if request.method == 'POST':
        key_value = key_mapping.get(request.form['key'])

        features = [
            float(request.form['danceability']),
            float(request.form['valence']),
            float(request.form['acousticness']),
            float(request.form['energy']),
            float(request.form['speechiness']),
            float(request.form['bpm']),
            float(request.form['instrumentalness']),
            key_value 
        ]

        features = [value/10 for value in features]
        
        result = best_model.predict([features])
        
        mode = "Major" if result[0] == 1 else "Minor"
        return render_template('index.html', prediction=mode)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':

    app.run(debug=True)
