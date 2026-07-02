"""
app.py
------
Entry point for the Flood Prediction System.

Run with:
    python app.py

Make sure you have trained the model first:
    python train.py
"""

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
