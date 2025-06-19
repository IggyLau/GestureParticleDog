from flask import Flask, request, jsonify
from collections import deque

app = Flask(__name__)
sequence_history = deque(maxlen=100)
latest_sequence = None

# Gesture sequence routes
@app.route("/upload_Fingersequence", methods=["POST"])
def upload_finger_sequence():
    data = request.get_json()
    sequence_history.append(data)
    app.logger.info(f"Appended new finger sequence: {data}")
    return jsonify({"status": "ok"})

@app.route("/get_Fingersequence", methods=["GET"])
def get_finger_sequence():
    return jsonify({"history": list(sequence_history)})

# Other sequence routes
@app.route("/upload_sequence", methods=["POST"])
def upload_sequence():
    global latest_sequence
    latest_sequence = request.get_json()["sequence"]
    return jsonify({"status": "ok"})

@app.route("/get_sequence", methods=["GET"])
def get_sequence():
    return jsonify({"sequence": latest_sequence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=50007)