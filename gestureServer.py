from flask import Flask, request, jsonify
from collections import deque
import threading

app = Flask(__name__)
sequence_history = deque(maxlen=100)
latest_sequence = None
latest_fist_status = False

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

# Fist status routes
@app.route("/upload_fist", methods=["POST"])
def upload_fist():
    global latest_fist_status
    data = request.get_json()
    latest_fist_status = data.get("fist_closed", False)
    return jsonify({"status": "ok"})

@app.route("/get_fist", methods=["GET"])
def get_fist():
    return jsonify({"fist_closed": latest_fist_status})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=50007)