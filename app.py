from flask import Flask, jsonify, request

from config import Config
from src.webhooks.whatsapp_webhook import handle_webhook

app = Flask(__name__)

# Register webhook endpoint
app.add_url_rule("/", "whatsapp_webhook", handle_webhook, methods=["GET", "POST"])


# @app.route("/", methods=["POST"])
# def handle_post():
#     print("-------------- New Request POST --------------")
#     print("Headers:", request.headers)
#     print("Body:", request.json)
#     return jsonify({"message": "Thank you for the message"})


@app.route("/", methods=["GET"])
def handle_get():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    print("-------------- New Request GET --------------")
    print("Headers:", request.headers)
    print("Body:", request.args)

    if mode and token:
        if mode == "subscribe" and token == "12345":
            print("WEBHOOK_VERIFIED")
            return challenge, 200
        else:
            print("Responding with 403 Forbidden")
            return "Forbidden", 403
    else:
        print("Replying Thank you.")
        return jsonify({"message": "Thank you for the message"})


if __name__ == "__main__":
    print(f"Example Facebook app listening at {Config.PORT}")
    app.run(host="0.0.0.0", port=Config.PORT, debug=True)
