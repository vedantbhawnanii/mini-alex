from flask import jsonify, request

from src.model.rag_engine import RagAgent
from src.services.whatsapp_client import WhatsAppClient
from src.utils.logger import ColorLogger

logger = ColorLogger(__name__)
# client = MongoClient(Config.MONGODB_URI)
# db = client.get_database()
# messages_collection = db.messages

whatsapp_client = WhatsAppClient()
rag = RagAgent()


def handle_webhook():
    if request.method == "GET":
        return verify_webhook()

    else:
        # Process POST requests
        try:
            print(request)
            data = request.json
            entry = data.get("entry", [{}])[0]
            changes = entry.get("changes", [{}])[0]
            value = changes.get("value", {})
            message = value.get("messages", [{}])[0]
            # Store message in database
            # if message:
            # store_message(message, value.get("metadata", {}))

            if message.get("type") == "text":
                process_text_message(message, value.get("metadata", {}))

            return jsonify({"status": "success"}), 200

        except Exception as e:
            logger.error(f"Webhook processing error: {str(e)}")
            return jsonify({"error": "Processing failed"}), 500


def verify_webhook():
    print(request)
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == "12345":
        logger.info("Webhook verified successfully")
        return challenge, 200
    return jsonify({"error": "Verification failed"}), 403


# def store_message(message, metadata):
#     try:
#         messages_collection.insert_one(
#             {
#                 "from": message.get("from"),
#                 "text": message.get("text", {}).get("body"),
#                 "type": message.get("type"),
#                 "metadata": metadata,
#                 "timestamp": message.get("timestamp"),
#             }
#         )
#     except Exception as e:
#         logger.error(f"Message storage failed: {str(e)}")


def process_text_message(message, metadata):
    try:
        phone_number_id = metadata.get("phone_number_id")
        from_number = message.get("from")
        message_id = message.get("id")
        text_body = message.get("text", {}).get("body")
        # print(text_body)
        chain = rag.get_chain()
        response = rag.call_chain(text_body, chain)

        # Send reply
        whatsapp_client.send_text_message(
            phone_number_id,
            from_number,
            f"{response}",
            context_message_id=message_id,
        )

        # Mark as read
        whatsapp_client.mark_message_read(phone_number_id, message_id)

    except Exception as e:
        logger.error(f"Message processing failed: {str(e)}")
