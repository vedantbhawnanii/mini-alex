import requests

from config import Config


class WhatsAppClient:
    def __init__(self):
        self.base_url = Config.META_ENDPOINT
        self.headers = {
            "Authorization": f"Bearer {Config.GRAPH_API_TOKEN}",
            "Content-Type": "application/json",
        }

    def send_text_message(
        self, phone_number_id, to_number, text, context_message_id=None
    ):
        payload = {
            "messaging_product": "whatsapp",
            "to": to_number,
            "text": {"body": text},
        }

        if context_message_id:
            payload["context"] = {"message_id": context_message_id}
        print(
            f"In whatsapp client send message.\nPayload: {payload}\nheaders: {self.headers}\n"
        )
        response = requests.post(
            f"{self.base_url}/508861485653521/messages",
            json=payload,
            headers=self.headers,
        )
        print(response.json())
        return response.json()

    def mark_message_read(self, phone_number_id, message_id):
        response = requests.post(
            f"{self.base_url}/{phone_number_id}/messages",
            json={
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": message_id,
            },
            headers=self.headers,
        )
        return response.json()
