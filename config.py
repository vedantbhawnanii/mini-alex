import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    GRAPH_API_TOKEN = os.getenv("ACCESS_TOKEN")
    META_ENDPOINT = os.getenv("META_ENDPOINT")
    PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
    WEBHOOK_VERIFY_TOKEN = os.getenv("WEBHOOK_VERIFY_TOKEN")

    MONGODB_URI = os.getenv("MONGODB_URI")
    PORT = int(os.getenv("PORT", 3000))
