import os
import httpx

def handler(request):
    try:
        with httpx.Client(timeout=10.0) as client:
            client.post(
                "https://api-inference.huggingface.co/models/chahatsaini1309/moodmatch-emotion-model",
                headers={
                    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
                    "x-wait-for-model": "true",
                },
                json={"inputs": "warmup"},
            )
    except Exception:
        pass

    return {
        "statusCode": 200,
        "body": "ok"
    }
