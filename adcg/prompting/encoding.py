import base64
import json
import re
from pathlib import Path


MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def image_to_data_url(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"이미지를 찾을 수 없습니다: {image_path}"
        )

    mime_type = MIME_TYPES.get(image_path.suffix.lower())

    if mime_type is None:
        raise ValueError(
            f"지원하지 않는 이미지 형식입니다: "
            f"{image_path.suffix}"
        )

    encoded = base64.b64encode(
        image_path.read_bytes()
    ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def extract_json(text):
    text = str(text or "").strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                f"응답에서 JSON 객체를 찾지 못했습니다:\n{text}"
            )

        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError as error:
            raise ValueError(
                f"유효하지 않은 JSON 응답입니다:\n{text}"
            ) from error