from pydantic import BaseModel


class PreProcessRequest(BaseModel):
    b64_image: str
    image_name: str
