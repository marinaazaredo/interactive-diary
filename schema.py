from typing import Optional, List

from pydantic import BaseModel

class NoteCreate(BaseModel):
    title: Optional [str]
    text: str
    tags: List[str]

    class Config:
        schema_extra = {
          "example": {
                "title": "Shopping List",
                "text": "eggs, butter",
                "tags": ["SHOPPING"],
          }  
        }

class Note(BaseModel):
    id: int
    title: Optional [str]
    text: str
    tags: List[str]

    class Config:
        schema_extra = {
            "example": {
                "id": 123,
                "title": "Shopping List",
                "text": "eggs, butter",
                "tags": ["SHOPPING"],
        }
    }

