from fastapi import FastAPI
from typing import List
from pydantic import BaseModel

class ProfileAnswers(BaseModel):
    answers: List[int]










