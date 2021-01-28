from typing import List

from fastapi import FastAPI, HTTPException

from data import data
import schema

app = FastAPI(title="Demo App")

@app.get("/notes", response_model=List[schema.Note])
def get_notes():
    return data

@app.get("/notes/{note_id}", response_model=schema.Note)
def get_note(note_id: int):
    for note in data:
        if note.id == note_id:
            return note

    raise HTTPException(status_code=404, detail=f"vc acha que tem {note_id}? nao tem nao")

@app.post("/notes/{note_id}", response_model=schema.Note, status_code=201)
def post_note(note_create: schema.NoteCreate):

    #TODO: do the data handling here

    response = schema.Note(
        id=5,
        title=note_create.title,
        text=note_create.text,
        tags=note_create.tags,
    )

    return response

    