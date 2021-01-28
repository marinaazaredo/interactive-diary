import schema


data = [
    schema.Note(
        id=1,
        title="Note 1",
        text= "Text",
        tags=["important", "urgent"] 
    ),
     schema.Note(
        id=2,
        title="Note 2",
        text= "Text 2",
        tags=["important", "urgent"] 
    ),
     schema.Note(
        id=3,
        title="Note 3",
        text= "Text 3",
        tags=[] 
    ),
]