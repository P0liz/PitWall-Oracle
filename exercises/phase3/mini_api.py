from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


# Basically a python dictionary represented by a json
class Item(BaseModel):
    driver_name: str
    grid_position: int


app = FastAPI()


# Return driver lower position
@app.post("/predict_test")
async def order_items(items: List[Item]):
    items.sort(key=lambda x: x.grid_position)
    return items[0].driver_name
