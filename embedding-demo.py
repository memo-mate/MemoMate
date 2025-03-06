import lancedb  # type: ignore
import pandas as pd
from lancedb.embeddings import get_registry  # type: ignore
from lancedb.pydantic import LanceModel, Vector  # type: ignore

db = lancedb.connect("./tmp/db")
model = get_registry().get("sentence-transformers").create(name="BAAI/bge-small-en-v1.5", device="cpu")


class Words(LanceModel):  # type: ignore
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()  # type: ignore


df = pd.DataFrame({"text": ["hi hello sayonara", "goodbye world"]})
table = db.create_table("greets", schema=Words)
table.add(df)
query = "old greeting"
actual = table.search(query).limit(1).to_pydantic(Words)[0]
print(actual.text)
