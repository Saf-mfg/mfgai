import chromadb

client = chromadb.PersistentClient(path="./humhub_db")
collection = client.get_or_create_collection(name="humhub_content")
print("COUNT:", collection.count())
