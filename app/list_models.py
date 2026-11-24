from generate import client

for m in client.models.list().data:
    print(m.id)