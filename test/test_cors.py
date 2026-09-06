from api.main import app as app1
from app.api.main import app as app2
print('Same app:', app1 is app2)
print('Routes:', len(app1.routes))
from fastapi.testclient import TestClient
client = TestClient(app1)
resp = client.options('/api/v1/auth/signup', headers={
    'Origin': 'http://localhost:3000',
    'Access-Control-Request-Method': 'POST',
    'Access-Control-Request-Headers': 'Content-Type',
})
print('OPTIONS /auth/signup:', resp.status_code)
print('Allow-Origin:', resp.headers.get('access-control-allow-origin'))
print('Allow-Methods:', resp.headers.get('access-control-allow-methods'))
