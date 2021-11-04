from pathlib import Path

import uvicorn
from fastapi import FastAPI
from joblib import load

# 1. Load the trained regressor
regressor = load(Path(__file__).parent / 'resources' / 'regressor.joblib')

# 2. Create the app object
app = FastAPI()


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, stranger'}


# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'message': f'Hello, {name}'}


# 5. Route for model prediction, make a prediction from the passed
#    JSON data and return the sklearn model prediction
@app.post('/predict')
def predict(x1: float, x2: float, x3: float, x4: float):
    y = regressor.predict(X=[[x1, x2, x3, x4]])
    return {
        'prediction': y[0]
    }


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
