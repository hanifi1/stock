from flask import Flask
import git
from src.business_logic.process_query import create_business_logic
from src.algo.dummy_model import Predict

app = Flask(__name__)



@app.route('/', methods=['GET'])
def hello():
    return f'Please enter ticker name!\nEX: get_stock_val/<ticker>\n'


@app.route('/get_stock_val/<ticker>', methods=['GET'])
def get_stock_value(ticker):
    # bl = create_business_logic(ticker)
    prediction = Predict(ticker)

    return f'{prediction}\n'


@app.route('/getversion/')
def getversion():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    return f'{sha}\n'


if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='localhost', port=8081, debug=True)
