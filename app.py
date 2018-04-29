# pipenv run python app.py

from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
  return "Hello World!"


# endpoint to get detail by day number
# @app.route("/day", methods=["GET"])
# def day():
#     return "first day"


# endpoint to get detail by day number
@app.route("/day/<id>", methods=["GET"])
def day(id):

  if int(id) == 1:
    return '{ "day": "1", "banana": [100, 100], "apple": [82, 81, 80, 79, 78], "brocolli": [40, 35, 33] }'
  elif int(id) == 2:
    return '{ "day": "2", "banana": [95, 90], "apple": [81, 80, 79, 78], "brocolli": [54, 27] }'
  elif int(id) == 3:
    return '{ "day": "3", "banana": [90, 80], "apple": [80, 79, 78, 77], "brocolli": [0] }'
  elif int(id) == 4:
    return '{ "day": "4", "banana": [90, 70], "apple": [79, 78, 77], "brocolli": [66, 22, 98, 89] }'
  elif int(id) == 5:
    return '{ "day": "5", "banana": [60], "apple": [78, 77], "brocolli": [75, 25] }'
  else:
    return "{}"


if __name__ == '__main__':
  app.run(debug=True)
    