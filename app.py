from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/ssd")
def ssd_test():
    import keras
    return "Hello ssd"


if __name__ == '__main__':
    app.run(debug=True)
    