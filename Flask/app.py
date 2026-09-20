from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/second_page')
def second():
    return render_template("second.html")
    print("this is my second page")



if __name__ == '__main__':
    app.run(debug=True)