from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'traffic_secret_key'

@app.route("/")
def index():
    return render_template('Index.html')


@app.route("/about")
def about():
    return render_template('About.html')

@app.route("/login")
def login():
    return render_template('login.html')


if __name__ == "__main__":
    app.run(debug=True)