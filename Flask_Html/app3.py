from flask import Flask, redirect, url_for, render_template, request 
app = Flask(__name__)


@app.route("/")
def home():
    return "<h1>Text</h1>"

@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        user = request.form["nm"]
        return redirect(url_for("user" , usr=user))
    return render_template("index3.html")

@app.route("/<usr>")
def user(usr):
    return f"<h1>{usr}</h1>"

   
if __name__ == "__main__":
    app.run(debug=True)
