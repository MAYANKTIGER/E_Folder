from datetime import datetime
from email.policy import default
from turtle import title
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import PrimaryKeyConstraint, desc



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///Mytodo.db"
app.config['SQLALCHEMY_TRACK_MODIFICATION'] = False
db = SQLAlchemy(app)

class Todo(db.Model):
    sno = db.Column(db.Integer , primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    desc = db.Column(db.String(500), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"{self.sno} - {self.title}"
    




@app.route("/")
def hello_world():
    todo = Todo(title="First todo", desc="Get the emial IDs")
    db.session.add(todo)
    db.session.commit()
    return render_template('index.html')
    #return "<p>Hello, World!</p>"

@app.route("/products")
def products():
    return "this is product page"

if __name__ == "__main__":  
    app.run(debug=True)