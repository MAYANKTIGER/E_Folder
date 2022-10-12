from flask import Flask 

app = Flask(__name__)
print(type(app))

"""A Flask Application is an instance of Flask Class. Everything about the application, such as configuration and URL, will be
registered with this class 
__name__ is the name of the current Python module. The app needs to know where it's located to set up some paths,
 and __name__ is a convenient way to tell it that """



if __name__ == '__main__':
    app.run(debug=True)


# ref :: https://flask.palletsprojects.com/en/1.1.x/tutorial/factory/
