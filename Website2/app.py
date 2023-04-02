from unicodedata import category
from flask import Flask
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy

from flask import jsonify
from flask import request
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import datetime
from flask_migrate import Migrate, migrate


app = Flask(__name__,static_url_path='/static')


# adding configuration for using a sqlite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


 


# Creating an SQLAlchemy instance
db = SQLAlchemy(app)

# Settings for migrations
migrate = Migrate(app, db)
 
# Models
class advertisment(db.Model):
 
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(20), unique=False, nullable=True)
    description = db.Column(db.String(1000), unique=False, nullable=False)
    salary = db.Column(db.String(50), nullable=False)
    job_category = db.Column(db.String(20), unique=False, nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.datetime.utcnow)
 



@app.route("/",methods=['GET','POST'])
def home():

    
    if request.method == "POST":
        category = request.form.get('category')
 
        ads = advertisment.query.filter(advertisment.job_category == category).all()

        return render_template('homepage.html',title="jobs",
                           ads=ads)
    return render_template('homepage.html')

@app.route("/manager")
def manager():
    return render_template('manager.html')




@app.route("/post_data",methods=['GET','POST'])
def post_data():
    sentence_pred= ''

    if request.method == "POST":
        job_title_post = request.form.get('job_title', '')
        job_desc = request.form.get('job_desc', '')
        radio_post = request.form.get('suggested_choice', '')
        salary_post = request.form.get('salary', '')
        #load logistic regression pre-trained model
        lr_loaded = pickle.load(open('model.sav', 'rb'))

        transformer = TfidfTransformer()

        # load tf_idf pre fitted vocabulary
        tf1 = pickle.load(open("X_vectorizer.pkl", 'rb'))

        # Create new tfidfVectorizer with old vocabulary
        tf1_new = TfidfVectorizer(vocabulary = tf1) 

        #convert sentence to list to be used by the model
        job_desc = [job_desc]
        test_sentence_tfidf = tf1_new.fit_transform(job_desc)
        test_sentence_tfidf = test_sentence_tfidf


        sentence_pred = lr_loaded.predict(test_sentence_tfidf)
        sentence_pred = sentence_pred[0]
        print(sentence_pred)

        response = "failed"
        if radio_post == "true":

            ad_object = advertisment(title =job_title_post, description = job_desc[0],salary = salary_post,job_category =sentence_pred )
            db.session.add(ad_object)
            db.session.commit()
            feedback = "success"
            return jsonify(feedback=feedback)

        elif radio_post == "false":
            custom_category = request.form.get('custom_category', '')
            print(custom_category)
            ad_object = advertisment(title =job_title_post, description = job_desc[0],salary = salary_post,job_category =custom_category )
            db.session.add(ad_object)
            db.session.commit()
            feedback = "success"
            return jsonify(feedback=feedback)

        else:
            
            return jsonify(
                category=sentence_pred,
            )


