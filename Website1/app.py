from unicodedata import category
from flask import Flask
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate, migrate

from flask import jsonify
from flask import request
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import datetime

app = Flask(__name__,static_url_path='/static')


# adding configuration for using a sqlite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


 


# Creating an SQLAlchemy instance
db = SQLAlchemy(app)

# Settings for migrations
migrate = Migrate(app, db)
 
# Models
class job(db.Model):
 
    id = db.Column(db.Integer, primary_key=True)
    job_title = db.Column(db.String(20), unique=False, nullable=False)
    job_description = db.Column(db.String(1000), unique=False, nullable=False)
    salary = db.Column(db.String(50), nullable=False)
    category = db.Column(db.String(20), unique=False, nullable=False)
    date = db.Column(db.DateTime, default=datetime.datetime.utcnow)
 
    def __repr__(self):
        return f"job title : {self.job_title}"


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/employer")
def employer():
    return render_template('employer.html')


@app.route("/create_ad")
def create_ad():

    return render_template('create_ad.html')



@app.route("/get_form_data",methods=['GET','POST'])
def get_form_data():
    sentence_pred= ''


    if request.method == "POST":

        job_title_post = request.form.get('job_title', '')
        job_desc = request.form.get('job_desc', '')
        radio_post = request.form.get('suggested_choice', '')
        salary_post = request.form.get('salary', '')

        #load logistic regression pre-trained model
        lr_loaded = pickle.load(open('lr_model.sav', 'rb'))

        transformer = TfidfTransformer()

        # load tf_idf pre fitted vocabulary
        tf1 = pickle.load(open("X_tfidf.pkl", 'rb'))

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

            job_object = job(job_title =job_title_post, job_description = job_desc[0],salary = salary_post,category =sentence_pred )
            db.session.add(job_object)
            db.session.commit()
            message = "success"
            return jsonify(message=message)

        elif radio_post == "false":
            custom_category = request.form.get('custom_category', '')
            job_object = job(job_title =job_title_post, job_description = job_desc[0],salary = salary_post,category =custom_category )
            db.session.add(job_object)
            db.session.commit()
            message = "success"
            return jsonify(message=message)

        else:
            
            return jsonify(
                category=sentence_pred,
            )

@app.route("/job_seeker",methods=['GET', 'POST'])
def job_seeker():

    if request.method == "POST":
        custom_category = request.form.get('custom-category')

        all_jobs = job.query.filter(job.category == custom_category).all()
        return render_template('job_seeker.html',title="all_jobs",
                           all_jobs=all_jobs)

    return render_template('job_seeker.html')
