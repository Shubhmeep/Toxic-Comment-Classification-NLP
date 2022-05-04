# Core Pkgs
from click import Choice
import streamlit as st 
import altair as alt
import plotly.express as px 
import string
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer
snow_stemmer = SnowballStemmer(language='english')
from nltk.corpus import words
import operator
import nltk as nl
# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime
from libraries import *
# Utils
import joblib 
pipe_lr = joblib.load(open("models/model.pkl","rb"))


cuss_word_array_uppercase = ["Fuck","Fuck you","Ass","Shit","Piss off","Dick head","Asshole","Son of a bitch","Bastard","Bitch","Damn","Cunt","Bollocks","Bugger"
                                                            ,"Bloody Hell","Choad","Crikey","Rubbish","Shag","Wanker","Taking the piss","Twat","Bloody Oath"
                                                            , "Arse","Bloody","Bugger","Crap","Damn","Arsehole","Balls","Tits","Boobs","Cock","Dick","Pussy","Cunt","motherfuck","fatherfuck","Nigga"]

cuss_word_array_lowercase = []
for word in cuss_word_array_uppercase:
    cuss_word_array_lowercase.append(word.lower())

# Fxn
def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

def spell_check(word, n=2):
  ngram={}
  total=26**n
  for i in range(total):
    c=''
    k=i
    for j in range(n):
      c=chr(97+(k%26))+c
      k=k//26
      ngram[c]=set()
  lexicon=words.words()
  lexicon=[i.lower() for i in lexicon if i.isalnum()]
  for w in range(len(lexicon)):
    for c in range(0,len(lexicon[w])- n+1):
      ngram[lexicon[w][c:c+n]].add(w)
  freq_dict={}
  for c in range(0, len(word)-n+1):
    for w in ngram[word[c:c+n]]:
      if lexicon[w] not in freq_dict.keys():
        freq_dict[lexicon[w]]=1
      else:
        freq_dict[lexicon[w]]+=1
  top_freq=dict(sorted(freq_dict.items(),key=operator.itemgetter(1),reverse=True)[:5])
  return top_freq


def home():

    st.markdown("<h1 style='text-align: center; color: white;'>Toxic Comment Classification</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white;'>An AI based comment classification application</p>", unsafe_allow_html=True)
    def load_animation(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None

        return r.json()

    animation = load_animation("https://assets2.lottiefiles.com/packages/lf20_rpf5yhjm.json")
    return st_lottie(animation,height=500)




emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮","toxic":"😶‍🌫️"}
import requests

# Main Application
def main():
	
	menu = ["Home","Monitor"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice=="Home":
		home()
		


	elif choice == "Monitor":
		st.title("Let's classify your comment !!")
		with st.form(key='emotion_clf_form'):
			raw_text = st.text_area("Type Here")
			submit_text = st.form_submit_button(label='Submit')
		if submit_text:
			col1,col2  = st.columns(2)
			prediction = predict_emotions(raw_text)
			probability = get_prediction_proba(raw_text)
			
			

			with col1:
				st.info("Original Text")
				st.write(raw_text)
				st.info("Prediction")
				emoji_icon = emotions_emoji_dict[prediction]
				st.write("{}:{}".format(prediction,emoji_icon))
				st.write("Confidence:  {}".format(np.max(probability)))
				if prediction == 'toxic':
					st.error("Your Comment is toxic !!!")

			with col2:
				st.info("Prediction Probability")
				# st.write(probability)
				proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
				# st.write(proba_df.T)
				proba_df_clean = proba_df.T.reset_index()
				proba_df_clean.columns = ["comment","probability"]

				fig = alt.Chart(proba_df_clean).mark_bar().encode(x='comment',y='probability',color='comment')
				st.altair_chart(fig,use_container_width=True)

			if prediction == 'toxic':
			
				#tokenization of words and removing punctuations in the sentence
				x = string.punctuation
				data = word_tokenize(raw_text)
				clean_punct_words = [w for w in data if w not in x]
				clean_words = [w for w in clean_punct_words if w not in stop]

				stemmed = []
				med = {}

				for i in clean_words:
					#apply snowball stemming
					stemmed.append(snow_stemmer.stem(i))

				#print(stemmed)
			
				for j in stemmed:
					cuss = {}
					if j in cuss_word_array_lowercase:
						for dd in spell_check(j).keys():
							cuss[dd] = nl.edit_distance(j, dd, transpositions=True)
							#cuss[dd] = editDistance(j, dd, len(j), len(dd))
							# print(j,"-> ", dd, "\n","The minimum edit distance is:", editDistance(j, dd, len(j), len(dd)))
						#print(cuss)
						med[j] = cuss
				st.write(" ")
				st.write(" ")
				st.info("Candidate list of obscene words along with thier MED")
				st.write(med)

				for k in med.keys():
					#print(k)
					x = min(med[k].values())
					for i in med[k].keys():
						if med[k][i] == x:
							#print(i)
							st.info("Toxic words being replaced !!")
							st.write(k,"➡️",i)
							
							raw_text = raw_text.replace(k,i)
							break
				st.subheader("Corrected comment :")
				st.success(raw_text)
				


			
		
	

if __name__ == '__main__':
	main()












	# create_page_visited_table()
	# create_emotionclf_table()
	# if choice == "Home":
	# 	add_page_visited_details("Home",datetime.now())
	# 	st.subheader("Home-Emotion In Text")

	# 	with st.form(key='emotion_clf_form'):
	# 		raw_text = st.text_area("Type Here")
	# 		submit_text = st.form_submit_button(label='Submit')

	# 	if submit_text:
	# 		col1,col2  = st.beta_columns(2)

	# 		# Apply Fxn Here
	# 		prediction = predict_emotions(raw_text)
	# 		probability = get_prediction_proba(raw_text)
			
	# 		add_prediction_details(raw_text,prediction,np.max(probability),datetime.now())

	# 		with col1:
	# 			st.success("Original Text")
	# 			st.write(raw_text)

	# 			st.success("Prediction")
	# 			emoji_icon = emotions_emoji_dict[prediction]
	# 			st.write("{}:{}".format(prediction,emoji_icon))
	# 			st.write("Confidence:{}".format(np.max(probability)))



	# 		with col2:
	# 			st.success("Prediction Probability")
	# 			# st.write(probability)
	# 			proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
	# 			# st.write(proba_df.T)
	# 			proba_df_clean = proba_df.T.reset_index()
	# 			proba_df_clean.columns = ["emotions","probability"]

	# 			fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
	# 			st.altair_chart(fig,use_container_width=True)



	# elif choice == "Monitor":
	# 	add_page_visited_details("Monitor",datetime.now())
	# 	st.subheader("Monitor App")

	# 	with st.beta_expander("Page Metrics"):
	# 		page_visited_details = pd.DataFrame(view_all_page_visited_details(),columns=['Pagename','Time_of_Visit'])
	# 		st.dataframe(page_visited_details)	

	# 		pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
	# 		c = alt.Chart(pg_count).mark_bar().encode(x='Pagename',y='Counts',color='Pagename')
	# 		st.altair_chart(c,use_container_width=True)	

	# 		p = px.pie(pg_count,values='Counts',names='Pagename')
	# 		st.plotly_chart(p,use_container_width=True)

	# 	with st.beta_expander('Emotion Classifier Metrics'):
	# 		df_emotions = pd.DataFrame(view_all_prediction_details(),columns=['Rawtext','Prediction','Probability','Time_of_Visit'])
	# 		st.dataframe(df_emotions)

	# 		prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
	# 		pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction',y='Counts',color='Prediction')
	# 		st.altair_chart(pc,use_container_width=True)	



	# else:
	# 	st.subheader("About")
	# 	add_page_visited_details("About",datetime.now())


