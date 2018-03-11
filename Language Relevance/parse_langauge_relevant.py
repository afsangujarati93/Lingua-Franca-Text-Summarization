import sys
sys.path.append('..')
from Log_Handler import Log_Handler as lh
import inspect
import os
import json 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import csv

process_logger = lh.log_initializer("../Logs/process_log.log", True)
error_logger = lh.log_initializer("../Logs/error_log.log", False)
# method_name = inspect.stack()[0][3]
# try:
#	process_logger.debug("in "+ method_name +" method")
# except Exception as Ex:
# 	error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex))
# 	return None



def language_relevance_data():
	method_name = inspect.stack()[0][3]
	try:
		process_logger.debug("in "+ method_name +" method")
		input_dir = "Input Data/"		
		for file in os.listdir(input_dir):
			input_file_name = file[:-5]	
			input_data_file = open(input_dir + file, "r", encoding="utf8") 
			input_data_txt = input_data_file.read()
			language_relevant_words = process_language_data(input_data_txt)

			with open("../Language Words File/"+input_file_name + ".csv","w") as myfile:
			    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
			    wr.writerow(language_relevant_words)
			    print("Process complete")

	except Exception as Ex:
		# error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex))
		print("Exception occurred in " + method_name + "| Exception:" + str(Ex))
		return None


def process_language_data(input_data_txt):
	method_name = inspect.stack()[0][3]
	try:
		process_logger.debug("in "+ method_name +" method")
		json_lang_data = json.loads(input_data_txt)
		
		language_relevant_words = []

		stopWords = set(stopwords.words('english'))
		ps = PorterStemmer()
		for per_feed in json_lang_data:
			feed_text = per_feed["text"]			
			language_relevant_words = preprocess_text(feed_text, language_relevant_words, stopWords, ps)			
		
		return language_relevant_words
	except Exception as Ex:
		# error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex))
		print("Exception occurred in " + method_name + "| Exception:" + str(Ex))
		return None

def preprocess_text(input_text, language_relevant_words, stopWords, ps):
	method_name = inspect.stack()[0][3]
	try:
		process_logger.debug("in "+ method_name +" method")
		if not input_text:
			return language_relevant_words		
		
		# No numbers (or i.isnumeric())
		words = [i for i in word_tokenize(input_text[0].lower()) if i not in stopWords and i.isalpha() ]		

		for word in words:			
			if word not in language_relevant_words:
				language_relevant_words.append(ps.stem(word))

		return language_relevant_words
	except Exception as Ex:
		# error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex))
		print("Exception occurred in " + method_name + "| Exception:" + str(Ex))
		return None

if __name__ == '__main__':
	language_relevance_data()