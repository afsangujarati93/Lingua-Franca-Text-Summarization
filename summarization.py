import inspect
from Log_Handler import Log_Handler as lh
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import gutenberg
from nltk.corpus import brown
from nltk.tag import pos_tag
import nltk
import pandas as pd
from evaluation import evaluate_summary
import os
import pathlib
import csv
import matplotlib

# nltk.download('corpora/stopwords')

process_logger = lh.log_initializer("Logs/process_log.log", True)
error_logger = lh.log_initializer("Logs/error_log.log", False)
# method_name = inspect.stack()[0][3]
# try:
#	process_logger.debug("in "+ method_name +" method")
# except Exception as Ex:
# 	error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex))
# 	return None

class Preprocess_Prop:
  def __init__(self, tokenized_sentences, tokenized_word_sentences, 
				processed_word_list, avg_word_count, total_word_count, title_words_list,
				word_frequency_dict):
    self.tokenized_sentences = tokenized_sentences
    self.tokenized_word_sentences = tokenized_word_sentences
    self.processed_word_list = processed_word_list
    self.avg_word_count = avg_word_count
    self.total_word_count = total_word_count
    self.title_words_list = title_words_list
    self.word_frequency_dict = word_frequency_dict

def summarization_initializer(input_dir, sent_count, lang_word_path, lang_rel_flag, show_summaries, summary_case, lang_rel_file = ''):
	method_name = inspect.stack()[0][3]
	try:
		process_logger.debug("in "+ method_name +" method")

		scores_df_list = []

		file_count = len(list(pathlib.Path(input_dir).glob('*.txt')))
		txt_file_counter = 1
		for file in os.listdir(input_dir):			
			if file.endswith(".txt"):
				print(">" + str(txt_file_counter) + " out of " + str(file_count), end="\r")
				txt_file_counter += 1
				# print("File name:" + str(file))
				input_file_name = file[:-4]				
				process_logger.info("Processing input text file:" + str(file))
				input_data_file = open(input_dir + file, "r", encoding="utf8") 
				input_data_txt = input_data_file.read()

				print("after reading: " + file)

				title_file_path = input_dir + input_file_name +'.title'
				if os.path.isfile(title_file_path):
					process_logger.info("Title file name:" + str(title_file_path))
					input_title_file = open(title_file_path, "r", encoding="utf8")
					input_title_txt = input_title_file.read()										
				else: 
					process_logger.info("No title file found for file name:" + str(input_file_name))
					error_logger.debug("No title file found for file name:" + str(input_file_name))
					input_title_txt = ""

				lingua_franca_summary = summarization_process(input_file_name, input_data_txt, input_title_txt, sent_count, lang_word_path, lang_rel_flag)
				summarizer_list = []
				if summary_case == 1:
					df_rouge, summarizer_list = evaluate_summary(input_file_name, input_dir, sent_count, lingua_franca_summary, show_summaries)
					scores_df_list.append(df_rouge)
				elif summary_case == 2:
					# print("\nSystem Summary:\n" + str(lingua_franca_summary))
					file_summary = open("System Summary/" + input_file_name + "-" + lang_rel_file + ".txt", "w")
					file_summary.write(lingua_franca_summary)
				else: return

		return scores_df_list, summarizer_list
		
	except Exception as Ex:
		error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex))
		return None

def summarization_process(file_name, input_data_txt, input_title_txt, sent_count, lang_word_path, lang_rel_flag):
	method_name = inspect.stack()[0][3]
	try:
		df_features = pd.DataFrame()
		pp_returned_data = preprocess_data(file_name,input_data_txt, input_title_txt)

		df_features["tokenized_sentences"] = pp_returned_data.tokenized_sentences

		df_features["relative_position"] = relative_position(file_name, pp_returned_data.tokenized_sentences)
		
		df_features["proper_nouns"] = proper_nouns(file_name, pp_returned_data.tokenized_sentences)

		#considering edges with at least certain % of the avg number of words in a sentence
		bushy_edge_threshold = round(pp_returned_data.avg_word_count * (30/100))
		# print("Average:" + str(pp_returned_data.avg_word_count) + "| bushy threshold:" + str(bushy_edge_threshold))
		df_features["bushy_wthres"], df_features["bushy_wothres"] = bushy_path(file_name, pp_returned_data.tokenized_word_sentences, bushy_edge_threshold)

		df_features["similarity_score"] = sent_document_similarity(file_name, pp_returned_data.tokenized_word_sentences, pp_returned_data.total_word_count)

		df_features["title_relevance"] = title_relevance(file_name, pp_returned_data.tokenized_word_sentences, pp_returned_data.title_words_list)

		df_features["relative_length"] = relative_sent_length(file_name, pp_returned_data.tokenized_word_sentences, pp_returned_data.avg_word_count)

		df_features["sentence_freq"] = word_frequency(file_name, pp_returned_data.tokenized_word_sentences, pp_returned_data.word_frequency_dict)

		df_features["numeric_data"] = numeric_data(file_name, pp_returned_data.tokenized_word_sentences)

		if lang_rel_flag == "y":
			df_features["language_relevance"] = language_relevance(file_name, pp_returned_data.tokenized_word_sentences, lang_word_path)
		
		selected_sentences = process_features(file_name, df_features, sent_count, True)

		lingua_franca_summary = ''.join(map(str,selected_sentences))

		return lingua_franca_summary

	except Exception as Ex:
		error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex) + "| File name:" + str(file_name))
		return None


def preprocess_data(file_name, input_data_txt, input_title_txt):
	method_name = inspect.stack()[0][3]
	try:
		process_logger.debug("in "+ method_name+" method")
		ps = PorterStemmer()		
		tokenized_sentences = sent_tokenize(input_data_txt)
		process_logger.info("tokenized_sentences:" + str(tokenized_sentences) + "\n sentences length:" + str(len(tokenized_sentences)))
		stopWords = set(stopwords.words('english'))
		
		title_words = [i for i in word_tokenize(input_title_txt.lower()) if i not in stopWords and (i.isalpha() or i.isnumeric())]
		title_words_list = []
		for word in title_words:
			title_words_list.append(ps.stem(word))	

		avg_word_count = 0
		total_word_count = 0
		tokenized_word_sentences = []
		for sentence in tokenized_sentences:
			temp_stemmed_word_list = []
			#Loop through the tokenized version of sentences with are changed to lower case and 
			#than check if any of these words is not a part of the stop words set and also check if it is a alphabet and now punctuation and stuff
			#after that such value i is returned back to words
			words = [i for i in word_tokenize(sentence.lower()) if i not in stopWords and (i.isalpha() or i.isnumeric())]
			total_word_count += len(words)			
			for word in words:
				temp_stemmed_word_list.append(ps.stem(word))
			tokenized_word_sentences.append(temp_stemmed_word_list)			

		avg_word_count = round(total_word_count/len(tokenized_sentences))				
		process_logger.info("tokenized sentences with tokenized words in them:" + str(tokenized_word_sentences) + "| average word count:" + str(avg_word_count))

		words = [i for i in word_tokenize(input_data_txt.lower()) if i not in stopWords and (i.isalpha() or i.isnumeric())]		
		processed_word_list = []
		for word in words:
			processed_word_list.append(ps.stem(word))		    
		process_logger.info("processed word list after stemming:" + str(processed_word_list) + "\n word list length:" +  str(len(processed_word_list)))
		
		# Calculate frequency distribution
		word_frequency_dict = nltk.FreqDist(processed_word_list)
		
		return Preprocess_Prop(tokenized_sentences, tokenized_word_sentences, processed_word_list, avg_word_count, total_word_count, 
								title_words_list, word_frequency_dict)

	except Exception as Ex:
		error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex) + "| File name:" + str(file_name))
		return None

#assigning the first five sentences of the passage a value reducing from 5 to 1 and 
# last five sentences a value reducing from 5 for the last to 1 to the fifth to last
#all other sentences has a value of 0 (Step 1)
def relative_position(file_name, tokenized_sentences):
	method_name = inspect.stack()[0][3]
	try:
		process_logger.debug("in "+ method_name+" method")
		len_tok_sent = len(tokenized_sentences)
		start_range = range(0,5)
		end_range = range(len_tok_sent-5,len_tok_sent)
		lst_relative_position = []
		for index, sentence in enumerate(tokenized_sentences):
			if tokenized_sentences.index(sentence) in start_range:
				lst_relative_position.append(5 - (index))
			elif tokenized_sentences.index(sentence) in end_range:
				lst_relative_position.append(5 - (len_tok_sent - (index+1)))
			else:
				lst_relative_position.append(0)

		lst_relative_position = normalize_list(method_name, lst_relative_position)
		process_logger.info("relative position list:" + str(lst_relative_position))
		return lst_relative_position		
	except Exception as Ex:
		error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex) + "| File name:" + str(file_name))
		return None

#finding the sentences with proper nouns i.e. named entities and 
#creating a list with values corresponding to the number of proper nouns in a sentence (Step 2)
def proper_nouns(file_name, tokenized_sentences):
	method_name = inspect.stack()[0][3]
	try:
		process_logger.debug("in "+ method_name+" method")
		lst_proper_nouns = []
		for sentence in tokenized_sentences:
			propernouns = [word for word,pos in pos_tag(sentence.split()) if pos == 'NNP']
			lst_proper_nouns.append(len(propernouns))
		
		lst_proper_nouns = normalize_list(method_name, lst_proper_nouns)
		process_logger.info("proper noun list: " + str(lst_proper_nouns))
		return lst_proper_nouns		
	except Exception as Ex:
		error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex) + "| File name:" + str(file_name))
		return None

#bushy path method where the relation between a setence and another sentence is found and a score
# is given to each sentence accordingly. The calculation is done either by considering the threshold score
# or for all the sentence relation. (Step 3 and 5)
def bushy_path(file_name, tokenized_word_sentences, bushy_edge_threshold):
	method_name = inspect.stack()[0][3]	
	try:
		process_logger.debug("in "+ method_name+" method" + "| bushy threshold:" + str(bushy_edge_threshold))
		len_tok_word_sent = len(tokenized_word_sentences)
		words_com_thres_mat = np.empty([len_tok_word_sent,len_tok_word_sent])
		words_com_wothres_mat = np.empty([len_tok_word_sent,len_tok_word_sent])
		
		for i, sentence in enumerate(tokenized_word_sentences):
			words_com_thres_mat[i,i] = 0
			words_com_wothres_mat[i,i] = 0
			for j in range (i+1,len_tok_word_sent):
				common_words_list = list(set(tokenized_word_sentences[i]) & set(tokenized_word_sentences[j]))
				if len(tokenized_word_sentences[i]) == len(tokenized_word_sentences[j]) == 0:
					bushy_score = 0
				elif len(tokenized_word_sentences[i]) >= len(tokenized_word_sentences[j]):
					bushy_score = len(common_words_list)/len(tokenized_word_sentences[i])
				else:
					bushy_score = len(common_words_list)/len(tokenized_word_sentences[j])
				
				#another matrix has the value of connected edges 
				words_com_wothres_mat[i,j] = words_com_wothres_mat[j,i] = bushy_score
				
				# checking is the number of common words are more than the set threshold, or else the score would be 0
				#this is as good as neglecting the edges that are not greater than the set threshold
				if (len(common_words_list) >= bushy_edge_threshold):
					words_com_thres_mat[i,j] = words_com_thres_mat[j,i] = bushy_score
				else:					
					words_com_thres_mat[i,j] = words_com_thres_mat[j,i] = 0
		
		busy_thres_score_list = words_com_thres_mat.sum(axis=1)	
		busy_wothres_score_list = words_com_wothres_mat.sum(axis=1)	

		busy_thres_score_list = normalize_list(method_name, busy_thres_score_list)
		busy_wothres_score_list = normalize_list(method_name, busy_wothres_score_list)		
		process_logger.info("After calculating final per sentence score: " + str(busy_thres_score_list) + "|\n\n without threshold\n" + str(busy_wothres_score_list))
		return busy_thres_score_list, busy_wothres_score_list

	except Exception as Ex:
		error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex) + "| File name:" + str(file_name))
		return None

# This method finds the similarity of a sentence in the document with the rest of the 
# document. Basically it finds the common words between a sentence and rest of the document (Step 4)
def sent_document_similarity(file_name, tokenized_word_sentences, total_word_count):
	method_name = inspect.stack()[0][3]
	try:
		similarity_score_list = []
		process_logger.debug("in "+ method_name+" method")
		len_tok_word_sent = len(tokenized_word_sentences)
		for i, sentence in enumerate(tokenized_word_sentences):
			rest_document_list = []
			for j in range (0,len_tok_word_sent):
				if i != j:
					rest_document_list = rest_document_list + tokenized_word_sentences[j]					
			
			similar_word_doc = list(set(tokenized_word_sentences[i]) & set(rest_document_list))
			# process_logger.info("\nSimilarity so far: i =>" + str(i) + "| Actual Length:" + str(len(sentence)) + str(similar_word_doc))
			similar_word_score = len(similar_word_doc)/total_word_count
			similarity_score_list.append(similar_word_score)			

		similarity_score_list = normalize_list(method_name, similarity_score_list)
		process_logger.info("Sentence Similarity socre:" + str(similarity_score_list))
		return similarity_score_list
	except Exception as Ex:
		error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex) + "| File name:" + str(file_name))
		return None

# This method estimates the relevance of title in relation to given sentences
# A sentence is considered important if it has words that appear in the title (Step 6)
def title_relevance(file_name, tokenized_word_sentences, title_words_list):
	method_name = inspect.stack()[0][3]
	try:
		process_logger.debug("in "+ method_name+" method")
		title_rel_list = []		
		for i, sentence in enumerate(tokenized_word_sentences):
			similar_title_doc = list(set(sentence) & set(title_words_list))

			total_unique_list = title_words_list + list(set(sentence) - set(title_words_list))
			# print("title common list:" + str(similar_title_doc) + "| total_unique_list:" + str(total_unique_list))
			if len(total_unique_list) == 0:
				title_sent_score = 0
			else:
				title_sent_score = len(similar_title_doc)/len(total_unique_list)
			title_rel_list.append(title_sent_score)			

		title_rel_list = normalize_list(method_name, title_rel_list)
		process_logger.info("title relevance score list:" + str(title_rel_list))
		return title_rel_list

	except Exception as Ex:
		error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex) + "| File name:" + str(file_name))
		return None

# This method gives score on basis of the relative length of each sentence 
# and the average length of sentence (Step 7)
def relative_sent_length(file_name, tokenized_word_sentences, avg_word_count):
	method_name = inspect.stack()[0][3]
	try:
		process_logger.debug("in "+ method_name+" method")
		relative_len_list = []
		for i, sentence in enumerate(tokenized_word_sentences):
			relative_len_score = len(sentence) * avg_word_count
			relative_len_list.append(relative_len_score)
			# print("relative socre:" + str(relative_len_score))

		relative_len_list = normalize_list(method_name, relative_len_list)
		process_logger.info("relative length score:" + str(relative_len_list))
		return relative_len_list
	except Exception as Ex:
		error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex) + "| File name:" + str(file_name))
		return None

# This method score every sentence on basis of the frequency of words in a sentence 
# as per the whole document. (Step 8)
def word_frequency(file_name, tokenized_word_sentences, word_frequency_dict):
	method_name = inspect.stack()[0][3]
	try:
		process_logger.debug("in "+ method_name+" method")
		sent_freq_list = []
		for i, sentence in enumerate(tokenized_word_sentences):
			sent_length = len(sentence)
			sent_score = 0 
			for word in sentence:				
				sent_score += word_frequency_dict[word]
			if sent_length == 0:
				sent_freq_score
			else:
				sent_freq_score = sent_score/sent_length
			sent_freq_list.append(sent_freq_score)
			# print("sent freq score:" + str(sent_freq_score) + "\n\n")

		sent_freq_list = normalize_list(method_name, sent_freq_list)
		process_logger.info("Frequency of sentence in word:" + str(sent_freq_list))
		return sent_freq_list

	except Exception as Ex:
		error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex) + "| File name:" + str(file_name))
		return None

# This method scores the sentece 1 if it has numeric value
# else it is given a value of 0 (Step 10)
def numeric_data(file_name, tokenized_word_sentences):
	method_name = inspect.stack()[0][3]
	try:
		process_logger.debug("in "+ method_name+" method")		
		numeric_data_list = []
		for i, sentence in enumerate(tokenized_word_sentences):
			has_numeric = False
			for word in sentence:				
				if(word.isnumeric()):
					has_numeric = True
					break

			if has_numeric : numeric_data_list.append(1)
			else : numeric_data_list.append(0)			

		numeric_data_list = normalize_list(method_name, numeric_data_list)
		process_logger.info("Scoring sentences on basis of numeric content:" + str(numeric_data_list))		
		return numeric_data_list
	except Exception as Ex:
		error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex))
		return None

# This is the actual test feature that we introduced to consider the linguistic impact on summarization
# It compares every sentence with a list of words from a processed source (which has words which are more relevant to a particular region)
# Sentences with words from this list are given a bonus score. 
# The score is calculated by finding the common words and dividing by the number of words in the sentence and than normalized between 0 and 1
def language_relevance(file_name, tokenized_word_sentences, lang_word_path):
	method_name = inspect.stack()[0][3]
	try:
		process_logger.debug("in "+ method_name +" method")
		lang_rel_list = []
		
		language_relevant_words = []
		# for file in os.listdir(lang_word_dir):
		# 	if file.endswith(".csv"):
		# 		with open(lang_word_dir + file, 'r', encoding="utf8") as my_file:
		# 		    reader = csv.reader(my_file)
		# 		    language_relevant_words.append(list(reader))
		with open(lang_word_path + ".csv", 'r', encoding="utf8") as my_file:
				reader = csv.reader(my_file)
				language_relevant_words.append(list(reader))
				    
		language_relevant_words = list(matplotlib.cbook.flatten(language_relevant_words))

		for index, sentence in enumerate(tokenized_word_sentences):
			common_lang_words = list(set(sentence) & set(language_relevant_words))
			# total_unique_list = language_relevant_words + list(set(sentence) - set(language_relevant_words))
			if len(sentence) == 0:
				title_sent_score = 0
			else:
				title_sent_score = len(common_lang_words)/len(sentence)
			lang_rel_list.append(title_sent_score)			

		lang_rel_list = normalize_list(method_name, lang_rel_list)
		process_logger.info("Language Relevance score list:" + str(lang_rel_list))
		return lang_rel_list

	except Exception as Ex:
		error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex) + "| File name:" + str(file_name))
		return None

#This method normalizes the list of numbers between 0 and 1
def normalize_list(feature_name, list_to_normalize):
	method_name = inspect.stack()[0][3]
	try:
		normalized_list = []
		process_logger.debug("in "+ method_name +" method " + "| Actual List:" + str(list_to_normalize))
		
		max_val = max(list_to_normalize)
		min_val = min(list_to_normalize)

		if max_val == min_val == 0:
			# print("normalied ZERO list:" + str(list_to_normalize))
			process_logger.info("feature name: " + str(feature_name) +"|normalied ZERO list:" + str(list_to_normalize))
			return list_to_normalize
		elif max_val == min_val:
			for val in list_to_normalize:
				z_val = 1
				normalized_list.append(z_val) 
			process_logger.info("feature name: " + str(feature_name) +"|normalied SAME NUMBER list::" + str(list_to_normalize))
			# print("normalied SAME NUMBER list:" + str(normalized_list))
			return normalized_list
		else:
			for val in list_to_normalize:
				z_val = (val - min_val)/ (max_val - min_val)
				normalized_list.append(z_val)
			process_logger.info("feature name: " + str(feature_name) +"|normalied list::" + str(list_to_normalize))
			return normalized_list
		

	except Exception as Ex:
		error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex) + "| File name:" + str(file_name))
		return None

def process_features(file_name, df_features, sent_num_select, export_csv):
	method_name = inspect.stack()[0][3]
	try:
		process_logger.debug("in " + method_name + " method")

		#adding the features
		df_features["features_sum"] = df_features.sum(axis = 1)

		#sorting the sentences as per the feature sume
		df_features = df_features.sort_values(by = "features_sum", ascending = False)
		
		#resetting the index for the sentences can be selected on basis of index
		df_features = df_features.reset_index()		

		if(export_csv):
			df_features.to_csv("Summary Analysis/"+file_name+"-summaryanalysis.csv")
		#selecting the sentences on basis of the input, which would later be joined to make a summary
		# selected_sentences = df_features.loc[:(sent_num_select-1)]["tokenized_sentences"]
		df_features = df_features.loc[:(sent_num_select-1)].sort_values(by = "index")
		
		selected_sentences = df_features["tokenized_sentences"]
		process_logger.info("Selected sentences :" + str(selected_sentences))
		
		return selected_sentences

	except Exception as Ex:
		error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex) + "| File name:" + str(file_name))
		return None



