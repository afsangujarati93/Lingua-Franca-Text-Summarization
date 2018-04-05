from rouge import Rouge 
import inspect
from Log_Handler import Log_Handler as lh
import os
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank  import LexRankSummarizer 
from sumy.summarizers.luhn  import LuhnSummarizer  
from sumy.summarizers.text_rank  import TextRankSummarizer 
from sumy.summarizers.sum_basic  import SumBasicSummarizer 
from sumy.summarizers.kl  import KLSummarizer 
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import pandas as pd

process_logger = lh.log_initializer("Logs/process_log.log", True)
error_logger = lh.log_initializer("Logs/error_log.log", False)
def rouge_evaluation(system_summary, model_summary):
	method_name = inspect.stack()[0][3]	
	try:
		process_logger.debug("in "+ method_name +" method")
		rouge = Rouge()
		scores = rouge.get_scores(model_summary, system_summary, avg= True)
		# print("\nScores:" + str(scores))
		return scores
	except Exception as Ex:
		error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex))
		return None


def evaluate_summary(file_name, input_dir, sent_count, lingua_franca_summary, show_summaries):
	method_name = inspect.stack()[0][3]
	try:
		process_logger.debug("in "+ method_name +" method")
		file_model_summary = open(input_dir + file_name +".model", "r")
		model_summary = file_model_summary.read()

		rouge_scores_dict = {}
		rouge_scores = rouge_evaluation(lingua_franca_summary, model_summary)
		rouge_scores_dict[">>LINGUA FRANCA"] = rouge_scores
		file_summary = open("Test System Summary/" + file_name + "-" + "LINGUA FRANCA" + ".txt", "w")
		file_summary.write(lingua_franca_summary)

		LANGUAGE = "english"
		parser = PlaintextParser.from_file(input_dir + file_name + ".txt", Tokenizer(LANGUAGE))
		stemmer = Stemmer(LANGUAGE)
		
		lsa_summarizer = LsaSummarizer(stemmer)
		rouge_scores = sumy_summarizers("LSA", lsa_summarizer, parser.document, sent_count, model_summary, show_summaries, file_name)
		rouge_scores_dict["LSA"] = rouge_scores		

		lex_summarizer = LexRankSummarizer(stemmer)
		rouge_scores = sumy_summarizers("LEX RANK", lex_summarizer, parser.document, sent_count, model_summary, show_summaries, file_name)
		rouge_scores_dict["LEX RANK"] = rouge_scores

		luhn_summarizer = LuhnSummarizer(stemmer)
		rouge_scores = sumy_summarizers("LUHN", luhn_summarizer, parser.document, sent_count, model_summary, show_summaries, file_name)
		rouge_scores_dict["LUHN"] = rouge_scores
		
		text_rank_summarizer = TextRankSummarizer(stemmer)
		rouge_scores = sumy_summarizers("TEXT RANK", text_rank_summarizer, parser.document, sent_count, model_summary, show_summaries, file_name)
		rouge_scores_dict["TEXT RANK"] = rouge_scores
		
		sum_basic_summarizer = SumBasicSummarizer(stemmer)
		rouge_scores = sumy_summarizers("SUM BASIC", sum_basic_summarizer, parser.document, sent_count, model_summary, show_summaries, file_name)
		rouge_scores_dict["SUM BASIC"] = rouge_scores
		
		kl_summarizer = KLSummarizer(stemmer)
		rouge_scores = sumy_summarizers("KL SUM", kl_summarizer, parser.document, sent_count, model_summary, show_summaries, file_name)
		rouge_scores_dict["KL SUM"] = rouge_scores
		
		# score_reader(rouge_scores_dict)
		df_rouge, summarizer_list = process_rouge_scores(rouge_scores_dict)

		return df_rouge, summarizer_list

	except Exception as Ex:
		error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex))
		return None

def sumy_summarizers(summarizer_name, summarizer_object, input_doc, sent_count, model_summary, show_summaries, file_name):
	method_name = inspect.stack()[0][3]
	try:
		process_logger.debug("in "+ method_name +" method")
		summary_sent_list = []		
		
		for sentence in summarizer_object(input_doc, sent_count):
			# print( sentence)
			summary_sent_list.append(sentence)
		output_summary = ''.join(map(str,summary_sent_list))
		if show_summaries:
			print("\n"+ summarizer_name +":")
			print(output_summary)

		file_summary = open("Test System Summary/" + file_name + "-" + summarizer_name + ".txt", "w")
		file_summary.write(output_summary)

		rouge_scores = rouge_evaluation(output_summary, model_summary)
		return rouge_scores

	except Exception as Ex:
		error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex))
		return None

def score_reader(rouge_scores_dict):
	method_name = inspect.stack()[0][3]
	try:
		process_logger.debug("in " + method_name + " method")
		
		print_score_result(rouge_scores_dict, "rouge-1")
		print_score_result(rouge_scores_dict, "rouge-2")
		print_score_result(rouge_scores_dict, "rouge-l")		
		print(df_rouge)

	except Exception as Ex:
		error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex))
		return None

def process_rouge_scores(rouge_scores_dict):
	method_name = inspect.stack()[0][3]
	try:
		process_logger.debug("in "+ method_name +" method")
		df_rows = []
		summarizer_list = []
		i = 0
		for summarizer_name, rouge_scores in rouge_scores_dict.items():	
			rouge1_f = rouge_scores["rouge-1"]["f"]
			rouge1_p = rouge_scores["rouge-1"]["p"]
			rouge1_r = rouge_scores["rouge-1"]["r"]

			rouge2_f = rouge_scores["rouge-2"]["f"]
			rouge2_p = rouge_scores["rouge-2"]["p"]
			rouge2_r = rouge_scores["rouge-2"]["r"]

			rougel_f = rouge_scores["rouge-l"]["f"]
			rougel_p = rouge_scores["rouge-l"]["p"]
			rougel_r = rouge_scores["rouge-l"]["r"]			

			summarizer_list.append(summarizer_name)
			df_rows.append([rouge1_f, rouge1_p, rouge1_r, rouge2_f, rouge2_p, rouge2_r, rougel_f, rougel_p, rougel_r]) 
			i += 1			

		rouge_columns = [ 	"R1 F1-score", "R1 Precision", "R1  Recall", 
							"R2 F1-score", "R2 Precision", "R2  Recall", 
							"RL F1-score", "RL Precision", "RL  Recall", 
						]
		df_rouge = pd.DataFrame(df_rows, columns = rouge_columns)


		return df_rouge, summarizer_list
	except Exception as Ex:
		error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex))
		return None

def print_score_result(rouge_scores_dict, rouge_key):
	method_name = inspect.stack()[0][3]
	try:
		process_logger.debug("in "+ method_name +" method")

		print("\n")
		print("{0:<60}".format("=============================================================================================="))
		print("{0:<20} {1:<20} {2:<20}".format("", "", rouge_key.upper()))
		print("{0:<60}".format("=============================================================================================="))
		print("{0:<20} {1:<20} {2:<20} {3:<20}".format("Summarizer Name", "F1-score", "Precision", "Recall"))
		print("{0:<60}".format("=============================================================================================="))
		for summarizer_name, rouge_scores in rouge_scores_dict.items():	
			rouge_f = rouge_scores[rouge_key]["f"]
			rouge_p = rouge_scores[rouge_key]["p"]
			rouge_r = rouge_scores[rouge_key]["r"]

			print("{0:<20} {1:<20} {2:<20} {3:<20}".format(summarizer_name, str(rouge_f), str(rouge_p), str(rouge_r)))
			# print(summarizer_name + "\t" + str(rouge1_f) + "\t" + str(rouge1_p) + "\t" + str(rouge1_r))

	except Exception as Ex:
		error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex))
		return None

