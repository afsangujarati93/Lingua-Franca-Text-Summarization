# Lingua-Franca-Text-Summarization

It's an extrative text summarization technique that creates summaries considering the factor of users origin.
A bag of words(BOW) is created using BBC Sports and another bag using TOI Cricket. The sentence is compared with the selected BOW to give an additional score for user's origin.

## Getting Started
Clone the project and make sure you have installed all the necessary dependencies. 
Necessary dependencies include, installing the required python modules and having data in the necessary folder. 
* Refer the requirements.txt file for the necessary python modules for the project. 

Before explaining what goes in which folder I will explain the basic execution:
1. run python main.py
2. It will ask whether you want to just test the summarizer or create your own summary. Answer it using either 1 or 2 
3. Let's assume you chose 1, it will ask you what the maximum sentence length summary you want to generate for testing. So basically for testing it will create summaries from sentence length 1 up to whatever number you give as the input.
4. After that it will ask you if you want to use the language relevance factor i.e. the orgin of user
5. If you input y, it will ask you to type the name of the language relevant csv file i.e. your bag of words. For this project there is already bbcsports.csv or toicricket.csv, which you can use. 
6. If you choose 2 in step 2 i.e. you choose to simply generate the summary for your input text, it will ask you to enter the number of sentences you want for your summary e.g. 3 sentence summary or 4 sentence summary and so on.

Please refer below to understand the folders usage.
Folder Name	|	Folder Description
--------------------------	|	-------------------------------------------------------------------------------------------------
Final Input Data	|	Folder with all the input txt file and their respective title. The title file has to be with .title extension
Final Results	|	Final results i.e. csv files of precision, recall and f1-score along with the graphs while testing the summarizer
Language Relvance	|	Contains the script that converts json (which was scraped from the news websites) to csv file of all comma separated words
Language Words File	|	.csv files with the comma separated words. For this project it is bbcsports and toicricket
Logs	|	For apparent reasons it contains Log files
Summary Analysis	|	csv files which describes the scores given to each sentence for each feature, along with the final score. This is the crux of the whole project.
System Summary	|	Output summary if you simple decide to generate summaries. 
Test Input Data	|	All the input summary that would be used for testing along with their model summaries and title (if exists)
Test System Summary	|	Output summary in case of testing.

### And coding style tests

In case if you plan to contribute please consider the follow things:
* Make sure one function only does one thing
* use the template of method given below: 
method_name = inspect.stack()[0][3]
try:
process_logger.debug("in "+ method_name +" method")
except Exception as Ex:
 	error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex))
 	return None
* if something's complicated, leave a comment. 


## Authors 
*(Names are mentioned in lexical order)*

* **Afsan Gujarati** - *(afsan.gujarati@gmail.com)* 
* **Hari Ramesh**
* **Stacey Taylor**

## Acknowledgments

* We referred the work done by https://ieeexplore.ieee.org/document/7045732/ by making modifications and adding our feature on top of it
* Paul Tardy (https://github.com/pltrdy) helped with the queries regarding the rouge implementation.
