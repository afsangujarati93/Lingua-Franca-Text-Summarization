---


---

<h1 id="lingua-franca-text-summarization">Lingua-Franca-Text-Summarization</h1>
<p>It’s an extrative text summarization technique that creates summaries considering the factor of users origin.<br>
A bag of words(BOW) is created using BBC Sports and another bag using TOI Cricket. The sentence is compared with the selected BOW to give an additional score for user’s origin.</p>
<h2 id="getting-started">Getting Started</h2>
<p>Clone the project and make sure you have installed all the necessary dependencies.<br>
Necessary dependencies include, installing the required python modules and having data in the necessary folder.</p>
<ul>
<li>Refer the requirements.txt file for the necessary python modules for the project.</li>
</ul>
<p>Before explaining what goes in which folder I will explain the basic execution:</p>
<ol>
<li>run python <a href="http://main.py">main.py</a></li>
<li>It will ask whether you want to just test the summarizer or create your own summary. Answer it using either 1 or 2</li>
<li>Let’s assume you chose 1, it will ask you what the maximum sentence length summary you want to generate for testing. So basically for testing it will create summaries from sentence length 1 up to whatever number you give as the input.</li>
<li>After that it will ask you if you want to use the language relevance factor i.e. the orgin of user</li>
<li>If you input y, it will ask you to type the name of the language relevant csv file i.e. your bag of words. For this project there is already bbcsports.csv or toicricket.csv, which you can use.</li>
<li>If you choose 2 in step 2 i.e. you choose to simply generate the summary for your input text, it will ask you to enter the number of sentences you want for your summary e.g. 3 sentence summary or 4 sentence summary and so on.</li>
</ol>
<p>Please refer below to understand the folders usage.</p>

<table>
<thead>
<tr>
<th>Folder Name</th>
<th>Folder Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>Final Input Data</td>
<td>Folder with all the input txt file and their respective title. The title file has to be with .title extension</td>
</tr>
<tr>
<td>Final Results</td>
<td>Final results i.e. csv files of precision, recall and f1-score along with the graphs while testing the summarizer</td>
</tr>
<tr>
<td>Language Relvance</td>
<td>Contains the script that converts json (which was scraped from the news websites) to csv file of all comma separated words</td>
</tr>
<tr>
<td>Language Words File</td>
<td>.csv files with the comma separated words. For this project it is bbcsports and toicricket</td>
</tr>
<tr>
<td>Logs</td>
<td>For apparent reasons it contains Log files</td>
</tr>
<tr>
<td>Summary Analysis</td>
<td>csv files which describes the scores given to each sentence for each feature, along with the final score. This is the crux of the whole project.</td>
</tr>
<tr>
<td>System Summary</td>
<td>Output summary if you simple decide to generate summaries.</td>
</tr>
<tr>
<td>Test Input Data</td>
<td>All the input summary that would be used for testing along with their model summaries and title (if exists)</td>
</tr>
<tr>
<td>Test System Summary</td>
<td>Output summary in case of testing.</td>
</tr>
</tbody>
</table><h3 id="and-coding-style-tests">And coding style tests</h3>
<p>In case if you plan to contribute please consider the follow things:</p>
<ul>
<li>Make sure one function only does one thing</li>
<li>use the template of method given below:</li>
</ul>
<pre><code>method_name = inspect.stack()[0][3]
try:
process_logger.debug("in "+ method_name +" method")
except Exception as Ex:
 	error_logger.error("Exception occurred in " + method_name + "| Exception:" + str(Ex))
 	return None
</code></pre>
<ul>
<li>if something’s complicated, leave a comment.</li>
</ul>
<h2 id="authors">Authors</h2>
<p><em>(Names are mentioned in lexical order)</em></p>
<ul>
<li><strong>Afsan Gujarati</strong> - <em>(<a href="mailto:afsan.gujarati@gmail.com">afsan.gujarati@gmail.com</a>)</em></li>
<li><strong>Hari Ramesh</strong>  - <em>(<a href="mailto:hariharanramesh10@gmail.com">hariharanramesh10@gmail.com</a>)</em></li>
<li><strong>Stacey Taylor</strong> - <em>(<a href="mailto:st712976@dal.ca">st712976@dal.ca</a>)</em></li>
</ul>
<h2 id="acknowledgments">Acknowledgments</h2>
<ul>
<li>We referred the work done by <a href="https://ieeexplore.ieee.org/document/7045732/">https://ieeexplore.ieee.org/document/7045732/</a> by making modifications and adding our feature on top of it</li>
<li>Paul Tardy (<a href="https://github.com/pltrdy">https://github.com/pltrdy</a>) helped with the queries regarding the rouge implementation.</li>
<li><a href="http://README.md">README.md</a> template published using (<a href="https://stackedit.io/app#">https://stackedit.io/app#</a>) – Cheers</li>
</ul>

