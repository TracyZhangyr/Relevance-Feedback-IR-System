# COMS6111 Project 1

#### a. Team Members:

Yuerong Zhang, 
Ruixuan Fu



#### b. List of Files:

1. proj1
   1. main.py
   2. proj1-stop.txt
   
2. README.md

3. transcript.txt

   


#### c. Commands:

1. Run program:

   ```shell
   python3 main.py <google api key> <google engine id> <precision> <query>
   ```

   Sample commands:

   ```shell
   python3 main.py <google api key> <google engine id> 0.9 "per se"
   python3 main.py <google api key> <google engine id> 0.9 "brin"
   python3 main.py <google api key> <google engine id> 0.9 "cases"
   ```

2. Install dependencies:

   ```shell
   pip3 install --upgrade google-api-python-client
   ```

   ```shell
   pip3 install -U scikit-learn
   ```

   Our environment:

   ```shell
   sklearn: 1.0.2
   python: 3.9.10
   ```



#### d. Project Description

1. Internal design:

   1. General structure:
      1. Receive and check user's input
      2. Begin the main loop, for each iteration:
         1. Retrieve the top-10 results for the query from Google
         2. Terminate the program if search results < 10 in the first iteration
         3. Present the top-10 results to user and let them mark relevance, and get relevant docs and irrelevant docs index lists, as well as valid docs (exclude non-html files)
         4. If current precision >= target precision or = 0, stop program. Otherwise, derive at most 2 new words and add to the current query. After that, put the modified query into the next iteration. 


   2. Main components:
      1. callGoogleAPI(): through calling the google search API, fetch the top 10 search results according to user's query input.
      1. getSearchFeedback(): show the results' main info (url, title, summary) we fetched in component 1 one by one, then get user's feedback to each result's relavence by entering "Y" or "N". Build sets of relevant docs, unrelevant docs and valid docs.
      1. preprocess_docs(): reformat valid search results into documents = [doc1(str), doc2(str), ..., docn(str)] for later calculation. 
      1. tfidf(): calculate TF-IDF using sklearn's TfidfVectorizer, and compute query vector, document vectors, and word_IDF table. 
      1. Rocchio(): through applying the Rocchio algorithm, update each query vector's relevance weight, which will be used in iterations to augment query words.
      1. main(): the main loop to improve Google search results using the user-provided relevance feedback.

2. External libraries:

   1. googleapiclient: 

      We use this to call Google Custom Search Engine API to get top-10 search results for the query.

   2. sklearn: 

      We use sklearn's TfidfVectorizer to calculate TF-IDF, query vector, and document vectors.




#### e. Query-modification Method

First, we preprocess top-10 search resutls to construct a corpus contains all valid documents, each document is a string combination of search results' titles and snippets. Then we use sklearn's TfidfVectorizer to compute TD-IDF, query vector, document vectors, and word_IDF table. With marked relevant docs and query vectors,  we then use the Rocchio algorithm to modify query vector's in a direction closer to related docs or farther away non-related docs, which is reflected in its value's increasing or decreasing.
Through this way, we get a new set of updated weight value and their corresponding order number in the original set (like (0.09652, 4)), that can be used to find the real word in the word_IDF table words set (like "keller"). By sorting these weight values by decreasing order and filtering the words that have been included in current queries, we pick up two keywords with top-2 highest relevancy weights, as the new query keywords to add in the next round. We find this query expansion method works well in practice. 



#### f. Google Custom Search Engine JSON API Key & Engine ID

```shell
engine_ID = <your engine ID>
API_Key = <your API key>
```



#### g. Additional Information

1. Handling non-html files:

   In the top-10 results for the query, we simply ignore non-html files and will not show them to the user. 
   
2. Reorder:
   In practice, we find simply add our derived new words to the end of the current query is doing well enough for the next iteration search.

