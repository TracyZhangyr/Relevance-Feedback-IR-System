import sys
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer

# use Prof's proj1-stop.txt for stopwords
with open('proj1-stop.txt', 'r') as f:
    lines = f.readlines()
    STOPWORDS = [line.strip() for line in lines]

# DEBUGGING MODE
DEBUGGING = False

def callGoogleAPI(query):
    """
     Call Google Search API using auth info, 
     and build a service object to search for the query.
     Return the top-10 search results
    """

    service = build("customsearch", "v1",
                    developerKey=client_key)

    res = service.cse().list(
        q=query,
        cx=engine_key,
    ).execute()

    return res['items'][:10]


def getSearchFeedback(google_results):
    """
    Show user the search results of their query one by one.
    Get the feedback about relevance one by one.
    Return relevant docs and irrelevant docs index lists, as well as valid docs (exclude non-html files).
    """ 

    print("Google Search Results:")
    print("======================")

    relevant_docs = []
    irrelevant_docs = []
    valid_docs = []

    for i, result in enumerate(google_results):

        # For non-html files, we ignore them and will not show them to the user
        if 'fileFormat' in result.keys():
            print("Result {}\nThis is an non-html file, we ignore it. Please mark the next file.".format(i+1))
            continue
        
        valid_docs.append(i)

        url = result.get('link')
        title = result.get('title')
        summary = result.get('snippet')

        print("Result ", i + 1)
        print("[")
        print(" URL: ", url)
        print(" Title: ", title)
        print(" Summary: ", summary)
        print("]\n")

        while True:
            relevant_feedback = input("Relevant (Y/N)? ").lower()
            if not (relevant_feedback == "y" or relevant_feedback == "n"):
                print("Please enter a valid response!\n")
            else:
                break

        if relevant_feedback == "y":
            relevant_docs.append(i)
        elif relevant_feedback == "n":
            irrelevant_docs.append(i)
    
    return relevant_docs, irrelevant_docs, valid_docs


def preprocess_docs(valid_results):
    """
    Reformat valid search results into documents = [doc1(str), doc2(str), ..., docn(str)]
    """
    documents = []
    for d in valid_results:
        title = d['title'] if 'title' in d else ''
        snippet = d['snippet'] if 'snippet' in d else ''
        doc = title + ' ' + snippet
        documents.append(doc)
    
    return documents


def tfidf(query, documents):
    """
    Calculate TF-IDF using sklearn TfidfVectorizer
    Return query vector and document vectors (and IDF for word)
    """
    # use stopwords
    vectorizer = TfidfVectorizer(stop_words=STOPWORDS, sublinear_tf = True)

    # toarray() will flatten TF-IDF matrix to sparse matrix
    doc_vectors = vectorizer.fit_transform(documents).toarray()
    query_vector = vectorizer.transform([query]).toarray()[0]

    # {word:idf}
    idf = vectorizer.idf_
    word_idf = dict(zip(vectorizer.get_feature_names_out(), idf))

    return query_vector, doc_vectors, word_idf


def Rocchio(doc_to_vec, query_to_vec, relevent_docs, irrelevant_docs):
    """
    Rocchio algorithm: q_m = alpha * q_0 + beta * sum(D_r)/|D_r| - gamma * sum(D_ur)/|D_ur|
    q_m is the updated vector, q_0 is the previous vector which will be moved to q_m
    D_r is the relevent docs, D_ur is the unrelevent docs
    """

    # parameters for Rocchio algorithm
    alpha = 1 # basic parameter
    beta = 0.75 # the parameter for relevant vector
    gamma = 0.15 # the parameter for not relevant vector

    D_r = len(relevent_docs)
    D_ur = len(irrelevant_docs)
    q_m = [alpha * q for q in query_to_vec]

    m = 0
    for v in doc_to_vec:
        n = 0
        for val in v:
            if m in relevent_docs:
                q_m[n] = q_m[n] + (beta / D_r) * val
            elif m in irrelevant_docs:
                q_m[n] = q_m[n] - (gamma / D_ur) * val
            n = n + 1
        m = m + 1

    return q_m


def main():
    """
    The main loop to improve Google search results using the 
    user-provided relevance feedback
    """

    # 1. Receive and check user's input
    if len(sys.argv) != 5:
        print("Usage: python3 main.py <google api key> <google engine id> <precision> <query>")
        return 
    
    global client_key, engine_key, target_precision

    client_key = sys.argv[1]  # auth info's client key (developerKey)
    engine_key = sys.argv[2]  # auth info's engine key (cx)
    target_precision = float(sys.argv[3])  # the target precision value user entered
    query = sys.argv[4]  # the query user entered

    if target_precision < 0 or target_precision > 1:
        print("Precision should be a number in between 0 and 1!")
        return 
    
    iteration = 1
    while True:
        # info of the current iteration
        print("Parameters:\nClient key  = {}\nEngine key  = {}\nQuery       = {}\nPrecision   = {}".format(client_key,
                                                                                                            engine_key,
                                                                                                            query,
                                                                                                            target_precision))
        # 2. Retrieve the top-10 results for the query from Google
        search_results = callGoogleAPI(query)

        # terminate the program if search results < 10 in the first iteration
        if iteration == 1 and len(search_results) < 10:
            print("There are fewer than 10 results overall for the input query in the first iteration, stop program.")
            break
        

        # 3. Present the top-10 results to user and let them mark relevance,
        # and get relevant docs and irrelevant docs index lists, as well as valid docs (exclude non-html files)
        relevant_docs, irrelevant_docs, valid_docs = getSearchFeedback(search_results)


        # 4.1 If current precision >= target precision or = 0, stop program. 
        # compute current precision
        current_precision = len(relevant_docs)/len(valid_docs)

        # filter the valid docs
        valid_results = [search_results[i] for i in valid_docs] if len(valid_docs) < 10 else search_results
        # reformat the valid docs 
        documents = preprocess_docs(valid_results)
            
        print("======================\nFEEDBACK SUMMARY\nQuery {}\nPrecision {}".format(query,current_precision))

        if current_precision >= target_precision:
            print("Desired precision reached, done")
            break
        
        print("Still below the desired precision of {}\nIndexing results ....\nIndexing results ....".format(target_precision))

        if current_precision == 0:
            print("Augmenting by\nBelow desired precision, but can no longer augment the query")
            break
        
        # 4.2 Otherwise, derive at most 2 new words and add to the current query. 
        #     After that, reorder the modified query in the best possible order and put it into the next iteration. 

        # TF-IDF
        query_vector, doc_vectors, word_idf = tfidf(query, documents)

        # Rocchio algorithm
        current_queries = set()
        new_words = []
        keys = list(word_idf.keys())

        # apply Rocchio to get updated weight values
        # eg: [0.8144098191520823, 0.09652708437438301, 0.8144098191520823 ...]
        updated_vecs = Rocchio(doc_vectors, query_vector, relevant_docs, irrelevant_docs)
        # sort in dereasing order on weight values
        # eg: [(0.8144098191520823, 0), (0.8144098191520823, 1), (0.09652708437438301, 4) ...]
        sorted_vecs = sorted([(weight, i) for i, weight in enumerate(updated_vecs)], key=lambda x: x[0], reverse=True)

        # get current query words
        for i, word in enumerate(word_idf.keys()):
            if query_vector[i] != 0:
                current_queries.add(word)

        # get the top2 highest weight values (eg: (0.8144098191520823, 0), (0.8144098191520823, 1))
        # find its corresponding word (eg: "keller") in keys set
        i = 0
        for weight, number in sorted_vecs:
            key = keys[number]
            if key in current_queries:
                continue
            if i >= 2:
                break
            new_words.append(key)
            i = i + 1
    
        print("Augmenting by  {}".format(' '.join(new_words)))


        # 5. Modify the current query
        query += ' ' + ' '.join(new_words)

        # increase the iteration number by 1, and go to the next iteration
        iteration  += 1


    #debug
    if DEBUGGING:
        print("\n{} iterations executed. Finished.\n".format(iteration))



if __name__ == '__main__':
    main()
