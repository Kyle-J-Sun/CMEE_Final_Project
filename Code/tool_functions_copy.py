from numpy import array
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from imp import reload

import pymysql
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from zhon import hanzi
import string
from nltk.corpus import stopwords
import nltk
import time as t

import warnings
import sys
# To ignore all warnings that arise here to enhance clarity
warnings.filterwarnings('ignore')

def remove_punctuation(s):
    """ Removing punctuation """
    try:  # python 2.x
        text = s.translate(None, string.punctuation)
        text = s.translate(None, hanzi.punctuation)
    except:  # python 3.x
        translator = s.maketrans('', '', string.punctuation)
        text = s.translate(translator)
        trans_zhon = s.maketrans('', '', hanzi.punctuation)
        text = s.translate(trans_zhon)
    return text
    
def remove_stop_words(s, returnText = False):
    """ Remove all stop words """
    texts = [
            "",
            "a",
            "able",
            "about",
            "above",
            "according",
            "accordingly",
            "across",
            "actually",
            "after",
            "afterwards",
            "again",
            "against",
            "all",
            "allow",
            "allows",
            "almost",
            "alone",
            "along",
            "already",
            "also",
            "although",
            "always",
            "am",
            "among",
            "amongst",
            "an",
            "and",
            "another",
            "any",
            "anybody",
            "anyhow",
            "anyone",
            "anything",
            "anyway",
            "anyways",
            "anywhere",
            "apart",
            "appear",
            "appreciate",
            "appropriate",
            "are",
            "around",
            "as",
            "aside",
            "ask",
            "asking",
            "associated",
            "at",
            "available",
            "away",
            "awfully",
            "b",
            "be",
            "became",
            "because",
            "become",
            "becomes",
            "becoming",
            "been",
            "before",
            "beforehand",
            "behind",
            "being",
            "believe",
            "below",
            "beside",
            "besides",
            "best",
            "better",
            "between",
            "beyond",
            "both",
            "brief",
            "but",
            "by",
            "c",
            "came",
            "can",
            "cannot",
            "cant",
            "cause",
            "causes",
            "certain",
            "certainly",
            "changes",
            "clearly",
            "co",
            "com",
            "come",
            "comes",
            "concerning",
            "consequently",
            "consider",
            "considering",
            "contain",
            "containing",
            "contains",
            "corresponding",
            "could",
            "course",
            "currently",
            "d",
            "definitely",
            "described",
            "despite",
            "did",
            "different",
            "do",
            "does",
            "doing",
            "done",
            "down",
            "downwards",
            "during",
            "dr",
            "e",
            "each",
            "edu",
            "eg",
            "eight",
            "either",
            "else",
            "elsewhere",
            "enough",
            "entirely",
            "especially",
            "et",
            "etc",
            "even",
            "ever",
            "every",
            "everybody",
            "everyone",
            "everything",
            "everywhere",
            "ex",
            "exactly",
            "example",
            "except",
            "f",
            "far",
            "few",
            "fifth",
            "first",
            "five",
            "followed",
            "following",
            "follows",
            "for",
            "former",
            "formerly",
            "forth",
            "four",
            "from",
            "further",
            "furthermore",
            "g",
            "get",
            "gets",
            "getting",
            "give",
            "given",
            "gives",
            "go",
            "goes",
            "going",
            "gone",
            "got",
            "gotten",
            "greetings",
            "h",
            "had",
            "happens",
            "hardly",
            "has",
            "have",
            "having",
            "he",
            "hello",
            "help",
            "hence",
            "her",
            "here",
            "hereafter",
            "hereby",
            "herein",
            "hereupon",
            "hers",
            "herself",
            "hi",
            "him",
            "himself",
            "his",
            "hither",
            "hopefully",
            "how",
            "howbeit",
            "however",
            "i",
            "ie",
            "if",
            "ignored",
            "immediate",
            "in",
            "inasmuch",
            "inc",
            "indeed",
            "indicate",
            "indicated",
            "indicates",
            "inner",
            "insofar",
            "instead",
            "into",
            "inward",
            "is",
            "it",
            "its",
            "itself",
            "j",
            "just",
            "k",
            "km",
            "keep",
            "keeps",
            "kept",
            "know",
            "knows",
            "known",
            "l",
            "last",
            "lately",
            "later",
            "latter",
            "latterly",
            "least",
            "less",
            "lest",
            "let",
            "like",
            "liked",
            "likely",
            "little",
            "look",
            "looking",
            "looks",
            "ltd",
            "m",
            "mainly",
            "many",
            "may",
            "maybe",
            "me",
            "ms",
            "mean",
            "meanwhile",
            "merely",
            "might",
            "more",
            "moreover",
            "most",
            "mostly",
            "much",
            "must",
            "my",
            "myself",
            "n",
            "name",
            "namely",
            "nd",
            "near",
            "nearly",
            "necessary",
            "need",
            "needs",
            "neither",
            "never",
            "nevertheless",
            "new",
            "next",
            "nine",
            "no",
            "nobody",
            "non",
            "none",
            "noone",
            "nor",
            "normally",
            "not",
            "nothing",
            "novel",
            "now",
            "nowhere",
            "o",
            "obviously",
            "of",
            "off",
            "often",
            "oh",
            "ok",
            "okay",
            "old",
            "on",
            "once",
            "one",
            "ones",
            "only",
            "onto",
            "or",
            "other",
            "others",
            "otherwise",
            "ought",
            "our",
            "ours",
            "ourselves",
            "out",
            "outside",
            "over",
            "overall",
            "own",
            "p",
            "particular",
            "particularly",
            "per",
            "perhaps",
            "placed",
            "please",
            "plus",
            "possible",
            "presumably",
            "probably",
            "provides",
            "q",
            "que",
            "quite",
            "qv",
            "r",
            "rather",
            "rd",
            "re",
            "really",
            "reasonably",
            "regarding",
            "regardless",
            "regards",
            "relatively",
            "respectively",
            "right",
            "s",
            "said",
            "same",
            "saw",
            "say",
            "saying",
            "says",
            "second",
            "secondly",
            "see",
            "seeing",
            "seem",
            "seemed",
            "seeming",
            "seems",
            "seen",
            "self",
            "selves",
            "sensible",
            "sent",
            "serious",
            "seriously",
            "seven",
            "several",
            "shall",
            "she",
            "should",
            "since",
            "six",
            "so",
            "some",
            "somebody",
            "somehow",
            "someone",
            "something",
            "sometime",
            "sometimes",
            "somewhat",
            "somewhere",
            "soon",
            "sorry",
            "specified",
            "specify",
            "specifying",
            "still",
            "sub",
            "such",
            "sup",
            "sure",
            "t",
            "take",
            "taken",
            "tell",
            "tends",
            "th",
            "than",
            "thank",
            "thanks",
            "thanx",
            "that",
            "thats",
            "the",
            "their",
            "theirs",
            "them",
            "themselves",
            "then",
            "thence",
            "there",
            "thereafter",
            "thereby",
            "therefore",
            "therein",
            "theres",
            "thereupon",
            "these",
            "they",
            "think",
            "third",
            "this",
            "thorough",
            "thoroughly",
            "those",
            "though",
            "three",
            "through",
            "throughout",
            "thru",
            "thus",
            "to",
            "together",
            "too",
            "took",
            "toward",
            "towards",
            "tried",
            "tries",
            "truly",
            "try",
            "trying",
            "twice",
            "two",
            "u",
            "un",
            "under",
            "unfortunately",
            "unless",
            "unlikely",
            "until",
            "unto",
            "up",
            "upon",
            "us",
            "use",
            "used",
            "useful",
            "uses",
            "using",
            "usually",
            "uucp",
            "v",
            "value",
            "various",
            "very",
            "via",
            "viz",
            "vs",
            "w",
            "want",
            "wants",
            "was",
            "way",
            "we",
            "welcome",
            "well",
            "went",
            "were",
            "what",
            "whatever",
            "when",
            "whence",
            "whenever",
            "where",
            "whereafter",
            "whereas",
            "whereby",
            "wherein",
            "whereupon",
            "wherever",
            "whether",
            "which",
            "while",
            "whither",
            "who",
            "whoever",
            "whole",
            "whom",
            "whose",
            "why",
            "will",
            "willing",
            "wish",
            "with",
            "within",
            "without",
            "wonder",
            "would",
            "would",
            "x",
            "y",
            "yes",
            "yet",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "z",
            "zero",
            "project",
            "overall project summary", 
            "abstract",
            "project description", 
            "description", 
            "summary", 
            "description provided by applicant:",
            "overall",
            "applicant",
            "purpose",
            "summaryabstract",
            "descriptionabstract",
            "made",
            "highly",
            "research",
            "important",
            "study",
            "examine",
            "questions",
            "range",
            "funding",
            "funded",
            "program",
            "large",
            "based",
            "areas",
            "high",
            "field",
            "show",
            "provided",
            "provide",
            "successful",
            "application",
            "proposal",
            "leading",
            "approach",
            "closely",
            "knowledge",
            "continued",
            "support",
            "received",
            "methods",
            "david",
            "greatly",
            "seminar",
            "faced",
            "shown",
            "needed",
            "area",
            "academic",
            "worldwide",
            "proposed",
            "great",
            "goal",
            "focused",
            "specific",
            "remains",
            "essential",
            "small",
            "big",
            "large",
            "recently",
            "investigated",
            "supports",
            "successfully",
            "require"
    ]
    return [word for word in s if word not in texts] if returnText == False else ' '.join([word for word in s if word not in texts])


class MySQLPipline:
    """ Define Class for MySQL Connection"""

    def __init__(self, database="funding"):
        """ Initialize object """
        self.conn = pymysql.connect(
            host='localhost', port=3306, user='root', passwd='Kyle9975', db=database, charset='utf8')
        self.conn.autocommit(True)
        self.cursor = self.conn.cursor()

    def NIHDataset(self):
        """ Processing the SQL query"""
        sql = """
                SELECT DISTINCT
                    ProjectTitle,
                    ProjectNumber,
                    ContactPIProjectLeader,
                    ContactPIPersonID,
                    ProjectStartDate,
                    ProjectEndDate,
                    OrganizationName,
                    ProjectAbstract 
                FROM
                    FundedDataNIH 
                WHERE
                    ProjectAbstract IS NOT NULL
                    AND ContactPIProjectLeader IS NOT NULL
                """
        self.cursor.execute(sql)
        data = self.cursor.fetchall()
        data = pd.DataFrame(data)
        data.columns = ["title", "projID", "PIName", "PIID", "sDate",
                        "eDate", "institution", "abstract"]
        return data

    def ERCDataset(self):
        """ Processing the SQL query"""
        sql = """
                SELECT
                    ProjectTitle,
                    `Year`,
                    ProjectAcronym,
                    ProjectBudget,
                    PI,
                    Topic,
                    Country,
                    `Grant type`,
                    Abstract 
                FROM
                    FundedDataERC
                WHERE
                    Abstract IS NOT NULL
                    AND PI IS NOT NULL;
            """
        self.cursor.execute(sql)
        data = self.cursor.fetchall()
        data = pd.DataFrame(data)
        data.columns = ["title", "year", "projAcronym", "projBudget",
                        "PI", "Topic", "Country", "grantType", "abstract"]
        return data

    def NSFDataset(self):
        """ Processing the SQL query"""
        sql = """
                SELECT
                    AwardID,
                    AwardTitle,
                    AGENCY,
                    FirstName_Investigator,
                    LastName_Investigator,
                    AwardEffectiveDate,
                    AwardExpirationDate,
                    AwardAmount,
                    AbstractNarration,
                    Name_Institution,
                    POR_COPY_TXT_POR 
                FROM
                    FundedDataNSF 
                WHERE
                    AbstractNarration IS NOT NULL
                    AND FirstName_Investigator IS NOT NULL
                    AND LastName_Investigator IS NOT NULL
                    AND AGENCY is not null
            """
        self.cursor.execute(sql)
        data = self.cursor.fetchall()
        data = pd.DataFrame(data)
        data.columns = ["ID", "title", "agency", "firstName", "secondName", "effectiveDate",
                        "expirationDate", "amount", "abstract",
                        "institution", "por_copy_txt"]
        return data

    def UKRIDataset(self):
        """ Processing the SQL query"""
        sql = """
                SELECT 
                    a.title,
                    a.proRef,
                    a.resSub,
                    a.resTopic,
                    a.orgName,
                    a.sDate,
                    a.eDate,
                    a.institution,
                    a.department,
                    a.projType,
                    a.PIFirstName,
                    a.PISurname,
                    a.Amount,
                    a.Abstract
                FROM
                    UKRI_Funded_ALL_Raw a
                WHERE
                    a.Abstract IS NOT NULL
                    AND a.PIFirstName IS NOT NULL
                    AND a.PISurname IS NOT NULL
                    AND a.projType = "Research Grant"
                """
        self.cursor.execute(sql)
        data = self.cursor.fetchall()
        data = pd.DataFrame(data)
        data.columns = ["title", "projNumber", "resSub", "resTopic", "orgName", "StartDate", "EndDate", "instName",  "department", "projType",
                                               "PIFirName", "PISurname", "amount", "abstract"]
        return data

    def close_Conn(self):
        """ Closing Connection """
        self.cursor.close()
        self.conn.close()
    
class modelProcessor:
    def __init__(self, dat, colname = "abstract", debug = False):
        """ Initialize """
        self.data = pd.DataFrame(dat) if debug == False else pd.DataFrame(dat[0:10])
        self.colname = colname
        self.words = None
        self.dictionary = None
        self.corpus = None
        self.themes = None
        
    def remove_punctuations(self, lang = 'eng'):
        
        def x(s, lang = 'eng'):
            """ Removing punctuation """
            if lang == 'eng' or lang == 'english':
                translator = s.maketrans(' ', ' ', string.punctuation)
                text = s.translate(translator)
            elif lang == 'chi' or lang == 'chinese':
                trans_zhon = s.maketrans(' ', ' ', hanzi.punctuation)
                text = s.translate(trans_zhon)
            else:
                raise ValueError("Only support for english or chinese")
            return text
        
        self.data[self.colname] = self.data[self.colname].apply(x, lang = lang)
        return self.data
        
    def remove_abstract(self, regex):
        """ Remove project of which abstract is not available"""
        # To distinguish if the abstract is available
        pd.options.mode.chained_assignment = None
        self.data["tempCol"] = self.data["abstract"].apply(
            lambda s: len(re.findall(regex, s)) != 0)
        # Remove all projects whose no_abstract is True
        self.data = self.data[self.data["tempCol"] == False]
        self.data.reset_index(drop=True, inplace=True)
        self.data = pd.DataFrame(self.data).iloc[:, 0:len(pd.DataFrame(self.data).columns)-1]
        return pd.DataFrame(self.data)
    
    def remove_numberSpaces(self):
        def noDigitsSpace_check(text_colnumber):
            """ Check if projects have digits or multiple spaces"""
            emp = []
            for i in range(self.data.shape[0]):
                s = pd.DataFrame(self.data).iat[i, text_colnumber]
                if re.search(r'\s+[0-9]+\s+', s) and re.search(r'\s{2,}', s) and re.search(r'[0-9]+\s+', s):
                    emp.append(False)
                else:
                    emp.append(True)
            return all(emp)
        
        # Remove digits and multiple spaces
        while True:
            self.data[self.colname] = self.data[self.colname].apply(lambda s: re.sub(r'\s+[0-9]+\s+', " ", s.strip()))
            self.data[self.colname] = self.data[self.colname].apply(lambda s: re.sub(r'[0-9]+\s', " ", s.strip()))
            self.data[self.colname] = self.data[self.colname].apply(lambda s: re.sub(r'\s{2,}', " ", s.strip()))
            no_digits = noDigitsSpace_check(1)
            print(no_digits)
            if no_digits:
                break
        return pd.DataFrame(self.data)
    
    def multiple_regexFind(self, pattern = '^[0-9]*$', return_df = True):
        col = self.data[self.colname].apply(lambda s: True if re.match(pattern, s) else False).tolist()
        return [i for i, x in enumerate(col) if x] if return_df == False else pd.DataFrame(self.data)[pd.DataFrame(self.data).index.isin([i for i, x in enumerate(col) if x])]
        
    def remove_empty_text(self):
        self.data = self.data[~self.data[self.colname].isna()]
        self.data = self.data[self.data[self.colname] != ""]
        self.data = self.data[self.data[self.colname]!= "NA"]
        return self.data
    
    def remove_duplicated(self):
        self.data = self.data[~self.data.duplicated()]
        return self.data
    
    def remove_short_text(self, numBytes = 70):
        col = self.data[self.colname].apply(lambda s: True if len(s) > numBytes else False).tolist()
        self.data = self.data[pd.DataFrame(self.data).index.isin([i for i, x in enumerate(col) if x])]
        return self.data
    
    def clean_text(self):
        print("This may take long time to run...")
        st = t.time()
        self.remove_punctuations()
        self.data[self.colname] = self.data[self.colname].apply(lambda s: s.replace("ltbrgtltbrgt", " "))
        self.data[self.colname] = self.data[self.colname].apply(lambda s: s.replace("\ufeff"," ").strip())
        self.data[self.colname] = self.data[self.colname].apply(lambda x: x.replace("\n", " ").replace("ampquot", " ").replace("ampamp", " "))
        self.remove_numberSpaces()
        self.remove_empty_text().reset_index(drop = True, inplace = True)
        self.remove_abstract(regex=r'^[0-9]*$').reset_index(drop = True, inplace = True)
        self.remove_duplicated().reset_index(drop = True, inplace = True)
        self.remove_short_text().reset_index(drop = True, inplace = True)
        print("Finished! It takes {} seconds to run.".format(str(t.time() - st)))
        return self.data
    
    def remove_stop_words(self):
        # Splitting abstract
        self.data['corpus'] = self.data[self.colname].apply(lambda s: str(s).lower().split(" "))
        # Removing stop words
        self.data["corpus"] = self.data["corpus"].apply(remove_stop_words)
        # self.data["text_without_stopWords"] = self.data["text_without_stopWords"].apply(remove_stop_words, returnText = True)
        return pd.DataFrame(self.data)
        
    def get_corpus(self):
        """ get bag of words, words dictionary and corpus matrix """
        # Convert to lists
        self.words = np.asarray(self.data.corpus).tolist()
        self.dictionary = Dictionary(self.words)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.words]
        
    def topicNumSelection(self, numTopics_range=range(1, 50), num_iterations=250, coMetrics="c_v", verbose=False):
        """ Conduct Topic Number Selection by Comparing Coherence Metrics """
        coherence = pd.DataFrame(columns=["num_topics", coMetrics], data=None)
        for numTopic in tqdm(numTopics_range):
            m = LdaModel(corpus=self.corpus, id2word=self.dictionary,
                        iterations=num_iterations, num_topics=numTopic)
            cModel = CoherenceModel(
                model=m, corpus=self.corpus, dictionary=self.dictionary, coherence=coMetrics, texts=self.words)
            values = cModel.get_coherence()
            coherence = coherence.append(
                {"num_topics": numTopic, coMetrics: values}, ignore_index=True)
            if verbose:
                print("|  {}  |  {}  |".format(numTopic, values))
        return coherence
    
    def LDA_Model(self, numTopic, num_iterations=200):
        self.model = LdaModel(corpus=self.corpus, id2word=self.dictionary, iterations=num_iterations, num_topics=numTopic)
    
    def LDA_Visualisation(self):
        pyLDAvis.enable_notebook()
        return gensimvis.prepare(self.model, self.corpus, self.dictionary)
        
    def cv_repeat(self, repeatTimes=10, iterations=100, topicRange=(5, 40), figsize=(12, 6)):
        """ Repeat topic number selection function """
        fig, ax = plt.subplots(figsize=figsize)
        count = 0
        while count <= repeatTimes:
            count += 1
            grid = self.topicNumSelection(num_iterations=iterations, numTopics_range=range(topicRange[0], topicRange[1]))
            grid.to_csv("../Results/topicCVs_{}.csv".format(int(np.random.uniform(0, 300))), index=False)
            ax.plot(grid.num_topics, grid.c_v)
            ax.scatter(grid.num_topics, grid.c_v)
                
        ax.tick_params(axis='both', labelsize=15)
        ax.set_xlabel('Number of Topic', fontsize=16)
        ax.set_ylabel("C_V value", fontsize=16)
        return None
    
    def add_themes(self, themes):
        self.themes = themes
        
    def topic_prediction(self, txt):
        # Data Cleaning
        txt = remove_punctuation(txt.lower().replace("&amp;", "")).replace("\n", "").strip()
        txt = re.sub(r'\s+[0-9]+\s+', " ", txt)
        # Data Transformation
        unseen_words = [txt.split()]
        unseen_dictionary = Dictionary(unseen_words)
        unseen_corpus = [unseen_dictionary.doc2bow(text) for text in unseen_words]
        topicDistribution = self.model[unseen_corpus]
        # Sort Results and Match Topic Names
        dic = {}
        for topic in topicDistribution[0]:
            dic[topic[0]] = topic[1]
        dic = dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))
        return {self.themes[k-1]: dic[k] for k in dic}
    



