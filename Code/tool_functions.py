from numpy import array
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from imp import reload

from pyinflect import getAllInflections
from nltk.stem import PorterStemmer # Great stemmer
from nltk.stem import LancasterStemmer # over-stemming easily, more aggresive stemmer
from nltk.stem.snowball import SnowballStemmer # stemmer for non-english languages
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer # lemmatizer (词形还原)
from sklearn.feature_extraction.text import TfidfVectorizer # get tf-idf matrix

from gensim.utils import simple_preprocess

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

def stemSentence(sentence, stemmer):
    token_words = word_tokenize(sentence)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(stemmer.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def all_stop_words():
    def getAllStopWords(wordsList):
        allwords = []
        for word in list(set(wordsList)):
            allwords.append(word)
            allwords = allwords + getInflectionLists(word)
        return set(allwords)
    stopwords =   [ "",
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
                    "provide",
                    "successful",
                    "application",
                    "proposal",
                    "lead",
                    "approach",
                    "closely",
                    "knowledge",
                    "continued",
                    "support",
                    "receive",
                    "method",
                    "david",
                    "greatly",
                    "seminar",
                    "face",
                    "shown",
                    "needed",
                    "area",
                    "academic",
                    "worldwide",
                    "proposed",
                    "great",
                    "goal",
                    "focus",
                    "specific",
                    "remains",
                    "essential",
                    "small",
                    "big",
                    "large",
                    "recently",
                    "supports",
                    "successfully",
                    "require",
                    "students",
                    "training",
                    "support",
                    "program",
                    'https',
                    'doi',
                    'org',
                    'www',
                    "http",
                    "identify",
                    "function",
                    "call",
                    "measure",
                    "understand",
                    "china",
                    "uk",
                    "easet",
                    "north",
                    "africa",
                    "india",
                    "west",
                    "south",
                    "january",
                    "february",
                    "march",
                    "april",
                    "may",
                    "june",
                    "july",
                    "august",
                    "september",
                    "october",
                    "november",
                    "december",
                    "measure",
                    "call",
                    "local",
                    "investigate",
                    "english",
                    "chinese",
                    "apply",
                    "cover",
                    "programme",
                    "propose",
                    "award",
                    "enable",
                    "participation",
                    "decide",
                    "pursue",
                    "work",
                    "survey",
                    "underly",
                    "objective",
                    "contact",
                    "progress",
                    "due",
                    "extend",
                    "induce",
                    "test",
                    "count",
                    "detected",
                    "recent",
                    "th", "nd", "rd",
                    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "lt",
                    "relate", "data", "significant", "develop", "make", "base"
                    "oxford", "programme", "japanese", "durham",
                    "british", "london", "sector", "affect", "sheet", "aspect", "main", 'increase', 'input', 'vast', 'amount'
                    "detail", 'future', 'dataset', 'unique', 'linkage', "predict", 'channel', 'experience', 
                    "bed", "poorly", 'estimate'
                    ]
    all_stopWords = getAllStopWords(stopwords)
    port = PorterStemmer()
    return set([port.stem(word) for word in all_stopWords])

def remove_short_words(wl, length=2):
    return [word for word in wl if len(word) > length]

def remove_punctuation(s, lang='eng'):
    """ Removing punctuation """
    if lang == 'eng' or lang == 'english':
        translator = s.maketrans(
            string.punctuation, ' '*len(string.punctuation))
        text = s.translate(translator)
    elif lang == 'chi' or lang == 'chinese':
        trans_zhon = s.maketrans(
            hanzi.punctuation, ' '*len(hanzi.punctuation))
        text = s.translate(trans_zhon)
    else:
        raise ValueError("Only support for english or chinese")
    return text

def getInflectionLists(word):
    inflection = []
    for elems in list(getAllInflections(word).values()):
        for elem in elems:
            if elem != "":
                inflection.append(elem)
            else:
                continue
    return list(set(inflection))
    
def remove_stop_words(s, wordList):
    """ Remove all stop words """
    return [word for word in s if word not in wordList and len(word) > 2]


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
                SELECT
                    *,
                    REGEXP_SUBSTR ( A.AwardNoticeDate, '[0-9]{4}' ) AS yr ,
                    REGEXP_SUBSTR ( A.ProjectStartDate, '[0-9]+/[0-9]+/[0-9]+' ) as sDate,
                    REGEXP_SUBSTR ( A.ProjectEndDate, '[0-9]+/[0-9]+/[0-9]+' ) as eDate
                FROM
                    (
                    SELECT DISTINCT
                        ProjectTitle,
                        ProjectNumber,
                        ContactPIProjectLeader,
                        ContactPIPersonID,
                        AwardNoticeDate,
                        ProjectStartDate,
                        ProjectEndDate,
                        OrganizationName,
                        ProjectAbstract,
                        OrganizationCity,
                        OrganizationState,
                        OrganizationCountry,
                        TotalCost
                    FROM
                        FundedDataNIH 
                    WHERE
                        ProjectAbstract IS NOT NULL 
                        AND ContactPIProjectLeader IS NOT NULL 
                    AND AwardNoticeDate IS NOT NULL 
                    ) A
                ORDER BY
                    --yr ASC
                """
        self.cursor.execute(sql)
        data = self.cursor.fetchall()
        data = pd.DataFrame(data)
        data.columns = ["title", "aID", "PILeader", "PIID", "aDate", "sDatetime", "eDateTime", "institution", "abstract", "city", "county", "country", "amount", "year", "sDate", "eDate"]
        return data

    def ERCDataset(self):
        """ Processing the SQL query"""
        sql = """
                SELECT DISTINCT
                    a.project_number,
                    a.project_ID,
                    SUBSTR(a.sDate, 7, 4) as `year`,
                    a.title,
                    b.Abstract,
                    b.Topic,
                    b.City,
                    b.Country,
                    b.HostInstitution,
                    a.amount,
                    a.sDate,
                    a.eDate 
                FROM
                    ERC_NEW a,
                    FundedDataERC b 
                WHERE
                    a.title = b.ProjectTitle
                    AND SUBSTR(a.sDate, 7, 4) > 2015
            """
        self.cursor.execute(sql)
        data = self.cursor.fetchall()
        data = pd.DataFrame(data)
        data.columns = ["aID", "callID",  "year", "title", "abstract", "topic", "county", "country",
                        "institution", "amount", "sDate", "eDate"]
        return data

    def NSFDataset(self):
        """ Processing the SQL query"""
        sql = """
                SELECT
                    AwardID,
                    AwardTitle,
                    Name_Institution,
                    AGENCY,
                    FirstName_Investigator,
                    LastName_Investigator,
                    SUBSTR(AwardEffectiveDate, 7, 4),
                    AwardEffectiveDate,
                    AwardExpirationDate,
                    AwardAmount,
                    AbstractNarration,
                    countryName_institution,
                    stateName_institution
                FROM
                    FundedDataNSF 
                WHERE
                    AbstractNarration IS NOT NULL
                    AND FirstName_Investigator IS NOT NULL
                    AND LastName_Investigator IS NOT NULL
                    AND AGENCY is not null
                    AND SUBSTR(AwardEffectiveDate, 7, 4) >= 2015
            """
        self.cursor.execute(sql)
        data = self.cursor.fetchall()
        data = pd.DataFrame(data)
        data.columns = ["aID", "title", "institution", "agency", "firstName", "lastName", "year", "sDate",
                        "eDate", "amount", "abstract", "country", "county"]
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
                        SUBSTR(a.sDate, 7, 4),
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
                        AND SUBSTR(a.sDate, 7, 4) >= 2015
                """
        self.cursor.execute(sql)
        data = self.cursor.fetchall()
        data = pd.DataFrame(data)
        data.columns = ["title", "aID", "resSub", "resTopic", "orgName", "year", "sDate", "eDate", "institution",  "department", "projType", "PIFirName", "PISurname", "amount", "abstract"]
        return data
    
    def get_data(self, sql, colnames):
        self.cursor.execute(sql)
        data = self.cursor.fetchall()
        data = pd.DataFrame(data)
        data.columns = colnames
        return data

    def close_Conn(self):
        """ Closing Connection """
        self.cursor.close()
        self.conn.close()
    
class nlp_ldamodel:
    def __init__(self, words, dictionary, corpus, themes = None):
        """ Initialize """
        # self.preprocessor = self.preprocessor(dataset = )
        self.words = words
        self.dictionary = dictionary
        self.corpus = corpus
        self.themes = themes
        self._stopwords = all_stop_words()
        
    def topicNumSelection(self, numTopics_range=range(1, 50), num_iterations=250, coMetrics="c_v", verbose=False):
        """ Conduct Topic Number Selection by Comparing Coherence Metrics """
        coherence = pd.DataFrame(columns=["num_topics", coMetrics], data=None)
        for numTopic in tqdm(numTopics_range):
            m = LdaModel(corpus=self.corpus, id2word=self.dictionary, iterations=num_iterations, num_topics=numTopic)
            cModel = CoherenceModel(model=m, corpus=self.corpus, dictionary=self.dictionary, coherence=coMetrics, texts=self.words)
            values = cModel.get_coherence()
            coherence = coherence.append(
                {"num_topics": numTopic, coMetrics: values}, ignore_index=True)
            if verbose:
                print("|  {}  |  {}  |".format(numTopic, values))
        return coherence
    
    def LDA_Model(self, numTopic, num_iterations=200):
        self.model = LdaModel(corpus=self.corpus, id2word=self.dictionary, iterations=num_iterations, num_topics=numTopic)
        return self.model
    
    def LDA_Visualisation(self):
        pyLDAvis.enable_notebook()
        return gensimvis.prepare(self.model, self.corpus, self.dictionary)
        
    def cv_repeat(self, repeatTimes=10, iterations=100, topicRange=(5, 40), countFrom = 0):
        """ Repeat topic number selection function """
        # fig, ax = plt.subplots(figsize=figsize)
        count = int(countFrom)
        while count <= repeatTimes + count:
            count += 1
            grid = self.topicNumSelection(num_iterations=iterations, numTopics_range=range(topicRange[0], topicRange[1]))
            grid.to_csv("../Results/topicCVs_{}.csv".format(count), index=False)
            # ax.plot(grid.num_topics, grid.c_v)
            # ax.scatter(grid.num_topics, grid.c_v)
        # ax.tick_params(axis='both', labelsize=15)
        # ax.set_xlabel('Number of Topic', fontsize=16)
        # ax.set_ylabel("C_V value", fontsize=16)
        return None
    
    def add_themes(self, themes):
        self.themes = themes
        
    def topic_prediction(self, txt):
        """hhhhhh"""
        # Data Cleaning
        txt = re.sub('&[A-Za-z]+;[A-Za-z]+;[A-Za-z]+;', ' ', txt)
        txt = re.sub('&[A-Za-z]+;[A-Za-z]+;', ' ', txt)
        txt = re.sub('&[A-Za-z]+;', ' ', txt)
        txt = re.sub('[^A-Za-z\s]', ' ', txt)
        txt = remove_punctuation(txt.lower())
        txt = remove_punctuation(txt.lower(), lang = 'chi').strip()
        port = PorterStemmer()
        txt = stemSentence(txt, stemmer = port)
        # Data Transformation
        words = [txt.split()]
        dict_ = Dictionary(words)
        corpus = [dict_.doc2bow(text) for text in words]
        topicDistribution = self.model[corpus]
        # # Sort Results and Match Topic Names
        # dic = {}
        # for topic in topicDistribution[0]:
        #     dic[topic[0]] = topic[1]
        # dic = dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))
        return topicDistribution[0]

class preprocessor:
    def __init__(self, dataset, colname='abstract'):
        self.dataset = pd.DataFrame(dataset)
        self.colname = colname
        self._stopwords = all_stop_words()
        
    def remove_punctuations(self, lang='eng', colname='abstract'):

        def x(s, lang='eng'):
            """ Removing punctuation """
            if lang == 'eng' or lang == 'english':
                translator = s.maketrans(
                    string.punctuation, ' '*len(string.punctuation))
                text = s.translate(translator)
            elif lang == 'chi' or lang == 'chinese':
                trans_zhon = s.maketrans(
                    hanzi.punctuation, ' '*len(hanzi.punctuation))
                text = s.translate(trans_zhon)
            else:
                raise ValueError("Only support for english or chinese")
            return text

        self.dataset[colname] = self.dataset[colname].apply(x, lang=lang)
        return self.dataset

    def remove_abstract(self, regex):
        """ Remove project of which abstract is not available"""
        # To distinguish if the abstract is available
        pd.options.mode.chained_assignment = None
        self.dataset["tempCol"] = self.dataset["abstract"].apply(
            lambda s: len(re.findall(regex, s)) != 0)
        # Remove all projects whose no_abstract is True
        self.dataset = self.dataset[self.dataset["tempCol"] == False]
        self.dataset.reset_index(drop=True, inplace=True)
        self.dataset = pd.DataFrame(self.dataset).iloc[:, 0:len(
            pd.DataFrame(self.dataset).columns)-1]
        return pd.DataFrame(self.dataset)

    def remove_numberSpaces(self):
        self.dataset[self.colname] = self.dataset[self.colname].apply(
            lambda s: re.sub(r'\s+[0-9]+\s+', " ", s.strip()))
        self.dataset[self.colname] = self.dataset[self.colname].apply(
            lambda s: re.sub(r'[0-9]+\s', " ", s.strip()))
        self.dataset[self.colname] = self.dataset[self.colname].apply(
            lambda s: " ".join(s.split()))
        return self.dataset

    def multiple_regexFind(self, pattern='^[0-9]*$', return_df=True):
        col = self.dataset[self.colname].apply(
            lambda s: True if re.match(pattern, s) else False).tolist()
        self.reset_index()
        return [i for i, x in enumerate(col) if x] if return_df == False else pd.DataFrame(self.dataset)[pd.DataFrame(self.dataset).index.isin([i for i, x in enumerate(col) if x])]

    def multiple_regexReplace(self, pattern='', subInto=''):
        self.dataset[self.colname] = self.dataset[self.colname].apply(
            lambda s: re.sub(pattern=pattern, repl=subInto, string=s))
        return self.dataset

    def remove_empty_text(self):
        self.dataset = self.dataset[~self.dataset[self.colname].isna()]
        self.dataset = self.dataset[self.dataset[self.colname] != ""]
        self.dataset = self.dataset[self.dataset[self.colname] != "NA"]
        return self.dataset

    def remove_duplicated(self):
        self.dataset = self.dataset[~self.dataset[["title", "abstract", "agency"]].duplicated()]
        return self.dataset

    def reset_index(self):
        return self.dataset.reset_index(drop=True, inplace=True)

    def remove_short_text(self, numBytes=70):
        col = self.dataset[self.colname].apply(
            lambda s: True if len(s) > numBytes else False).tolist()
        self.dataset = self.dataset[pd.DataFrame(
            self.dataset).index.isin([i for i, x in enumerate(col) if x])]
        return self.dataset
    
    def remove_short_words(self, colname = "corpus"):
        self.dataset[colname] = self.dataset[colname].apply(remove_short_words)
        return self.dataset

    def clean_text(self, stemText = True):
        """
        This function clean all text data in the dataset.
        Returns:
            [dataset]: [cleaned dataset]
        """        
        print("This may take long time to run...")
        st = t.time()
        print("Starting to remove punctuations, numbers and multiple spaces...")
        self.multiple_regexReplace('&[A-Za-z]+;[A-Za-z]+;[A-Za-z]+;', ' ')
        self.multiple_regexReplace('&[A-Za-z]+;[A-Za-z]+;', ' ')
        self.multiple_regexReplace('&[A-Za-z]+;', ' ')
        self.multiple_regexReplace('\s+http[A-Za-z\s]*', ' ')
        self.multiple_regexReplace('[^A-Za-z\s]', ' ')
        self.remove_punctuations()
        self.remove_punctuations(lang='chi')
        self.remove_punctuations(colname='title')
        self.remove_punctuations(lang='chi', colname='title')
        self.dataset[self.colname] = self.dataset[self.colname].apply(lambda s: s.replace("ltbrgtltbrgt", " ").replace(
            "ampquot", " ").replace("ampamp", " ").replace("ltbrgt", " ").strip())
        self.dataset[self.colname] = self.dataset[self.colname].apply(lambda s: s.replace("\n", " ").replace(u'\ufeff', " ").replace(
            u"\uf0e0", " ").replace(u"\xa0", " ").replace(u"\xad", " ").replace(u'\uf062', ' ').replace(u'\u200b', ' ').replace(u"\uf061b", " ").replace(u"\t", " ").replace('‐', ' ').replace(';', ' ').strip())
        self.remove_numberSpaces()
        print("Starting to remove projects with invalid abstracts...")
        self.remove_empty_text().reset_index(drop=True, inplace=True)
        self.remove_abstract(regex=r'^[0-9]*$').reset_index(drop=True, inplace=True)
        self.remove_duplicated().reset_index(drop=True, inplace=True)
        self.remove_short_text().reset_index(drop=True, inplace=True)
        if stemText:
            print("Starting to stem the abstracts...")
            st2 = t.time()
            port = PorterStemmer()
            # self.dataset["stemmed"] = self.dataset[self.colname].apply(stemSentence, stemmer=port)
            stemmed = [stemSentence(self.dataset.at[i, self.colname], port) for i in tqdm(range(self.dataset.shape[0]))]
            # print(len(stemmed))
            self.dataset['stemmed'] = stemmed
            print("Finished! It takes {0} seconds to run, stemming takes {1} seconds to run.".format(str(t.time() - st), str(t.time() - st2)))
        else:
            print("Finished! It takes {0} seconds to run.".format(str(t.time() - st)))
        return self.dataset

    def getAllCorpus(self, colname=''):
        if colname == '':
            colname = 'corpus'
        return [elem for i in range(self.dataset.shape[0]) for elem in self.dataset.at[i, colname]]

    def getWordOccurance(self, word='', colname="corpusSet"):
        col = self.dataset[colname].apply(lambda s: 1 if word in s else 0)
        return sum(col)/len(col)

    def remove_stop_words(self, getSet=False, colname = 'stemmed'):
        """ Removing Stop Words """
        st = t.time()
        # Splitting abstract
        self.dataset['corpus'] = self.dataset[colname].apply(lambda s: " ".join(remove_stop_words(str(s).lower().split(), wordList=self._stopwords)))
        if getSet:
            self.dataset["corpusSet"] = self.dataset["corpus"].apply(lambda ls: set(ls))
        print("Finished! It takes {} seconds to run.".format(str(t.time() - st)))
        return pd.DataFrame(self.dataset)

    # def setStopWordsList(self, stopWordsList, how='add'):
    #     if type(stopWordsList) != list or type(stopWordsList) != tuple or type(stopWordsList) != set:
    #         raise ValueError(
    #             "Please provide correct format for parameter 'stopWordsList', which can be (list, tuple or set)")
    #     if how == 'add':
    #         self._stopwords = self._stopwords.add(stopWordsList
    #     elif how == 'cus' or how == 'customise' or how == 'customize':
    #         self._stopwords = stopWordsList
    #     elif how == 'delete' or how == 'del':
    #         for word in stopWordsList:
    #             self._stopwords.remove(word)
    #     else:
    #         raise ValueError(
    #             "Please provide correct value ('add', 'del', 'customise' or 'cus') for 'how' parameter")

    # def word_stemmer(self, stemmedColname='corpus_stemmed', getSet=False):
    #     print("start stemming...")
    #     sT = t.time()
    #     port = PorterStemmer()
    #     self.dataset[stemmedColname] = self.dataset['corpus'].apply(corpus_stemmer, stemmer = port)
    #     if getSet:
    #         self.dataset["corpusSet"] = self.dataset[stemmedColname].apply(lambda ls: set(ls))
    #     print("Stemming finished! Stemming takes {} seconds to run".format(t.time() - sT))
    #     return self.dataset

    def get_corpus(self, colname='corpus_stemmed'):
        """ get bag of words, words dictionary and corpus matrix """
        # Convert to lists
        words = np.asarray(self.dataset[colname]).tolist()
        dictionary = Dictionary(words)
        corpus = [dictionary.doc2bow(text) for text in words]
        return words, dictionary, corpus



