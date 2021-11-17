from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.core.paginator import Paginator

from search.models import Article, Inverted_index
from .forms import UploadDocumentForm

import os
import re
from difflib import get_close_matches
# /search folder location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
import pandas as pd

import copy
import xml.etree.ElementTree as ET

# Count Sentence / words / charater / find
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

file_type = ""
json_data = []
xml_data = []
filename = ""
back_space = "@$^*(<+=\{\\/'\"[-'_|"
front_space = ":~!#%^*)>+=\}\\,./?;'\"\]-_|"

nltk.download('wordnet')
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

# =====================================================================

from gensim.models.word2vec import Word2Vec   
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

# ==================================== HW3 ====================================
def pca():
    model = Word2Vec.load('search/word2vec.model')
    X = model.wv[model.wv.key_to_index]
    pca = PCA(n_components=3)
    result = pca.fit_transform(X) 
    return result

def pca_skip():
    model = Word2Vec.load('search/word2vec_skip.model')
    X = model.wv[model.wv.key_to_index]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X) 
    return result

# 預先算好pca的result，可能可以控制要word2vec再算
pca_result = pca()
pca_skip_gram_result = pca_skip()

def draw(model,similar_words,result,target):
    words = list(model.wv.key_to_index)
    pyplot.figure(figsize=(6,4))
    # fig = pyplot.figure(1, figsize=(4,3))
    # pyplot.clf()
    # ax = Axes3D(fig, rect=[0,0, .95, 1], elev=48, azim=134)
    # pyplot.cla()
    pyplot.title('CBOW')
    for word in similar_words:
        pyplot.scatter(result[model.wv.key_to_index[word],0], result[model.wv.key_to_index[word],1],c="green",marker="x",s=25)
        pyplot.text(result[model.wv.key_to_index[word],0]+0.01,result[model.wv.key_to_index[word],1],word)
    pyplot.scatter(result[model.wv.key_to_index[target],0], result[model.wv.key_to_index[target],1],c="red",marker="x",s=60)
    pyplot.text(result[model.wv.key_to_index[target],0]+0.01,result[model.wv.key_to_index[target],1],target)

    pyplot.grid()
    pyplot.savefig('search/static/image/statistic.png')
    pyplot.close()

def draw_skip(model,similar_words,result,target):
    words = list(model.wv.key_to_index)
    pyplot.figure(figsize=(6,4))
    pyplot.title('skip-gram')
    for word in similar_words:
        pyplot.scatter(result[model.wv.key_to_index[word],0], result[model.wv.key_to_index[word],1],c="gray",marker="x",s=25)
        pyplot.text(result[model.wv.key_to_index[word],0]+0.01,result[model.wv.key_to_index[word],1],word)
    pyplot.scatter(result[model.wv.key_to_index[target],0], result[model.wv.key_to_index[target],1],c="red",marker="x",s=60)
    pyplot.text(result[model.wv.key_to_index[target],0]+0.01,result[model.wv.key_to_index[target],1],target)

    pyplot.grid()
    pyplot.savefig('search/static/image/statistic_skip.png')
    pyplot.close()

def draw2(model,similar_words,result):
    words = list(model.wv.key_to_index)
    pyplot.figure(figsize=(6,4))
    pyplot.title('CBOW')
    pyplot.scatter(result[:,0], result[:,1],c="green",marker="x",s=18)
    for word in similar_words:
        pyplot.text(result[model.wv.key_to_index[word],0]+0.05,result[model.wv.key_to_index[word],1],word)
    pyplot.grid()
    pyplot.savefig('search/static/image/default.png')
    pyplot.close()

def draw2_skip(model,similar_words,result):
    words = list(model.wv.key_to_index)
    pyplot.figure(figsize=(6,4))
    pyplot.title('skip-gram')
    pyplot.scatter(result[:,0], result[:,1],c="gray",marker="x",s=18)
    for word in similar_words:
        pyplot.text(result[model.wv.key_to_index[word],0]+0.05,result[model.wv.key_to_index[word],1],word)
    pyplot.grid()
    pyplot.savefig('search/static/image/default_skip.png')
    pyplot.close()

def most_similar(w2v_model, words, topn=10):
    similar_df = pd.DataFrame()
    for word in words:
        try:
            similar_words = pd.DataFrame(w2v_model.wv.most_similar(word, topn=topn), columns=[word, 'cos'])
            similar_df = pd.concat([similar_df, similar_words], axis=1)
        except:
            print(word, "not found in Word2Vec model!")
    return similar_df

# web design for word2vec model show
def word2vec(request):
    model = Word2Vec.load('search/word2vec.model')
    model_skip = Word2Vec.load('search/word2vec_skip.model')
    result = []
    result_skip=[]
    global pca_result
    target = ''
    if 'token' in request.GET:
        target = request.GET['token'].lower()
        if target not in model.wv.key_to_index.keys():
            error = "Word not in the 10000 article"
            return render(request, 'html/word2vec.html', locals())
        top10 = most_similar(model, [target])
        top10_skip = most_similar(model_skip, [target])
        similar_words = []
        for word in top10[target]:
            similar_words.append(word)
        draw(model,similar_words,pca_result,target)
        draw_skip(model_skip,similar_words,pca_skip_gram_result,target)
        for word,cos in zip(top10[target], top10['cos']):
            temp = []
            temp.append(word)
            level = int(cos*10000)/100
            temp.append(level)
            if level > 90:
                temp.append('bg-danger')
            elif level > 70:
                temp.append('bg-warning')
            elif level > 50:
                temp.append('bg-success')
            else :
                temp.append('')

            result.append(temp)
        for word,cos in zip(top10_skip[target], top10_skip['cos']):
            temp = []
            temp.append(word)
            level = int(cos*10000)/100
            temp.append(level)
            if level > 90:
                temp.append('bg-danger')
            elif level > 70:
                temp.append('bg-warning')
            elif level > 50:
                temp.append('bg-success')
            else :
                temp.append('')

            result_skip.append(temp)
    else:
        similar_words=['covid19', 'patients', 'coronavirus', 'vaccine', 'disease', 'infection', 'pandemic', 'study', 'positive', '2019', '2020', 'methods', 'data', 'detection', 'symptoms', 'days', 'diagnosis', 'individuals', 'spanish', 'china', 'flu', 'fever','hoped']
        draw2(model,similar_words,pca_result)
        draw2_skip(model_skip,similar_words,pca_skip_gram_result)
        # print(result)
        
    return render(request, 'html/word2vec.html', locals())

# ==================================== HW3 ====================================

def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma

# show on index.html
def index(request):

    article_data = Article.objects.all()

    return render(request, 'html/index.html', locals())

# insert csv file
def insert(request):
    import pandas as pd
    for i in range (1,1001):
        data_set = pd.read_csv(
            'search/csv_data/'+str(i)+'.csv')
        df_records = data_set.to_dict('records')
        Article.objects.bulk_create(Article(**vals)
                                for vals in data_set.to_dict('records'))
    return redirect('/search/index')

def text_prepare(text):
    # 將以下符號直接替換成空白
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    # 0~9 a~z #+_
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z#+_]')

    STOPWORDS = set(stopwords.words('english'))
    # lowercase text
    text = text.lower()
    # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    
    # delete symbols which are in BAD_SYMBOLS_RE from text
    text = BAD_SYMBOLS_RE.sub('', text)
    # delete stopwords from
    temp = ""
    if text in STOPWORDS:
        return temp
    else :
        return text

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

# search
def browser_search(request):
    if 'search' in request.POST:
        search = request.POST['search']
        target = lemmatize(search.lower())
        if(Inverted_index.objects.filter(word=target).exists()):
            print(1)
            unit = Inverted_index.objects.get(word=target) # word : [[]....]
            output_article = Article.objects.filter(index = unit.content_index[0][0]) # first one article
            for slot in unit.content_index[1:]:
                temp_article = Article.objects.filter(index = slot[0])
                output_article = output_article | temp_article
            
            index_dict={}
            for slot2 in unit.content_index[::-1]:
                if slot2[0] not in index_dict: 
                    index_dict[slot2[0]]=list()
                    index_dict[slot2[0]].append(slot2[1])
                else:
                    index_dict[slot2[0]].append(slot2[1])

            page_index=[]
            for one in index_dict:
                page_index.append(one)

            paginator = Paginator(page_index, 20) # Show 10 contacts per page.

            page_number = request.GET.get('page')
            page_obj = paginator.get_page(int(page_number))
            if int(page_number) * 20 >= len(page_index):
                page_index = page_index[(int(page_number) - 1) * 20 :]
            else:
                page_index = page_index[(int(page_number) - 1) * 20 : int(page_number) * 20]
            result = []
            for one in page_index:
                article = Article.objects.get(index = one)
                topic = article.title
                topic = wordlowerReplace(topic,target)
                context = article.abstract
                for location in index_dict[one]:
                    tokens = nltk.word_tokenize(context)
                    text = ""
                    for j,word in enumerate(tokens):
                        if j==location:
                            text += '<span style="background:yellow; color:black;">' + tokens[j] + '</span> ' 
                        elif tokens[j] in front_space :
                            if text[-1] == " ":
                                text = text[:-1]+tokens[j] + " "
                            else:
                                text += tokens[j] + " "
                        elif tokens[j] in back_space :
                            text += tokens[j]
                        elif tokens[j]=="''":
                            text += "\""
                        else :
                            text += tokens[j] + " "
                    context = text
                article_list=[]
                article_list.append(topic)
                article_list.append(context)
                result.append(article_list)
        else:
            unit = Inverted_index.objects.all()
            score=[]
            for i in range(1,len(unit)):
                word = unit.get(id=i).word
                score.append(word)
            match=(get_close_matches(target,score,n=5))
            print(match)

    return render(request, 'html/result.html', locals())


# CONSTRCUT inverted table
def inverted_constrcut(request):
    inverted_table = {}
    for i in range(0,1000):
        unit = Article.objects.get(index=i)
        tokens = nltk.word_tokenize(unit.abstract)
        for j,word in enumerate(tokens):
            tokens[j] = lemmatize(text_prepare(tokens[j]))
            if tokens[j]!='':
                temp = [i,j]
                if(tokens[j] not in inverted_table):
                    inverted_table[tokens[j]]=[temp]
                else:
                    inverted_table[tokens[j]].append(temp)
    data = (pd.DataFrame(list(inverted_table.items()),columns=['word', 'content_index']))
    data.to_csv("test.csv",index=False)
    data_set = pd.read_csv('test.csv')
    df_records = data_set.to_dict('records')
    Inverted_index.objects.bulk_create(Inverted_index(**vals)
                            for vals in data_set.to_dict('records'))
    return redirect('/search/check_table')

def check_table(request):
    table = Inverted_index.objects.all()
    return render(request, 'html/table.html', locals())


def graph(request):

    return render(request,'html/graph.html', locals())

# 判斷要拿的資料是xml還是json，不清空session內的值，希望保留先前的結果
def home(request):
    form = UploadDocumentForm()
    files = os.listdir(BASE_DIR +'\\search\\media')
    global filename
    global file_type
    global json_data
    global xml_data
    # Show content
    if 'file_name' in request.GET:
        print("correct")
        words_count=0
        chars_count=0
        sentences_count=0
        filename = request.GET['file_name']
        yourPath = BASE_DIR+'/search/media'
        if filename.endswith('.json'):
            file_type = "json"
            json_output = []
            with open(yourPath+'/'+filename,encoding="utf-8") as f:
                json_data = json.load(f)

            for post in json_data:
                user_data=[]
                user_data.append('<span style="font-size:30px; color:rgb(0, 183, 255);">'+ post['username'] +'</span>')
                username = post['username']

                user_data.append(post['tweet_text'])
                content = post['tweet_text']

                json_output.append(user_data)

                # Count detail
                sentences_count += len(sent_tokenize(content))
                chars_count += len(content)
                words_count += len(post['tweet_text'].split( ))
                
            request.session['json_output']=json_output
            request.session['sentences_count']=sentences_count
            request.session['words_count']=words_count
            request.session['chars_count']=chars_count
        
        elif filename.endswith('.xml'):
            file_type = "xml"
            xml_data=[]
            tree = ET.parse(yourPath+'/'+filename)
            root = tree.getroot()

            for article in root.findall('.//Article'):
                for titles in article.findall('.//ArticleTitle'):
                    title = titles.text
                seg = []
                for content in article.findall('.//AbstractText'):
                    label = content.attrib.get('Label')
                    sentence = sent_tokenize(content.text)
                    
                    seg.append([label,sentence])
                    sentences_count += len(sent_tokenize(content.text))
                    chars_count += len(content.text)
                    words_count += len(content.text.split( ))

                xml_data.append([title,seg])
            request.session['sentences_count']=sentences_count
            request.session['words_count']=words_count
            request.session['chars_count']=chars_count
            request.session['xml_output']=xml_data
        else:
            del request.session['sentences_count']
            del request.session['words_count']
            del request.session['chars_count']

    if('file_error' in request.session):
        error = request.session['file_error']
        del request.session['file_error']

    if file_type == 'json':
        if('json_output' in request.session):
            json_output = request.session['json_output']
            # del request.session['json_output']
    elif file_type == 'xml':
        if('xml_output' in request.session):
            xml_output = request.session['xml_output']
            # del request.session['xml_output']

    # Count 

    if('sentences_count' in request.session):
        sentences_count = request.session['sentences_count']
    if('words_count' in request.session):
        words_count = request.session['words_count']
    if('chars_count' in request.session):
        chars_count = request.session['chars_count']

    if('find_count' in request.session):
        find_count = request.session['find_count']
        del request.session['find_count']
    return render(request, 'html/home.html', locals())

# 處理上傳檔案，且xml/json要用不同的暫存器儲存
def upload_file(request):
    global json_data
    global xml_data

    form = UploadDocumentForm()

    if request.method == 'POST':
        form = UploadDocumentForm(request.POST, request.FILES)  # Do not forget to add: request.FILES
        files = request.FILES.getlist('file_field')
        if form.is_valid():
            # ===== IF NOT XML OR JSON =====

            # 列出指定路徑底下所有檔案(包含資料夾)
            yourPath = BASE_DIR+'/search/media'
            # allFileList = os.listdir(yourPath)
            # for file in allFileList:
            #     print(file)

            # Save file to media folder
            for f in files:
                filename = str(f)
                if filename.endswith('.xml')!= True and filename.endswith('.json')!=True:
                    error = ("error with filename")
                    request.session['file_error']=error
                    return redirect('/search/home#sec')
                handle_uploaded_file(f)

            filename = ""

    return redirect('/search/home#sec')

# 儲存上傳的檔案到media資料夾
def handle_uploaded_file(f):
    save_path = os.path.join(BASE_DIR,'search','media',f.name)
    with open(save_path, 'wb+') as fp:
        for chunk in f.chunks():
            fp.write(chunk)

# 不論大小寫的search中的replace (ctrl+f)
def lowerReplace(sentence,target):
    list_num = []
    sentence_temp = sentence
    while sentence_temp.lower().find(target)!= -1:
        current_index=sentence_temp.lower().find(target)
        print(1)
        if list_num:
            index = list_num[-1]+current_index+len(target)
        else:
            index = current_index
        list_num.append(index)
        sentence_temp = sentence_temp[current_index+len(target):]
    print(list_num)
    for num in list_num[::-1]:
        str_replace = '<span style="background:yellow;color:black;">'+sentence[num:num+len(target)]+'</span>'
        sentence=sentence[0:num]+str_replace+sentence[num+len(target):]
    return sentence

# porter algorithm search (normal browser search)
def wordlowerReplace(sentence,target):
    for i,words in enumerate(nltk.word_tokenize(sentence)):
        if lemmatize(text_prepare(words.lower())) == lemmatize(target.lower()):
            sentence = sentence.replace(words,'<span style="background:yellow;color:black;">' + words + '</span>')
    return sentence

# 判斷是xml/json並處理資料，找到相對應的token用replace<span>的方法mark
def search(request):
    global file_type
    global json_data
    global xml_data   
    if 'search_token' in request.POST:
        find_count=0
        stemmer = PorterStemmer()
        # target = request.POST['search_token'].lower()
        target = stemmer.stem(request.POST['search_token'].lower())
        print("target :", target)
        # don't search stopwords
        if target in stopwords.words('english'):
            return redirect('/search/home#sec')
        if file_type=="json":
            json_output = []
            for line in json_data:
                user_data=[]
                user_data.append(line['username'].replace(target,'<span style="background:yellow; color:black;">'+target+'</span>'))
                user_data.append(line['tweet_text'].replace(target,'<span style="background:yellow; color:black;">'+target+'</span>'))
                json_output.append(user_data)
                # find Count
                find_count += line['username'].count(target)
                find_count += line['tweet_text'].count(target)
            request.session['find_count']=find_count
            request.session['json_output']=json_output
        elif file_type == "xml":
            xml_output_temp=copy.deepcopy(xml_data)
            for article in xml_output_temp:
                # title
                find_count += article[0].count(target)
                article[0]=lowerReplace(article[0],target)
                for list in article[1]:
                    # label
                    if(list[0]):
                        find_count += list[0].count(target)
                        list[0] = lowerReplace(list[0],target)
                    # list 是 可變的但string不是
                    for i,sentence in enumerate(list[1]):
                        find_count += list[1][i].count(target)
                        list[1][i] = wordlowerReplace(list[1][i],target)

            request.session['find_count']=find_count
            request.session['xml_output']=xml_output_temp

    return redirect('/search/home#sec')

# 將檔案全部清空，並跳到輸入欄位
def clear(request):
    global file_type
    global json_data
    global xml_data
    file_type=""
    json_data=[]
    xml_data=[]
    if('json_output' in request.session):
        del request.session['json_output']
    if('xml_output' in request.session):
        del request.session['xml_output']
    if('sentences_count' in request.session):
        del request.session['sentences_count']
    if('words_count' in request.session):
        del request.session['words_count']
    if('chars_count' in request.session):
        del request.session['chars_count']
    if('find_count' in request.session):
        del request.session['find_count']
    return redirect('/search/home#sec')
