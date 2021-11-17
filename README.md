# IIR_HW3

## 利用Django來做網頁的呈現搜尋結果

## Introduction 
利用Word2Vec來分析解構文章內的字詞
一樣使用10000筆的Pubmed中covid19相關的資料，將文章經過前處理濾掉特殊符號和stopwords，  
使用genism的word2vec訓練，參數(window = 5, vector_size = 250, negative = 10, epochs=10)  
分別使用CBOW和skip-gram得出結果，不知道是不是參數的問題，兩者在cosine similarity 的表現相差不大  
並且在使用PCA降維顯示的時候，會有一群單字被分得很開，這些單字跟其他的字cosine similarity都很低  

## CBOW VS. Skip-gram
skip-gram 明顯沒辦法將單字分的像CBOW一樣開，
這就要去了解兩者分別的任務取向
CBOW要做的是在固定的window size下去預測一個單字填入空格
Skip-gram則是要用一個單字去predict一整句話
理論上CBOW會得到相似的字，例如cat和cats就應該很近
而Skip-gram就會得到語意代表類似的句子，如cat和dog就會比較近
也就是說兩者在區分字的方法就是不一樣的

## Result
![image](https://drive.google.com/uc?export=view&id=1Nde2bkbFtDQ7d2MvGJx4YUNFp-FNd9Av)   
![image](https://drive.google.com/uc?export=view&id=1Uc5EHAwRBzv53wKbQIGU78gVadEWnbxD)  

## The cool thing I found
![image](https://drive.google.com/uc?export=view&id=1Z4hbr1fvJ_DNgBHc1Mn1S_ZWEdZ6kPlh)  

## Something can do better
可以去找單字對應文章中出現的地方並顯示出來，可能covid19和2019有關，就顯示他們相關的證據，
如他們一起出現的句子。
別人有train文章分類module，3D的PCA顯示
