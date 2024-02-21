#############################
# Content Based Recommendation (İçerik Temelli Tavsiye)
#############################

#############################
# Film Overview'larına Göre Tavsiye Geliştirme
#############################
"""
Yeni kurulmuş online film izleme platformu kullanıcılarına film önerilerinde bulunmak istemektedir.
Kullanıcılarının login oranı çok düşük olduğu için kullanıcı alışkanlıklarını toplayamamaktadır. Bu sebeple iş birlikçi filtreleme
yöntemleri ile ürün önerileri geliştirememektedir.
Fakat kullanıcıların tarayıcıdaki izlerinden(cookilerinden) hangi filmleri izlediklerini bilmektedir. Bu bilgiye göre film önerilerinde bulununuz.


veriseti bilgisi
movies_metadata.csv 45000 film ile ilgili temel bilgileri barındırmaktadır
overview film açıklamalarını içermektedir.
"""





# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması

#################################
# 1. TF-IDF Matrisinin Oluşturulması
#################################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# https://www.kaggle.com/rounakbanik/the-movies-dataset
df = pd.read_csv(r"C:\Users\MerveATASOY\Desktop\data_scientist_miuul\egitim_teorik_icerikler\Bolum_6_Tavsiye_Sistemleri\dataset\the_movies_dataset\movies_metadata.csv", low_memory=False)  # DtypeWarning kapamak icin low_memory=False yapıldı
df.head()
df.shape
"""
budget :film bütçesi
genres : türleri
homepage : anasayfaları
id : veri seti içerisindeki id'leri
imdb_id : imdb'deki id
overview: açıklama
bizim için bu projede gerekli olan overview değişkenidir
"""



df["overview"].head()
# bu metinleri işlememiz lazım


tfidf = TfidfVectorizer(stop_words="english")
# her bir dökümanın hem kendi içinde hemde bütün dökümanda etkilerini göz önünde bulundurarak bir standardizasyon işlemi gerçekleştirildi
# in, on, an gibi ölçüm değeri taşımayan ve yaygın kullanılan ifadeler veri setinden çıkarılmalıdır
# çünkü oluşturulacak olan tf-idf matrisinde değerlerin ortaya çıkardığı problem var bu yüzden stop_words="english" kullanıldı
# bir şekilde on ifadesi geçen iki film birbirine yakın çıkarsa bu sonuçlarımızı saptırıyor olacaktır.


df[df['overview'].isnull()]

df['overview'] = df['overview'].fillna('') #NaN olanlar '' ile değiştirildi

tfidf_matrix = tfidf.fit_transform(df['overview'])

tfidf_matrix.shape   #(45466, 75827)  (açıklamalar, unique metinler)

df['title'].shape

tfidf.get_feature_names()
#burada gereksiz kelimelerde olabilir, anlamlı ya da anlamsız gelebilecek değerler silinip silinmeme konusunda değerlendirilebilir


tfidf_matrix.toarray()


#################################
# 2. Cosine Similarity Matrisinin Oluşturulması
#################################

cosine_sim = cosine_similarity(tfidf_matrix,
                               tfidf_matrix)

# benzerliği hesaplanmak istenen matrisi alır bir argüman ya da iki argüman şeklinde girilecek kullanılabilir
# bütün olası döküman çiftleri için tek tek cosine sim hesabı yapılmaktadır


cosine_sim.shape   #overview'lar
cosine_sim[1]

#1. satırda ilk bir filmin diğer tüm filmlerle olan benzerlik skoru yer almaktadır

#################################
# 3. Benzerliklere Göre Önerilerin Yapılması
#################################
#cosine_sim ile benzerlikler hesaplandı fakat bu skorları tek başına değrlendirmekte biraz zorlandık
#değerlendirme için bu filmlerin isimleri lazım


indices = pd.Series(df.index, index=df['title'])
# seri'nin index'ne filmin ismi, yanına ise bu isme sahip filmin hangi indexte olduğu nümerik bilgisi verildi


indices.index.value_counts()
# buradan bakıldığından title'larda çoklama olduğu görülür
# öyle birşey yapmalı ki bu çoklamalardan birini tutup diğerini silmeliyim

# önerilerde bulunabileceğim kişilerin güncellik açısından davranışlarını daha kolay şekillendirebileceğim varsayımıyla bu tür çoklama isimlendirmelerin en sonundakini alacağım
# son çekilen filme gidildi
indices = indices[~indices.index.duplicated(keep='last')]   # duplicate olanların sonuncusuna git demesi gerekmez miydi ?
# duplicate şunu yapar : bütün isimlendirmelere çoklama mı değil mi sorusunu sorar; True/False değerleri döner;  sonuncuyu tutar
# tilda olması duplicate olmayanlara git demek çünkü method duplicate olanlara true koyar
indices["Cinderella"]

indices["Sherlock Holmes"]

movie_index = indices["Sherlock Holmes"]

cosine_sim[movie_index] #cosine_sim'e bu index ile gidilirse sherlock holmes seçilmiş olunur, bu durumda sherlock ile diğer filmler arasındaki similartiy score'lara erişilir


similarity_scores = pd.DataFrame(cosine_sim[movie_index],
                                 columns=["score"])

movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
# azalan sırada en yüksek 10 film sıralandı

df['title'].iloc[movie_indices]

#################################
# 4. Çalışma Scriptinin Hazırlanması
#################################

def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a gore benzerlik skorlarını hesaplama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)

content_based_recommender("The Matrix", cosine_sim, df)

content_based_recommender("The Godfather", cosine_sim, df)

content_based_recommender('The Dark Knight Rises', cosine_sim, df)


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)


# SORU : Büyük ölçekli olarak bu iş veri tabanı seviyesinde nasıl gerçekleştirilir ?
# ÖRNEĞİN : Kullanıcıların en fazla izlediği 100  ya da 200 tane film belirlenir (id'ler alındı),
# Burada gerçekleştirilen işlemler alt kümeye indirgenen en çok ilgi gören 100 filmin her biri için gerçekleştirilir ve her biri için bir öneri seti oluşturulur ve bu bir tabloda tutulur
# id [önermek istenen id'ler]
# 1 [90, 12, 23, 45, 67]
# 2 [90, 12, 23, 45, 67]
# 3
