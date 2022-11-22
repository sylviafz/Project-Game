#!/usr/bin/env python
# coding: utf-8

# # Deskripsi Proyek

# **Disini saya sebagai data analis dari toko daring 'ice' akan menganalisis tentang penjualan game dari seluruh dunia, analisis ini akan sangat berguna untuk kampanye iklan di tahun 2017 untuk menaikkan penjualan toko daring kita**
# 
# **analisis ini juga nantinya menjelaskan bagaimana pola game-game dari sebuah platform dapat berhasil atau tidaknya menjual game yang dirilis, nantinya kita bisa melihat game seperti apa yang paling berpotensi dalam penjualan**
# 
# **Nantinya tahap-tahap yang ada di analisis ini seperti melihat informasi data, bagaimana distribusi data, memperbaiki data, melihat hubungan dari 1 variabel ke variabel lain, serta menentukan hipotesis untuk melihat bagaimana hasil akhir dari analisis yang mungkin akan sangat berguna untuk pengambilan keputusan dalam melakukan kampanye di tahun 2017 nanti.**
# 

# # Memuat library yang dibutuhkan

# In[1]:


#memuat library
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, scipy.stats as st


# In[2]:


pd.set_option('display.max_columns', None)


# # Memuat data

# In[3]:


df = pd.read_csv('/datasets/games.csv')


# # Melihat informasi pada data

# In[4]:


#melihat sample data
df.head()


# In[5]:


#melihat informasi data
df.info()


# **kesimpulan awal:**
# 
# **sekilas dilihat dari informasi data dan sample data yang ditampilkan, bahwa data yang saya miliki mempunyai nilai hilang, ada juga beberapa tipe data yang salah yang harus diperbaiki untuk melakukan analisis, saya juga akan mengaganti nama kolom untuk memudahkan analisis pada data ini**

# # Menyiapkan data

# ## Mengganti nama kolom

# **ini dilakukan agar lebih memudahkan dalam pemanggilan nama-nama kolom atau variabel dalam melakukan analisis, saya akan mengganti nama kolom dengan huruf kecil semua**

# In[6]:


#melihat nama kolom pada dataFrame
df.columns


# In[7]:


#mengganti nama kolom
df.columns = df.columns.str.lower()


# In[8]:


#melihat kembali nama kolom pada dataFrame
df.columns


# ## Mengganti nilai yang hilang

# **Dalam data ini dilihat dari informasi datanya ada beberapa kolom yang mempunyai nilai yang hilang, seperti pada kolom 'name', 'year_of_release', 'genre', 'critic_score', 'user_score' , dan 'rating'. untuk mengisi nilai yang hilang tersebut perlu melihat lebih dalam lagi bagaimana nilai pada kolom-kolom tersebut**

# ### Mengganti nilai hilang pada kolom 'name'

# In[9]:


#melihat bagaimana kolom yang hilang pada kolom 'name'
df.loc[df['name'].isna()]


# **ternyata hanya ada 2 nilai yang hilang pada kolom 'name', namu dilihat dari data tersebut sepertinya 2 nilai yang hilang pada kolom 'name' menjadi tidak berguna dikarenakan banyak juga nilai yang hilangnya speerti pada kolom 'genre', 'critic_score', dll. jadi saya akan menghapus 2 row tersebut**

# In[10]:


#mengahpus nilai yang hilang pada kolom 'name'
df.dropna(subset=['name'], inplace=True)


# In[11]:


df.loc[df['name'].isna()]


# **sekarang sudah tidak ada lagi nilai yang hilang pada kolom 'name'**

# ### Mengganti nilai yang hilang pada kolom 'genre'

# In[12]:


#melihat bagaimana kolom yang hilang pada kolom 'genre'
df.loc[df['genre'].isna()]


# **tampaknya sudah tidak ada nilai hilang pada kolom 'genre' dikarenakan nilai hilang pada kolom genre dan kolom 'name' tampak simetris, jadi ketika menghapus nilai hilang pada kolom 'name' diatas maka nilai yang hilang pada kolom 'genre' pun sudah teratasi**

# ### Mengganti nilai yang hilang pada kolom 'year_of_release'

# In[13]:


#melihat bagaimana nilai yang hilang pada kolom 'year_of_release'
df.loc[df['year_of_release'].isna()]


# In[14]:


#menghitung persentase nilai hilang pada kolom 'year_of_release'
df['year_of_release'].isna().sum() / len(df)


# **ternyata ada sekitar 10 persen nilai yang hilang pada kolom 'year_of_release'**

# In[15]:


#menambahkan kolom 'year_by_name' dimana kolom tersebut berisi tahun hasil irisan dari nilai di kolom name
#lalu nilai yang ada pada kolom 'year_by_name' adalah nilai hilang pada kolom 'year_of_release'
#selain itu nilai yang tidak hilang pada kolom year_of_release akan diganti 0
df['year_by_name'] = df['name'].str[-4:]
df.loc[~(df['year_of_release'].isna()), 'year_by_name'] = 0
df


# In[16]:


#mengganti tipe data 'year_by_name' ke tipe numerik
#dengan menggunakan errors = 'coerce', maka selain numerik akan dijadikan NaN
#lalu mengisi nilai yang hilang pada kolom 'year_by_name' sesuai dengan nilai pada baris 'year_of_release'
df['year_by_name'] = pd.to_numeric(df['year_by_name'], errors= 'coerce')
df.loc[df['year_by_name'].isna(), 'year_by_name'] = df['year_of_release']
df.loc[df['year_by_name'].isna()]
df['year_by_name'].value_counts()


# **sekarang masih ada nilai yang hilang di kolom 'year_by_name' dimana nilai hilang ini menandakan bahwa, nilai tersebut juga hilang pada kolom 'year_of_release' dan tidak ada clue tahun pada kolom 'name', maka saya akan mengganti nilai hilang tersebut pada kolom 'year_by_name' dengan nilai 0**

# In[17]:


#agar membedakan nilai 0 yang artinya memang tidak ada keterangan tahun pada nama game dengan nilai hilang
#makan nilai 0 pada kolom 'year_by_name' akan saya isi dengan nilai2 yang sesuai dengan kolom 'year_of_release'
df.loc[df['year_by_name']== 0, 'year_by_name'] = df['year_of_release']


# In[18]:


#lalu mengisi nilai yang hilang dengan nilai 0 yang artinya memang tidak ada keterangan tahun pada game tersebut
df['year_by_name'] = df['year_by_name'].fillna(0)


# In[19]:


#melihat bagaimana nilai 0 pada kolom 'year_by_name'
df.loc[df['year_by_name']== 0]


# In[20]:


#melihat nilai unik pada kolom 'year_by_name'
df['year_by_name'].unique()


# In[21]:


#masih terdapat nilai 500, yang artinya tidak mungkin ada game yang di rilis di tahun 500
#mari ganti nilai 500 tersebut dengan nilai 0
df.loc[df['year_by_name']== 500, 'year_by_name'] = 0


# In[22]:


#melihat kembali apakah masih ada nilai aneh dalam kolom 'year_by_name'
df['year_by_name'].unique()


# **sekarang say mempunyai 2 kolom tahun release game tersebut, dimana kolom 'year_of_release' adalah kolom asli dataset, dan kolom 'year_by_name' adalah kolom yang nilainya sama dengan nilai di kolom 'year_of_release' namun nilai hilangnya sudah diganti sesuai tahun yang ada di dalam kolom 'name' dan nilai 0 untuk game yang sama sekali tidak memounyai keterangan tahun release**

# ### Mengganti nilai yang hilang pada kolom 'critic_score'

# In[23]:


#melihat bagaiman nilai hilang pada kolom 'critic_score'
df.loc[df['critic_score'].isna()]


# In[24]:


#melihat nilai unik pada kolom 'critic_score'
df['critic_score'].unique()


# In[25]:


#mengganti nilai yang hilang dengan nilai 0
#nilai 0 artinya bahwa game tersebut tidak mempunya nilai ulasan dari kritikus
df['critic_score'] = df['critic_score'].fillna(0)


# In[26]:


#melihat kembali bagaimana nilai hilang pada kolom 'critic_score'
df.loc[df['critic_score'].isna()]


# **sekarang sudah tidak ada lagi nilai yang hilang pada kolom 'critic_score'**

# ### Mengganti nilai yang hilang pada kolom 'user_score'

# In[27]:


#melihat nilai yang hilang pada kolom 'user_score'
df.loc[df['user_score'].isna()]


# In[28]:


#melihat nilai unik pada kolom 'user_score'
df['user_score'].unique()


# In[29]:


#melihat bagaimana nilai 0 pada kolom 'user_score'
df.loc[df['user_score']== '0']


# **dilihat dari data diatas bahwa nilai nilai yang terdapat pada kolom 'user_score' sangat bervarisi, terdapat juga nilai 0 yang berarti bahwa memang game tersebut mendapat review jelek, juga terdapat nilai TBD yang artinya baru akan ditentukan nilai dari game tersebut oleh user. Nilai TBD ini akan saya ganti dengan nilai NaN, karena jika mengisi dengan nilai 0 bisa jadi salah asumsi karena pada kolom tersebut terdapat juga nilai 0 yang mengasumsikan bahwa game tersebut mungkin jelek. lalu untuk nilai yang hilang pada kolom 'user_score' ini artinya game tersebut memang tidak di review/ diberikan skor oleh pengguna, maka dari itu saya akan membiarkan nilai yang hilang tersebut. maka pada kolom ini bisa diasumsikan bahwa nilai yang disini bisa jadi karena gamenya tidak di reviwe, atau memang belum ditentukan nilai reviewnya**

# In[30]:


#mengganti nilai yang hilang pada kolom 'user_score' dengan 'na' (not available)
df.loc[df['user_score']== 'tbd', 'user_score'] = np.NaN


# In[31]:


#melihat kembali bagaimana nilai unik pada kolom 'user_score'
df['user_score'].unique()


# In[32]:


#melihat kembali bagaimana nilai hilang pad kolom 'user_score'
df.loc[df['user_score'].isna()]


# **sekarang kolom user_score sudah diperbaiki meskipun ada nilai yang hilang pada kolom tsb, saya asumsikan bahwa nilai yang hilang tersebut memang tidak/belum mendapat score dari user**

# ### Mengganti nilai yang hilang pada kolom 'rating'

# In[33]:


#melihat nilai yang hilang pada kolom 'rating'
df.loc[df['rating'].isna()]


# In[34]:


#melihat nilai unik pada kolom 'rating'
df['rating'].unique()


# **dilihat dari bagaimana nilai yang hilang tersebut, sepertinya nilai yang hilang pada kolom rating ini berhubungan dengan nilai pada kolom 'critic_score' dan kolom 'user_score', dimana kolom tersebut juga memiliki nilai yang hilang, maka dari itu, untuk mengisi nilai yang hilang pada kolom 'rating' saya juga akan menggantinya dengan nilai 'na' atau not available**

# In[35]:


#mengganti nilai yang hilang pada kolom 'user_score' dengan 'na' (not available)
df['rating'] = df['rating'].fillna('na')


# In[36]:


#melihat kembali bagaimana nilai yang hilang pada kolom 'rating'
df.loc[df['rating'].isna()]


# In[37]:


#melihat kembali nilai unik pada kolom 'rating'
df['rating'].unique()


# **sekrang sudah tidak ada lagi nilai yang hilang pada kolom rating**

# In[38]:


#melihat kembali informasi data yang sudah diperbaharui
df.info()


# **sepertinya datanya sudah benar, meskipun terlihat seperti ada nilai yang hilang pada kolom 'year_of_release' namun sudah dibenarkan nilai pada kolom tersebut dengan membuat kolom baru 'year_by_name'**

# ## Mengubah tipe data

# **dilihat dari informasi data diatas bahwa kolom year_of_release dan kolo year_by_name memiliki tipe data object/string, dimana seharusnya tahun adalah bilangan bulangan atau integer, jadi saya akan mengubah kolom tersebut ke tipe integer**

# In[39]:


#mengubah kolom 'year_by_name' ke integer
df['year_by_name'] = df['year_by_name'].astype(int)


# In[40]:


#mengubah kolom 'user_score' ke integer
df['user_score'] = df['user_score'].astype(float)


# In[41]:


#melihat kembali informasi data
df.info()


# **disini saya hanya mengubah tipe data kolom year_by_name saja ke integer, karena menurut saya tipe data di kolom lain sudah sesuai. juga untuk kolom year_of_release tidak saya ubah karena kolom tersebut adalah kolom dataset asli dari kolom year_by_name, jd dengan hanya mengubah kolom year_by_name yang nilainya sama dengan kolom year_of_release sudah cukup**

# ## Menghitung jumlah penjualan tiap game

# In[42]:


#membuat fungsi untuk menambahkan kolom 'total_sales' yang didapat dari hasil penjualan setiap wilayah
def sum(a,b,c,d):
    sum  = a + b + c + d
    return (sum)


# In[43]:


#membuat kolom yang berisi nilai total_sales
df['total_sales'] = list(map(sum, df['na_sales'], df['eu_sales'], 
                             df['jp_sales'], df['other_sales']))


# In[44]:


#melihat bagaimana nilai di kolom total_sales
df.head()


# **sekarang saya memiliki kolom total_sales yaitu total penjualan dari berbagai negara**

# # Analisis data

# ## Melihat game yang di rilis berdasarkan tahun

# In[45]:


#melihat kembali informasi data
df.info()


# In[46]:


#melihat berapa banyak game yang dirilis setiap tahunnya
year_group = df.groupby('year_by_name')['name'].count().sort_values(ascending=False).reset_index()
year_group.head(10)


# **ternyata paling banyak game yang dirilis pada tahun 2008, agar lebih jelas. saya akan menampilkannya dalam bentuk bar chart dibawah**

# In[47]:


##membuat visualisi untuk game rilis pertahun
plt.figure(figsize=(25,20))
splot = sns.barplot(data=year_group, x='year_by_name', y='name') # assign plot kedalam object
#kode anotasi
for g in splot.patches:
    splot.annotate(format(g.get_height(), '.1f'),
                   (g.get_x() + g.get_width() / 2., g.get_height()),
                   ha = 'center', va = 'center',
                   xytext = (0, 20), rotation=45,
                   textcoords = 'offset points')
#kode anotasi
plt.xticks(rotation = 45)
plt.show()


# **bisa dilihat dari bar chart diatas bahwa dari tahun ke tahun perilisan game semakin naik signifikan, puncaknya pada tahun 2008 dan 2009 banyak game yang di rilis oleh publisher, namun kemudian mulai menyusut lagi setelahnya, nilai 0 dari bar tersebut adalah game yang dirilis dengan tahun yang tidak diketahui**

# ## Melihat Variasi penjualan

# In[48]:


#melihat platform yang memiliki tingkat penjualan terbesar
group_platform = df.groupby('platform')['total_sales'].sum()
group_platform = group_platform.sort_values(ascending=False).reset_index()
group_platform.head()


# In[49]:


#membuat visualisi untuk platform berdasarkan penjualannya
plt.figure(figsize=(18,12))
sns.barplot(data=group_platform, x='platform', y='total_sales')
plt.xticks(rotation = 45)
plt.show()


# **bisa dilihat lagi dari tabel dan bar chat diatas bahwa yang memiliki penjualan paling besae itu yaitu pada platfrom PS2 yang memiliki total penjualan lebih dari 1 milyar dolar, lalu diikuti X360, PS3, Wii**

# In[50]:


#melihat distribusi pada platform yang memiliki tingkat penjualan paling besar yaitu PS2
group_plat = df.loc[df['platform'] == 'PS2']
group_plat = group_plat.groupby('year_by_name')['total_sales'].sum().reset_index()
group_plat


# In[51]:


#melihat distribusi penjualan pada platform PS2 berdasarkan tahun relaese
plt.figure(figsize= (12,9))
sns.barplot(data=group_plat , x='year_by_name', y='total_sales')
plt.title('sales PS2 by year')
plt.show()


# **bisa dilihat dari hasil data diatas pada diagram batang, penjualan pada platform PS2 yaitu platform paling populer dan memiliki penjualan terbanyak. PS2 mempunyai pendapatan paling banyak tahun 2004 dan semakin kesini sepertinya penjualannya semakin menurun**

# **sekarang mari lihat platfrom mana yang dulunya populer yang memiliki banyak penjualan, tapi kini kini tidak memiliki penjualan apapun**

# In[52]:


#melihat data platform dengan penjualan pertahun dengan pivot table
df_pivot = df.pivot_table(index = 'platform', values='total_sales', columns='year_by_name', aggfunc='sum').reset_index().fillna(0)
df_pivot


# **dilihat dari data diatas yang memiliki penjualan pada tahun 1980 hanya ada 1 platform yaitu platform 2600, namun pada tahun terakhir data diambil tahun 2016, sepertinya platfrom tersebut tidak mempunyai penjualan apapun semenjak 16 tahun kebelakang**

# In[53]:


#melihat distribusi pada pltform 2600
df_2600 = df.loc[df['platform'] == '2600']
df_2600


# **bisa dilihat dari tabel pivot table diatas bahwa ada platform yang dulunya sangat populer dan bahkan jaman sekarang tidak laku, atau bahkan apakah game tersebut sudah tidak ada? pada platform 2600 tahun 1980 platfrom tersebut mungkin mulai digemari hingga mendapat banyak penjualan, lalu pada tahun berikutnya platform tersebut merilis lagi game baru seperti pitfall! pada tahun 1981 dan pacman pada tahun 1982 yang juga mendapat banyak pendapatan. sepertinya tahun tersebut adalah masa jaya platform 2600 namun setelah tahun 1983 penjualan game pada platform tersebut terus menurun hingga saat ini**

# **pada tabel-tabel diatas kita bisa melihat bahwa rata2 sebuah perusahaan game bertaahan sangat bervariasi, ada yang hanya memiliki pendapatan ditahun pertama, lalu setelah itu tidak mendapat penjualan lagi sampai detik ini, atau ada juga yang bertahan hingga puluhan tahun seperti pada platfrom PC, atau ada juga yang hanya memilki pedapatan ditahun pertama dan kedua, lalu bertahun2 kemudian tidak mendapat penjualan lagi akan tetapi masih bertahan hingga masa kini. bisa kita simpulkan bahwa rata**

# In[54]:


#menemukan platform yang dulunya populer tetapi sekarang tidak memiliki penjualan apa pun
#asumsikan sekarang ini adalah tahun 2016
#mencari platform yang tidak memiliki penjualan apapun pada tahun 2016
platform_punah = list(df_pivot.loc[df_pivot[2016]== 0, 'platform'])
platform_punah


# **diatas sini adalah platfrom2 yang tidak memiliki penjualan sama sekali pada tahun 2016**

# In[55]:


#melihat bagaimana data pada platform punah
punah = df.loc[(df.platform).isin(platform_punah)]
punah.head()


# **dilihat dari tabel diatas, bahwa ternyata memang platfrom2 diatas yang diliris pada tahun 80-90an mempunyai penjualan yang bagus (dilihat dari kolom total sales) namun ternyata pada tahun 2016 platfrom2 tersebut mungkin sudah punah atau tidak bisa beroperasi lg karena sudah tidak memiliki penjualan**

# In[56]:


#melihat distribusi data penjualan pada platform yang tidak memiliki penjualan sekarang ini (tahun 2016)
punah_pivot = punah.pivot_table(index = 'year_by_name', values='total_sales', columns='platform', aggfunc='sum')
punah_pivot.head()


# In[57]:


#mengganti nilai hilang dengan 0 untuk visualisasi linechart agar lebih mudah dipahami
#menghapus nilai 0 pada kolom year_by_name
#melakukan reset_index untuk visualisasi
punah_viz = punah_pivot.fillna(0).drop([0]).reset_index()
punah_viz.head()


# In[58]:


#melihat distribusi platform punah dengan line chart
punah_viz.plot(x='year_by_name', figsize=(20, 10))
plt.title('Distribusi penjualan platfrom dari tahun ke tahun')
plt.show()


# **line chart diatas menunjukkan plaform2 dengan penjualan pertahun, dari linechart diatas juga bisa dilihat bagaimana suatu platform saat memiliki penjualan tinggi, dan penjualan rendah**
# **contoh pada platfrom PS2 yang memiliki penjualan tinggi ditahun 2002 sampai 2005 lalu setelah itu penjualan mereka turun** 

# **sekarang saya akan melihat bagaimana suatu platfrom rata2 berkembang dan bertahan**

# In[59]:


#membuat fungsi untuk menghitung berapa tahun sebuah platfrom berkembang dan bertahan
for col in punah_pivot.columns:
    punah_pivot.loc[punah_pivot[col] > 0, col] = 1


# In[60]:


#membuat pivot yang berisi jumlah tahun suatu platfrom dapat menjual game-nya
df_avg = punah_pivot.count().sort_values(ascending=False).reset_index()
df_avg = df_avg.head(10)
df_avg


# In[61]:


#menghitung rata-rata tahun sebuah platfrom bertahan
df_avg.columns = ['platform', 'count']
df_avg['count'].mean()


# **sekarang saya mengetahui bahwa rata2 sebuah platfrom bertahan atau sebuah platfrom mendapatkan penjualan game-nya itu adalah sekitar 11-12 tahun**

# **hasilnya ternyata dari top 10 game terlaris itu mereka rata2 bertahan di 11-12 tahun, mari kita ambil periode 11 tahun ini untuk menganalisi lebih lanjut untuk mengetahui game mana yang memiliki potensi paling menguntungkan untuk tahun 2017. karena data ini diambil dari tahun 2016 makan kita akan mengambil data dari tahun 2005-2016**

# ## Menganalis data berdasarkan periode yang terlah ditentukan

# **berdasarkan analisis diatas saya akan menentukan periode data 11 tahun di tahun 2005-2016, dimana periode tersebut umumnya sebuah platfrom memiliki masa kejayaannya dan data tersebut bisa digunakan untuk membangun acuan di tahun 2017**

# In[62]:


#mengambil data dari periode 2005-2016
dfset = df.query("year_by_name >= 2005 and year_by_name <= 2016")
dfset.head()


# In[63]:


#membuat tabel untuk melihat platform yang memiliki penjualan terbanyak ditahun 2006-2016
dfset.groupby('platform')['total_sales'].sum().sort_values(ascending=False)


# **dilihat dari data diatas bahwa platfrom yang memiliki penjualan terbanyak pada periode tersebut yaitu platfrom X360 dengan total penjualan 961.30 juta dolar**

# In[64]:


#membuat tabel untuk melihat penjualan platfrom dari tahun ke tahun (2005-2016)
dfset_pivot = dfset.pivot_table(index = 'year_by_name', values='total_sales', columns='platform', 
                                aggfunc='sum').reset_index()
dfset_pivot


# 
# **dilihat dari tabel diatas, bahwa ada beberapa platfrom yang menyusut penjualannya seperti pada platfrom PS2 dan PS3 karena platfrom tersebut terganti/ terupdate oleh game PS4 yang mulai tumbuh pada saat tahun rilisnya di tahun 2013.**
# 
# **platfrom2 yang berpotensi menghasilkan keuntungan adalah platfrom yang berusia kurang dari 10 tahun seperti hasil analisis saya diatas pada usia rata2 sebuah platfrom bertahan. dalam kasus ini platfrom yang berusia mendekati 10 tahun akan semakin kehilangan penjualannya, sebaliknya platfrom yang berusia muda berpotensi untuk berkembang seperti platfrom XOne, PS4, 3DS, WiiU**

# In[65]:


#membuat boxplot penjualan global berdasarkan platfrom
plt.figure(figsize= (20,15))
sns.boxplot(data= dfset, x='platform', y ='total_sales')
plt.show()


# **terlihat banyak sekali outlier sehingga tidak bisa melihat nilai rata2 untuk penjualan setiap platform, jadi mari kita buang terlebih dahulu outliernya**

# In[66]:


#membuang outlier pada kolom total_sales
#membuat variabel baru hanya untuk viasulisasi, untuk melihat bagaimana data penjualan pada tiap platform
Q1 = dfset['total_sales'].quantile(0.25)
Q3 = dfset['total_sales'].quantile(0.75)
IQR = Q3 - Q1

df_out = (dfset['total_sales'] >= Q1 - 1.5 * IQR) & (dfset['total_sales'] <= Q3 + 1.5 *IQR)
df_out = dfset.loc[df_out]


# In[67]:


#melihat lagi bagaimana data pada boxplot yang sudah dihapus outliernya
plt.figure(figsize= (20,15))
sns.boxplot(data= df_out, x='platform', y ='total_sales')
plt.show()


# In[68]:


#melihat penjualan rata2 setiap platfrom
dfset.pivot_table(index='platform', values='total_sales', aggfunc='mean')


# In[69]:


#melihat distribusi penjualan
dfset_pivot.describe()


# **dilihat dari boxplot diatas bahwa penjualan game berdasarkan platfrom sangat berbeda signifikan, terlihat ada yang berhasil memiliki penjualan paling tinggi seperti pada pltfrom Wii dan ada juga yang memiliki penjualan sedikit pada platform DC. Untuk penjualan rata2 dari masing2 platform pun sangat variatif namun tidak jauh berbeda**

# ## Melihat bagaimana ulasan pengguna dan profesional mempengaruhi penjualan

# In[70]:


#membuat variabel baru untuk melihat korelasi antara ulasan pengguna dan para profesional terhadap penjualan
df_corr = dfset[['platform', 'critic_score', 'user_score', 'total_sales', 'genre']]
df_corr


# In[71]:


#membuat scatter plot untuk melihat korelasi antara ulasan user dan profesional terhadap total sales pada seluruh platfrom
pd.plotting.scatter_matrix(df_corr, figsize = (15,10))
plt.show()


# In[72]:


#melihat korelasi
plt.figure(figsize=(10,10))
df_corr.corr()
sns.heatmap(df_corr.corr(), annot=True)
plt.title('Correlation Matrix Plot Maps')
plt.show()


# **secara keseluruhan korelasi antara**
# 1. ulasan pengguna dan penjualan : memiliki korelasi positif sangat lemah yang artinya semakin tinggi score yang diberikan user akan diikuti pula dengan tingginya juga penjualan meskipun korelasinya lemah
# 2. ulasan profesional dan penjual : memliki korelasi positif lemah yang artinya semakin baik ulasan yang diberikan para profesional akan diikuti pulan oleh tingginya penjualan
# 

# **sekarang saya akan melihat korelasi ulasan pengguna dan profesional di salah satu platfrom**

# In[73]:


#memilih platfrom X360 karena platfrom tsb memiliki penjualan paling banyak pada periode tsb
#sekakarag kita membuat variabel baru yang hanya berisi platfrom X360
df_corr_x360 = df_corr.loc[df_corr['platform'] == 'X360']
df_corr_x360


# In[74]:


#melihat korelasi ulasan user dan profesional terhadap penjualan platfrom x360
plt.figure(figsize=(10,10))
df_corr_x360.corr()
sns.heatmap(df_corr_x360.corr(), annot=True)
plt.title('Correlation Matrix Plot Maps For X360')
plt.show()


# In[75]:


#scatter matrix untuk ulasan pengguna dan profesional pada platform X360
pd.plotting.scatter_matrix(df_corr_x360, figsize = (15,10))
plt.show()


# **dilihat dari korelasinya untuk platform X360 itu sendiri memiliki:**
# 1. korelasi ulasan pengguna dan penjualan memiliki korelasi positif sangat lemah sebesar 0.11 yang artinya pada tingginya score yang diberikan oleh pengguna diikuti pula oleh tingginya penjualan
# 2. korelasi ulasan profesional dan penjualan memiliki korelasi positif lemah sebesar 0.31, lebih besar dari pada korelasi ulasan pengguna dengan penjualan, yang artinya semakin tinggi score yang diberikan oleh profesional kepada game daro platform x360, diikuti juga dengan tingginya penjualan

# ## Membandingkan penjualan game yang sama pada platform lain

# In[76]:


#melihat nama game yang sama pada setiap platfrom
df_x360 = dfset.loc[dfset['platform'] == 'X360', 'name'].unique()
dfset.loc[dfset['name'].isin(df_x360)]


# **ternyata ada beberapa nama game yang sama yang di rilis oleh berbagai platfrom, seperti pada game 'Grand Theft Auto V' yang dirilis oleh platfrom X360, PS3**

# **saya akan menampilkan bagaimana penjualan game 'Grand Theft Auto V' pada setiap platform**

# In[77]:


#membandingkan penjualan game yang sama pada platform lain.
#membandingkan penjualan pada game 'Grand Theft Auto V' dengan platform lain
same_game = dfset.loc[dfset['name'] == 'Grand Theft Auto V']
same_game


# **ternyata ada 5 platfrom yang merilis game 'Grand Theft Auto V' dan masing2 memiliki tahun rilis yang berbeda serta penjualan yang berbeda, dari tabel diatas yang memiliki penjualan paling banyak dari game 'Grand Theft Auto V' yaitu platfrom PS3 dengan tahun rilis 2013 dan penjualan sebesar 21.05 juta dolar. lalu ada platfrom PC juga yang merilis game tersebut pada tahun 2015 dengan total penjualan sebesar 1.17 juta dolar. sepertinya game ini sangat menarik banyak peminat sehingga banyak platform yang merilis ulang game tersebut dan tetap menghasilkan penjualan**

# ## Menganalisis genre yang paling menguntungkan

# In[78]:


#membuat tabel untuk menganalisis genre
df_genre = dfset[['platform', 'genre', 'total_sales']]
df_genre


# In[79]:


#melihat bagaimana distribusi genre pada total pendapatan
genre_group = df_genre.groupby('genre')['total_sales'].sum().sort_values(ascending=False).reset_index()
genre_group


# In[80]:


#membuat visualisasi untuk distribusi penjualan berdasarkan genre dengan barplot
plt.figure(figsize=(15,5))
sns.barplot(data=genre_group, x='genre', y='total_sales')
plt.show()


# **dari hasil analisis genre diatas bisa disimpulkan bahwa yang memiliki pendapatan paling tinggi yaitu pada genre action dengan total pendapatan lebih dari 1 milyar dolar, ini menandakan sepertinya para pemain game lebih menyukai game yang ber-genre action. lalu dari data diatas pun bisa dilihat bahwa game yang ber-genre srategy sepertinya kurang diminati karena hanya memiliki penjualan paling rendah yaitu 78.42 juta dolar. sangat signifikan perbedaannya, hal ini dapat diartikan bahwa game yang ber-genre action dapat membawa banyak keuntungan**

# # Pemrofilan pengguna untuk masing-masing wilayah

# ## Melihat variasi pangsa pasar dari satu wilayah ke wilayah lainnya untuk 5 platform teratas

# In[81]:


#melihat lagi bagaimana platform teratas (yang memiliki penjualan terbesar)
dfset.groupby('platform')['total_sales'].sum().sort_values(ascending=False)


# In[82]:


#membuat tabel yang berisi 5 platform teratas
plat_top = ['X360', 'PS3', 'Wii', 'DS', 'PS4']
df_top = dfset.loc[df['platform'].isin(plat_top)]
df_top.head()


# In[83]:


#pivot penjualan pada setiap platform di setiap negara
df_country = df_top.pivot_table(index = 'platform', values=['na_sales', 'eu_sales', 'jp_sales'],
                                  aggfunc='sum').T
df_country


# In[84]:


#melihat pangsa pasar dari satu wilayah ke wilayah lain untuk platfrom DS menggunakan pie chart
df_country.plot.pie(y='DS', figsize=(7, 7), autopct='%1.0f%%', fontsize=14, ylabel='')
plt.title('Pie chart for DS', fontsize=17, y= 1, backgroundcolor='black', color='white')
plt.legend(loc='lower right', bbox_to_anchor=(0.7, 0.1, 0.5, 0.5), prop={'size': 15})
plt.show()


# **pada pie chart diatas bisa dilihat bagaimana penjualan pada masing2 wilayah untuk platfrom DS. Ternyata 51% penyumbang pendapatan ada di North America yaitu sebesar 371.99 juta dolar. untuk penjualan di jepang dan benua eropa tidak terlalu berbeda yaitu 25% pendapatan dari eropa dengan total penjualan 184.48 juta dolar dan 24% sisanya dihasilkan dari jepang dengan total pendapatan 171.35 juta dolar**

# In[85]:


#melihat pangsa pasar dari satu wilayah ke wilayah lain untuk platfrom PS3 menggunakan pie chart
df_country.plot.pie(y='PS3', figsize=(7, 7), autopct='%1.0f%%', fontsize=14, ylabel='')
plt.title('Pie chart for PS3', fontsize=17, y= 1, backgroundcolor='black', color='white')
plt.legend(loc='lower right', bbox_to_anchor=(0.7, 0.1, 0.5, 0.5), prop={'size': 14})
plt.show()


# **lalu untuk platfrom PS3 pun North America masih menjadi penyumbang penjualan terbesar yaitu sekitar 49% dari total penjualan North America mendapat penjualan game sebesar 390.19 juta dolar, lalu penyumbang penjualan kedua ada eropa yang menyumbang 41% dari total pendapatan penjualan game pada platfrom PS3, dan yang terakhir jepang yang menyumbang 10% dari total pendapatan yaitu sekitar 79.41 juta dolar**

# In[86]:


#melihat pangsa pasar dari satu wilayah ke wilayah lain untuk platfrom PS4 menggunakan pie chart
df_country.plot.pie(y='PS4', figsize=(7, 7), autopct='%1.0f%%',fontsize=14, ylabel='')
plt.title('Pie chart for PS4', fontsize=17, y= 1.03, backgroundcolor='black', color='white')
plt.legend(loc='lower right', bbox_to_anchor=(0.7, 0.1, 0.5, 0.5), prop={'size': 14})
plt.show()


# **selanjutnya pada pie chart diatas menunjukkan pangsa pasar pada plaform PS4 yang ternyata penjualan paling tinggi didapat dari benua eropa yang menyumbang 53% dari total penjualan keseluruhan sebanyak 141.09 juta dolar, lalu yang kedua north america yang menyumbang 41% dari total pengjualan sekitar 108.74 juta dolar, dan yang ketiga ada jepang yang menyumbang sekitar 6% dari total penghasilan.**
# 
# **padahal dari platform sebelumnya yaitu PS3 north america menyumbang pendapatan paling tinggi, namun pada platform terusannya yaitu PS4 yang menyumbang paling tinggi adalah eropa yang sebelumnya menyumbangkan pendapatan ke-2 pada platform PS3, mungkin orang2 jepang mulai tertaik game yang berada di PS4**

# In[87]:


#melihat pangsa pasar dari satu wilayah ke wilayah lain untuk platfrom Wii menggunakan pie chart
df_country.plot.pie(y='Wii', figsize=(7, 7), autopct='%1.0f%%',fontsize=14, ylabel='')
plt.title('Pie chart for Wii', fontsize=17, y= 1.01, backgroundcolor='black', color='white')
plt.legend(loc='lower right', bbox_to_anchor=(0.7, 0.1, 0.5, 0.5), prop={'size': 14})
plt.show()


# **untuk platfrom Wii, pie chart diatas menunjukkan bahwa penjualan paling besar berada diwiliyah north america sebesar 60% atau sebesar 486.87 juta dolar, dan yang paling kecil berada di wilayah jepang sebesar 8% sebesar 68.28 juta dolar**

# In[88]:


#melihat pangsa pasar dari satu wilayah ke wilayah lain untuk platfrom X360 menggunakan pie chart
df_country.plot.pie(y='X360', figsize=(7, 7), autopct='%1.0f%%', fontsize=14, ylabel='')
plt.title('Pie chart for X360', fontsize=17, y= 1.01, backgroundcolor='black', color='white')
plt.legend(loc='lower right', bbox_to_anchor=(0.7, 0.1, 0.5, 0.5), prop={'size': 14})
plt.show()


# **terakhir untuk pangsa pasar pada platform xbox 360, lagi-lagi north america yang menyumbang pendapatan paling besar yaitu sekitar 68% atau sebanyak 595.74 juta dolar, dan jepang hanya menyumbang pendapatan 1%**

# **kesimpulan dari variasi pangsa pasar ini yakni penjualan paling besar di dapat dari wilayah north america, lalu eropa, dan yang terakhir yaitu jepang**

# ## Melihat pangsa pasar berdasarkan genre

# In[89]:


#melihat bagaimana genre teratas (yang memiliki penjualan terbesar)
genre_group = dfset.groupby('genre')['total_sales'].sum().sort_values(ascending=False)
genre_group


# In[90]:


#membuat tabel yang berisi 5 genre teratas
genre_top = ['Action', 'Sports', 'Shooter', 'Misc', 'Role-Playing']
df_genre = dfset.loc[df['genre'].isin(genre_top)]
df_genre.head()


# In[91]:


#pivot setiap genre
df_genre = df_genre.pivot_table(index = 'genre', values=['na_sales', 'eu_sales', 'jp_sales'],
                                  aggfunc='sum').T.reset_index()
df_genre


# In[92]:


#membuat pie chart untuk melihat pangsa pasar antar wilayah berdasarkan genre
i = 1
plt.figure(figsize=(15, 20))
# plotting data on chart
for genre in genre_top:
    plt.subplot(3, 2, i)
    plt.title(f'Pie chart market share by Genre {genre}', fontsize=15)
    plt.pie(df_genre[genre], labels=df_genre['index'], autopct='%.0f%%') # tambahkan attribut `textprops={'fontsize': 14}` ~ Chamdani
    plt.legend(loc='lower right', bbox_to_anchor=(0.7, 0.1, 0.5, 0.5), prop={'size': 12})
    i = i + 1 

plt.show()


# **pie chart diatas menunjukkan bahwa dari ke 5 top genre (Action, Misc, Role-Playing, Shooter, Sports) yang di analisis ternyata penjualan paling banyak berada di wilayah North america, lalu eropa di urutan kedua dan jepang di urutan ketiga. namun pada genre role_playing di wilayah jepang menempati urutan kedua penjualan tertinggi setelah north america, mungkin masyarakat jepang banyak yg tertarik pada game yang ber-genre role-playing selain action**

# ## Menganalisis Rating ESRB terhadap penjualan di masing2 wilayah

# **ESRB adalah sebuah organisasi yang menetapkan usia dan konten untuk sebuah game. Berikut rating yang diberikan:**
# 1. E (everyone) : game tersebut cocok untuk segala usia
# 2. M (Mature) : game tersebut berisi konten yang cocok untuk usia 17 tahun dan diatasnya 
# 3. T (Teen) : game tersebut berisi konten yang cocok untuk usia 13 tahun dan diatasnya
# 4. AO (Adult Only) : game tersebut berisi kontent yang cocok untuk usia 18 tahun dan diatasnya
# 5. E10+ (Everyone 10+) : game tersebut berisi konten yang cocok untuk usia 10 tahun dan diatasnya 
# 6. EC (Early Childhood) : game ini berisi konten yang cocok untuk anak pra sekolah
# 7. RP (Rating Pending) : rating ini untuk game yang masih promosi dan belum mendapatkan rating dari ESRB, namun diperikirakan akan membawa rating dewasa (Mature)
# 
# selain itu ada nilai 'na' pada kolom rating, itu saya isi untuk mengganti nilai hilang, yang artinya memang tidak diberikan rating atau not available

# In[93]:


#melihat nilai unik dari kolom rating
dfset['rating'].unique()


# In[94]:


#membuat variabel baru untuk menganalisis rating
df_rating = dfset[['platform', 'total_sales', 'rating', 'na_sales', 'jp_sales', 'eu_sales', 'total_sales']]
df_rating.head()


# In[95]:


#membuat pivot tabel penjualan di wilayah north america berdasarkan rating
rating_pivot_na = df_rating.pivot_table(index='rating', values='na_sales', aggfunc='sum').reset_index()
rating_pivot_na


# In[96]:


#membuat chart untuk melihat analisis data berdasarkan rating di wilayah north america
plt.figure(figsize =(6,6))
sns.barplot(data=rating_pivot_na, x= 'rating', y="na_sales",
           order=rating_pivot_na.sort_values('na_sales').rating)
plt.show()


# **dari data pivot juga bar chart diatas untuk penjualan di wilayah north america bisa dilihat bahwa yang memiliki rating E dari ESRB memiliki penjualan paling banyak yakni sebesar 873.37 juta dolar. rating E ini artinya game yang dirilis dan mempunyai rating ini bisa dimainkan oleh semua kalangan dari semua umur, oleh karena itu tidak heran game dengan label rating E memiliki penjualan paling besar**

# In[97]:


#membuat pivot penjualan di wilayah jepang berdasarkan rating
rating_pivot_jp = df_rating.pivot_table(index='rating', values='jp_sales', aggfunc='sum').reset_index()
rating_pivot_jp


# In[98]:


#membuat chart untuk memvisualisasikan data pada penjualan di wilayah jepang berdasarkan rating
plt.figure(figsize =(6,6))
sns.barplot(data=rating_pivot_jp, x= 'rating', y="jp_sales",
           order=rating_pivot_jp.sort_values('jp_sales').rating)
plt.show()


# **sekarang untuk wilayah jepang ternyata banyak penjualan game yang tidak memiliki label rating dari ESRB, yaitu pengahasilan sebanyak 291.69 juta dolar, selain itu untuk game dengan rating E tetap menghasilkan penjualan terbanyak sebanyak 147.33 juta dolar**

# In[99]:


#membuat pivot penjualan di wilayah eroap berdasarkan rating
rating_pivot_eu = df_rating.pivot_table(index='rating', values='eu_sales', aggfunc='sum').reset_index()
rating_pivot_eu


# In[100]:


#membuat chart untuk memvisualisasikan data pada penjualan di wilayah eropa berdasarkan rating
plt.figure(figsize =(6,6))
sns.barplot(data=rating_pivot_eu, x= 'rating', y='eu_sales',
           order=rating_pivot_eu.sort_values('eu_sales').rating)
plt.show()


# **dari wilayah eropa pun game dengan rating E yang artinya bisa dimainkan oleh semua kalangan umur tetap memiliki penjualan paling besar**

# In[101]:


#sekarang melihat penjualan dari semua wilayah berdasarkan rating
#membuat pivot tabel penjualan semua wilayah berdasarkan rating
rating_pivot_total = df_rating.pivot_table(index='rating', values='total_sales', aggfunc='sum').reset_index()
rating_pivot_total.columns = ['rating', 'total_sales', 'count']
rating_pivot_total


# In[102]:


#membuat chart untuk memvisualisasikan data pada penjualan di semua wilayah berdasarkan rating
plt.figure(figsize =(6,6))
sns.barplot(data=rating_pivot_total, x= 'rating', y='total_sales',
           order=rating_pivot_total.sort_values('total_sales').rating)
plt.show()


# **ternyata dari semua wilayah, game dengan rating E memiliki penjualan yang paling banyak, lalu di iiktui oleh rating M, T, E10+**

# # Menguji hipotesis

# **disini saya akan menguji 2 hipotesis tentang rata2 rating pengguna pada 2 platfrom dan juga rating pengguna berdasarkan 2 genre**
# 
# **saya akan menentukan tingkat signifikansi sebanyak 5% dimana saya mengambil keputusan untuk menolak Hipotesis sebanyak-banyak 5% dan mengambil keputusan sedikitnya 95% dari total peluang**
# 
# **metode yang dilakukan dalam melakuka hipotesis yaitu dengan menggunakan t-test independen karena menggunakan 2 variabel**

# ### Hipotesis 1

# **untuk hipotesis pertama yaitu saya ingin mengetahui rata -rata rating pengguna 2 platform yaitu Xbox One dan PC apakah sama atau tidak. maka dirumsukan hipotesis sebagai berikut**
# 
# H0 = Rata-rata rating pengguna platform Xbox One dan PC adalah sama.
# 
# H1 = Rata-rata rating pengguna platform Xbox One dan PC adalah tidak sama.

# In[103]:


#membuat variabel baru yang berisi platfrom yang akan di uji
df_p = df.loc[(df['platform'] == 'XOne') | (df['platform'] == 'PC')]
df_p.head()


# **diatas adalah tabel yang berisi dengan 2 platfrom yang dibutuhkan untuk hipotesis**

# In[104]:


#memilih kolom platfrom dan ulasan pengguna untuk memudahkan analisis hipotesis
df_p = df_p[['platform', 'user_score']]
df_p.head()


# In[105]:


#membuat variabel baru yang hanya berisi platform PC
pc_group = df_p.loc[df_p['platform']=='PC']
pc_group.head()


# In[106]:


#mleihat nilai hilang pada tabel pc_group
pc_group.isna().sum()


# **nilai hilang harus dihapuskan dari data tersebut agar lebih akurat dalam mengambil keputusan berdasarkan hipotesis, tidak diubah menjadi nilai baru karena bisa mengubah distribusi datanya**

# In[107]:


#menghapus nilai hilang pada tabel pc_group
pc_group = pc_group.dropna()
pc_group.isna().sum()


# In[108]:


#melihat distribusi data pada tabel pc_group
pc_group.describe()


# In[109]:


#menentukan varian pada tabel pc_group
np.var(pc_group)


# In[110]:


#sekarang membuat tabel yang berisi platfrom Xbox One
x_group = df_p.loc[df_p['platform']=='XOne']
x_group.head()


# In[111]:


#melihat nilai hilang pada tabel x_group
x_group.isna().sum()


# In[112]:


#mengahapus nilai hilang pada tabel x_group
#mengecek kembali apakah masih ada nilai yang hilang
x_group = x_group.dropna()
x_group.isna().sum()


# In[113]:


#melihat distribusi data pada tabel x_group
x_group.describe()


# In[114]:


#menentukan varian dari tabel x_group
np.var(x_group)


# **datanya sudah siap, sekarang saya akan melakukan pengujian hipotesis**
# 
# H0 : Rata-rata rating pengguna platform Xbox One dan PC adalah sama.
# 
# H1 : Rata-rata rating pengguna platform Xbox One dan PC adalah tidak sama.

# In[115]:


#menentukan alpha 5%
#menguji nilai peluang
alpha = 0.05
results = st.ttest_ind(pc_group['user_score'], x_group['user_score'] ,equal_var=False)
results.pvalue


# In[116]:


#melihat bagaiman hasil hipoesis
if (results.pvalue < alpha):
    print('Kita menolak hipotesis nol')
else:
    print('Kita tidak dapat menolak hipotesis nol')


# **dari hasil pengujian hipotesis menunjukkan kita menolak H0 yang artinya rating rata2 dari PC dan Xbox One tidak sama, dimana rata2 rating user PC lebih banyak dari Xbox one.p-value 4.9350723 juga menunjukkan kita menerima H1 yang memang rata2 dari kedua plafrom tersebut berbeda**
# 
# **rata2 rating pengguna user di PC adalah 7.062468 dan variansinya adalah 2.337747, lalu rata2 rating pengguna user di Xbox One adalah 6.521429 dan variansinya adalah 1.896519, meskipun rata2nya tidak berbeda jauh namun variannsi berbeda**
# 
# **mungkin game di PC lebih banyak disukai oleh pengguna, atau bisa saja para pengguna Xbox One sangat kritis dalam menilai suatu permainan**

# ### Hipotesis 2

# **hipotesis kedua untuk mengetahui apakah rata2 rating pengguna berdasarkan genre sama atau tidak. saya akan memilih genre yang paling banyak diminati yaitu genre Action dan Sports, perumusan hipotesis nol dan hipotesis alternatifnya sebagai berikut:**
# 
# H0 :  Rata-rata rating pengguna genre Action dan Sports sama.
#     
# H1: Rata-rata rating pengguna genre Action dan Sports berbeda.

# In[117]:


#membuat variabel baru yang berisi genre Action dan sports
genre_h = df.loc[(df['genre'] == 'Action') | (df['genre'] == 'Sports')]
genre_h.head()


# In[118]:


#membuat variabeli baru untuk genre action saja
genre_act = df.loc[df['genre'] == 'Action']
genre_act.head()


# In[119]:


#membuat tabel yang berisi genre dan rating pengguna
#melihat apakah ada nilai hilang
genre_act = genre_act[['genre', 'user_score']]
genre_act.isna().sum()


# In[120]:


#mengahpus nilai hilang
#mengecek kembali nilai hilang
genre_act = genre_act.dropna()
genre_act.isna().sum()


# In[121]:


#menghitung variansi dari tabel genre_action
np.var(genre_act)


# In[122]:


#melihat distribusi data dari tabel genre_act
genre_act.describe()


# In[123]:


#membuat tabel yang berisi genre sport dan kolom ulasan pengguna
genre_sp = df.loc[df['genre'] == 'Sports']
genre_sp = genre_sp[['genre', 'user_score']]
genre_sp.head()


# In[124]:


#menghapus nilai hilang pada tabel genre_sp
#mengecek kembali apakah masih ada nilai hilang pada tabel genre_sp
genre_sp = genre_sp.dropna()
genre_sp.isna().sum()


# In[125]:


#menghitung variansi data genre_sp
np.var(genre_sp)


# In[126]:


#melihat distribusi data pad tabel genre_sp
genre_sp.describe()


# **data sudah siap, saatnya melakukan pengujian hipotesis**
# 
# H0 :  Rata-rata rating pengguna genre Action dan Sports sama.
#     
# H1: Rata-rata rating pengguna genre Action dan Sports berbeda.

# In[127]:


#menguji hipotesis dengan signifikansi 5%
alpha = 0.05
results = st.ttest_ind(genre_act['user_score'], genre_sp['user_score'] ,equal_var=False)
results.pvalue


# In[128]:


#melihat hasil peluang terhadapa alpha
if (results.pvalue < alpha):
    print('Kita menolak hipotesis nol')
else:
    print('Kita tidak dapat menolak hipotesis nol')


# **dari hasil uji hipotesis kedua menunjukkan bahwa kita tidak dapat menolak H0 yang artinya rating pengguna dari genre action dan sports adalah sama, yaitu rata rating dari action adalah 7.05 dan rata2 rating sports adalah 6.96, memang sangat kecil sekali perbedaannya.**

# # kesimpulan umum

# **dari hasil analisis diatas bisa dilihat bagaiamana suatu game dapat memiliki penjualan tinggi karena ada beberapa faktor**
# 1. usia sebuah platform
#     >ternyata sebuah platform atau publisher sangat berpengaruh terhadap penjualan game-game yang akan dirilis, dimana akan menjual game sebaiknya memilih platform yang berusia muda atau <10 tahun.
# 2. ulasan pengguna dan profesional
#     >ulasan para pengguna dan profesional sedikit banyak juga berpengaruh terhadap penjualan game. oleh karena itu kita bisa meminta ulasan-ulasan dari para pengguna dan profesional mengenai game yang akan kita rilis
# 3. rating ESRB
#     >dilihat dari rating ESRB juga mempengaruhi penjualan, dimana sebaiknya kita menjual game yang bisa di mainkan oleh semua kalangan yang memiliki rating E jadi setiap orang dari segala umur bisa membeli dan memainkan game kita.
# 4. genre
#     >genre yang paling banyak diminati juga ternyata genre action dan sports. jika kita akan menjual game mungkin bisa dipertimbang untuk kedua genre tersebut yang memiliki banyak peminat
# 5. wilayah
#     >seperti pada hasil analisis diatas untuk melakukan iklan kampanye game sebaiknya dimulai dari wilayah yang memang memiliki antusiasme tinggi dalam bermain game seperti wilayah north america. wilayah tersebut bahkan menyumbang pendapatan terbesar daripada wilayah lain.

# In[ ]:




