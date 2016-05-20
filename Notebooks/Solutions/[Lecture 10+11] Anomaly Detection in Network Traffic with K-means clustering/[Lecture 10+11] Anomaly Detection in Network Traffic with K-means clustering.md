
<div>
<h1>Run the cell below to generate the road map (do not modify it)</h1></div>


```python
%%javascript
var kernel = IPython.notebook.kernel;var thename = window.document.getElementById("notebook_name").innerHTML;var command = "THE_NOTEBOOK = " + "'"+thename+"'";kernel.execute(command);command="os.environ['THE_NOTEBOOK'] = THE_NOTEBOOK";kernel.execute(command);var cell = IPython.notebook.get_cell(2);cell.execute();IPython.notebook.get_cell(3).focus_cell();var x = $('.code_cell');$(x[1]).children('.input').hide();
```


    <IPython.core.display.Javascript object>



```python
outputdir = "/tmp/tools/"
!mkdir -p $outputdir
!wget "https://www.dropbox.com/s/4g0pigmro4vo1b4/menutemplate?dl=0" -O /tmp/tools/menutemplate >> /tmp/toollog 2>&1 
!wget "https://www.dropbox.com/s/3flttpzhsja8td7/construct_menu.py?dl=0" -O /tmp/tools/construct_menu.py >> /tmp/toollog 2>&1 
!python /tmp/tools/construct_menu.py "{THE_NOTEBOOK}.ipynb" {outputdir}
from IPython.core.display import HTML
output_file_name = outputdir + THE_NOTEBOOK.replace(" ", "").replace("[", "").replace("]", "") + ".ipynb.html"
with open(output_file_name) as fp:
    html = fp.read()
HTML(html)
```





   <script src="http://code.jquery.com/jquery-latest.min.js" type="text/javascript"></script>
   <script>
      (function($){
         $(document).ready(function(){

         $('#cssmenu li.active').addClass('open').children('ul').show();
            $('#cssmenu li.has-sub>a').on('click', function(){
               $(this).removeAttr('href');
               var margin_left = parseInt($(this).css('padding-left').replace('px', ''));
               var element = $(this).parent('li');
               if (element.hasClass('open')) {
                  element.removeClass('open');
                  element.find('li').removeClass('open');
                  element.find('ul').slideUp(200);
               }
               else {
                  element.addClass('open');
                  element.children('ul').slideDown(200);
                  element.siblings('li').children('ul').slideUp(200);
                  element.siblings('li').removeClass('open');
                  element.siblings('li').find('li').removeClass('open');
                  element.siblings('li').find('ul').slideUp(200);
                  element.children().find('a').css('padding-left', margin_left + 16 + "px");
               }
            });

            var screen_height = $(window).height(); 
            var headerHeight = $("#header").outerHeight();
            var max_height_menu = screen_height - headerHeight;
            $("#cssmenu").css('max-height', max_height_menu + 'px');
            $("#cssmenu").css('top', headerHeight + 'px');
            $("#menu-activator").css('top', headerHeight + 'px');
            $("#cssmenu").show();
            $("#menu-activator").hide();


            $("#menu-activator").mouseover(function(){
              $("#cssmenu").show();
              $("#menu-activator").hide();
            });


            $("#cssmenu").mouseleave(function(){
              $("#cssmenu").hide();
              $("#menu-activator").show();
            });

            $("div.anchor").parent('div').addClass('bcolor');

         });
      })(jQuery);
   </script>
   <style>
      @import url(http://fonts.googleapis.com/css?family=Raleway:400,200);
      #cssmenu,
      #cssmenu ul,
      #cssmenu ul li,
      #cssmenu ul li a {
        margin: 0;
        padding: 0;
        border: 0;
        list-style: none;
        line-height: 1;
        display: block;
        position: relative;
        -webkit-box-sizing: border-box;
        -moz-box-sizing: border-box;
        box-sizing: border-box;
      }
      #cssmenu {
        width: 220px;
        font-family: Raleway, sans-serif;
        color: #ffffff;
        max-height:600px;
        overflow-y:scroll;
        overflow-x:hidden;
        position:fixed;
        top:130px;
        left:0px;
      }
      #cssmenu ul ul {
        display: none;
      }
      #cssmenu > ul > li.active > ul {
        display: block;
      }
      .align-right {
        float: right;
      }
      #cssmenu > ul > li > a {
        padding: 16px 22px;
        cursor: pointer;
        z-index: 2;
        font-size: 16px;
        text-decoration: none;
        color: #ffffff;
        background: #565656;
        -webkit-transition: color .2s ease;
        -o-transition: color .2s ease;
        transition: color .2s ease;
      }
      #cssmenu > ul > li > a:hover {
        color: #d8f3f0;
      }
      #cssmenu ul > li.has-sub > a:after {
        position: absolute;
        right: 26px;
        top: 19px;
        z-index: 5;
        display: block;
        height: 10px;
        width: 2px;
        background: #ffffff;
        content: "";
        -webkit-transition: all 0.1s ease-out;
        -moz-transition: all 0.1s ease-out;
        -ms-transition: all 0.1s ease-out;
        -o-transition: all 0.1s ease-out;
        transition: all 0.1s ease-out;
      }
      #cssmenu ul > li.has-sub > a:before {
        position: absolute;
        right: 22px;
        top: 23px;
        display: block;
        width: 10px;
        height: 2px;
        background: #ffffff;
        content: "";
        -webkit-transition: all 0.1s ease-out;
        -moz-transition: all 0.1s ease-out;
        -ms-transition: all 0.1s ease-out;
        -o-transition: all 0.1s ease-out;
        transition: all 0.1s ease-out;
      }
      #cssmenu ul > li.has-sub.open > a:after,
      #cssmenu ul > li.has-sub.open > a:before {
        -webkit-transform: rotate(45deg);
        -moz-transform: rotate(45deg);
        -ms-transform: rotate(45deg);
        -o-transform: rotate(45deg);
        transform: rotate(45deg);
      }
      #cssmenu ul ul li a {
        padding: 14px 22px;
        cursor: pointer;
        z-index: 2;
        font-size: 14px;
        text-decoration: none;
        color: #dddddd;
        background: #49505a;
        -webkit-transition: color .2s ease;
        -o-transition: color .2s ease;
        transition: color .2s ease;
      }
      #cssmenu ul ul ul li a {
        /*padding-left: 32px;*/
      }
      #cssmenu ul ul li a:hover {
        color: #ffffff;
      }
      #cssmenu ul ul > li.has-sub > a:after {
        top: 16px;
        right: 26px;
        background: #dddddd;
      }
      #cssmenu ul ul > li.has-sub > a:before {
        top: 20px;
        background: #dddddd;
      }
      #menu-activator {
        position: fixed;
        top: 130px;
        left: 0px;
        width: 60px;
        padding: 16px 20px;
        display: block;
        background-color: #565656;
        font-size: 16px;
        color: white;
      }

      div.bcolor, span.bcolor {
      	border-radius: 15px; 
      	background: rgb(224,239,130);
      	padding:10px;
      }

      div.answer, div.comment, div.note {
        background-color: rgb(255, 235, 218);
        padding: 10px;
        border-radius: 5px;
      }

      div.comment::before, div.answer::before, div.note::before {
          background-color: yellow;
          color: red;
          font-weight: bold;
          margin-right:5px;
          padding:3px;
      }

      div.comment::before {
        content: "Comment:";
      }

      div.answer::before {
        content: "Answer:";
      }

      div.note::before {
        content: "Note:";
      }

      @media print{
      	#cssmenu {
      		display:none;
      	}
      }
   </style>
   <title>Roadmap</title>
</head>
<body>
<div id="menu-activator">O</div>
<div id='cssmenu'>
<ul>
<li class='has-sub'><a href="#">ROADMAP</a>
<ul><li class='has-sub'><a href='#Anomaly-Detection-in-Network-Traffic-with-K-means-clustering'>Anomaly Detection in Network Traffic with K-means clustering</a><ul><li><a href='#Goals'>Goals</a></li><li><a href='#Steps'>Steps</a></li></ul></li><li class='has-sub'><a href='#1.-K-means'>1. K-means</a><ul><li><a href='#1.1.-Introduction'>1.1. Introduction</a></li><li><a href='#1.2.-How-does-it-work-?'>1.2. How does it work ?</a></li><li class='has-sub'><a href='#1.3.-Illustrative-example'>1.3. Illustrative example</a><ul><li class='has-sub'><a href='#Question-1'>Question 1</a><ul><li><a href='#Question-1.1'>Question 1.1</a></li><li><a href='#Question-1.2'>Question 1.2</a></li><li><a href='#Question-1.3'>Question 1.3</a></li><li><a href='#Question-1.4'>Question 1.4</a></li><li><a href='#Question-1.5'>Question 1.5</a></li><li><a href='#Question-1.6'>Question 1.6</a></li><li><a href='#Question-1.7'>Question 1.7</a></li></ul></li></ul></li></ul></li><li class='has-sub'><a href='#2.-Usecase:-Network-Intrusion'>2. Usecase: Network Intrusion</a><ul><li><a href='#2.1.-Data'>2.1. Data</a></li><li class='has-sub'><a href='#2.2.-Clustering-without-using-categorical-features'>2.2. Clustering without using categorical features</a><ul><li class='has-sub'><a href='#2.2.1.-Loading-data'>2.2.1. Loading data</a><ul><li><a href='#Question-2'>Question 2</a></li><li><a href='#Question-3'>Question 3</a></li></ul></li><li class='has-sub'><a href='#2.2.3.-Evaluating-model'>2.2.3. Evaluating model</a><ul><li><a href='#Question-4'>Question 4</a></li><li><a href='#Question-5'>Question 5</a></li></ul></li><li class='has-sub'><a href='#2.2.4.-Choosing-K'>2.2.4. Choosing K</a><ul><li><a href='#Question-6'>Question 6</a></li></ul></li><li class='has-sub'><a href='#2.2.5-Normalizing-features'>2.2.5 Normalizing features</a><ul><li><a href='#Question-7'>Question 7</a></li><li><a href='#Question-8'>Question 8</a></li><li><a href='#Question-9'>Question 9</a></li><li><a href='#to-be-continued-####'>to be continued ####</a></li></ul></li></ul></li><li class='has-sub'><a href='#2.3.-Clustering-using-categorical-features'>2.3. Clustering using categorical features</a><ul><li class='has-sub'><a href='#2.3.1-Loading-data'>2.3.1 Loading data</a><ul><li><a href='#Question-10'>Question 10</a></li></ul></li><li class='has-sub'><a href='#2.3.2.-Building-models'>2.3.2. Building models</a><ul><li><a href='#Question-11'>Question 11</a></li></ul></li></ul></li><li class='has-sub'><a href='#2.4.-Anomaly-detection'>2.4. Anomaly detection</a><ul><li><a href='#Question-12'>Question 12</a></li><li><a href='#Question-13'>Question 13</a></li><li><a href='#Question-14'>Question 14</a></li></ul></li></ul></li></ul>
</li>
<li class='has-sub'><a href="#">QUESTIONS</a>
<ul><li class='has-sub'><a href='#Question-1'>Question 1</a><ul><li><a href='#Question-1.1'>Question 1.1</a></li><li><a href='#Question-1.2'>Question 1.2</a></li><li><a href='#Question-1.3'>Question 1.3</a></li><li><a href='#Question-1.4'>Question 1.4</a></li><li><a href='#Question-1.5'>Question 1.5</a></li><li><a href='#Question-1.6'>Question 1.6</a></li><li><a href='#Question-1.7'>Question 1.7</a></li></ul></li><li><a href='#Question-2'>Question 2</a></li><li><a href='#Question-3'>Question 3</a></li><li><a href='#Question-4'>Question 4</a></li><li><a href='#Question-5'>Question 5</a></li><li><a href='#Question-6'>Question 6</a></li><li><a href='#Question-7'>Question 7</a></li><li><a href='#Question-8'>Question 8</a></li><li><a href='#Question-9'>Question 9</a></li><li><a href='#Question-10'>Question 10</a></li><li><a href='#Question-11'>Question 11</a></li><li><a href='#Question-12'>Question 12</a></li><li><a href='#Question-13'>Question 13</a></li><li><a href='#Question-14'>Question 14</a></li></ul>
</li>
</ul>
</div>





# Anomaly Detection in Network Traffic with K-means clustering

We can categorize machine learning algorithms into two main groups: **supervised learning** and **unsupervised learning**. With supervised learning algorithms, in order to predict unknown values for new data, we have to know the target value for many previously-seen examples. In contrast, unsupervised learning algorithms explore the data which has no target attribute to find some intrinsic structures in them.

Clustering is a technique for finding similar groups in data, called **clusters**. Clustering is often called an unsupervised learning task as no class values denoting an a priori grouping of the data instances are given.

In this notebook, we will use K-means, a very well known clustering algorithm to detect anomaly network connections based on statistics about each of them. For a thorough overview of K-means clustering, from a research perspective, have a look at this wonderful [tutorial](http://theory.stanford.edu/~sergei/slides/kdd10-thclust.pdf).

## Goals
We expect students to:
* Learn (or revise) and understand the K-means algorithm
* Implement a simple K-means algorithm
* Use K-means to detect anomalies network connection data

## Steps

1. In section 1, we will have an overview about K-means then implement a simple version of it.
2. In section 2, we build models with and without categorical features.
3. Finally, in the last section, using our models, we will detect unsual connections.

# 1. K-means
## 1.1. Introduction
Clustering is a typical and well-known type of unsupervised learning. Clustering algorithms try to find natural groupings in data. Similar data points (according to some notion of similarity) are considered in the same group. We call these groups **clusters**.

K-Means clustering is a simple and widely-used clustering algorithm. Given value of $k$, it tries to build $k$ clusters from samples in the dataset. Therefore, $k$ is an hyperparameter of the model. The right value of $k$ is not easy to determine, as it highly depends on the data set and the way that data is featurized.

To measue the similarity between any two data points, K-means requires the definition of a distance funtion between data points. What is a distance? It is a value that indicates how close two data points are in their space. In particular, when data points lie in a $d$-dimensional space, the Euclidean distance is a good choice of a distance function, and is supported by MLLIB.

In K-means, a cluster is a group of points, with a representative entity called a centroid. A centroid is also a point in the data space: the center of all the points that make up the cluster. It's defined to be the arithmetic mean of the points. In general, when working with K-means, each data sample is represented in a $d$-dimensional numeric vector, for which it is easier to define an appropriate distance function. As a consequence, in some applications, the original data must be transformed into a different representation, to fit the requirements of K-means.

## 1.2. How does it work ?
Given $k$, the K-means algorithm works as follows:

1. Randomly choose $k$ data points (seeds) to be the initial centroids
2. Assign each data point to the **closest centroid**
3. Re-compute (update) the centroids using the current cluster memberships
4. If a convergence criterion is not met, go to step 2

We can also terminate the algorithm when it reaches an iteration budget, which yields an approximate result.
From the pseudo-code of the algorithm, we can see that K-means clustering results can be sensitive to the order in which data samples in the data set are explored. A sensible practice would be to run the analysis several times, randomizing objects order; then, average the cluster centres of those runs and input the centres as initial ones for one final run of the analysis.

## 1.3. Illustrative example
One of the best ways to study an algorithm is trying implement it.
In this section, we will go step by step to implement a simple K-means algorithm.

![](https://farm2.staticflickr.com/1604/24934700445_833f0a5649_t.jpg)

<div class='anchor' ></div>
### Question 1

#### Question 1.1
Complete the below function to calculate an euclidean distance between any two points in $d$-dimensional data space


```python
import numpy as np

# calculate distance between two d-dimensional points
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum([np.power(p1[i] - p2[i], 2) for i in range(len(p1))]))

# test our function
assert (round(euclidean_distance([1,2,3] , [10,18,12]), 2) == 20.45), "Function's wrong"
```

<div class='anchor' ></div>

#### Question 1.2
Given a data point and the current set of centroids, complete the function below to find the index of the closest centroid for that data point.


```python
def find_closest_centroid(datapoint, centroids):
    # find the index of the closest centroid of the given data point.
    distances = list(map(lambda centroid: euclidean_distance(datapoint, centroid), centroids))
    return np.argmin(distances)

assert(find_closest_centroid( [1,1,1], [ [2,1,2], [1,2,1], [3,1,2] ] ) == 1), "Function's wrong"
```

<div class='anchor' ></div>

#### Question 1.3
Write a function to randomize $k$ initial centroids.


```python
np.random.seed(22324)

# randomize initial centroids
def randomize_centroids(data, k):
    centroids = list(range(len(data)))
    np.random.shuffle(centroids)
    return [data[centroid] for centroid in centroids[:k]]

assert(len(
    randomize_centroids(
        np.array([ 
            np.array([2,1,2]), 
            np.array([1,2,1]), 
            np.array([3,1,2]) 
             ]), 
        2)) == 2), "Wrong function"
```

<div class='anchor' ></div>

#### Question 1.4
Write function `check_converge` to check the stop creteria of the algorithm.


```python
MAX_ITERATIONS = 1000

# return True if clusters have converged , otherwise, return False  
def check_converge(centroids, old_centroids, num_iterations, threshold=0):
    # if it reaches an iteration budget
    if(num_iterations < MAX_ITERATIONS):
        return True
    # check if the centroids don't move (or very slightly)
    return (len(list(filter(lambda x: euclidean_distance(x[0], x[1])>threshold ,zip(centroids, old_centroids)))) == 0)
```

<div class='anchor' ></div>

#### Question 1.5
Write function `update_centroid` to update the new positions for the current centroids based on the position of their members.


```python
# centroids: a list of centers
# cluster: a list of k elements. Each element i-th is a list of data points that are assigned to center i-th
def update_centroids(centroids, cluster):
    new_centroids = []
    for i in range(len(centroids)):
        new_centroids.append(list(np.average(cluster[i],axis=0)))
    return new_centroids

```

<div class='anchor' ></div>

#### Question 1.6
Complete the K-means algorithm scheleton below, with the functions you wrote above.


```python
# data : set of data points
# k : number of clusters
# centroids: initial list of centroids
def kmeans(data, k=2, centroids=None):
    
    # randomize the centroids if they are not given
    if not centroids:
        centroids = randomize_centroids(data, k)

    old_centroids = centroids[:]

    iterations = 0
    while True:
        iterations += 1

        # init empty clusters
        clusters = [[] for i in range(k)]

        # assign each data point to the closest centroid
        for datapoint in data:
            # find the closest center of each data point
            centroid_idx = find_closest_centroid(datapoint, centroids)
            
            # assign datapoint to the closest cluster
            clusters[centroid_idx].append(datapoint)
        
        # keep the current position of centroids before changing them
        old_centroids = centroids[:]
        
        # update centroids
        centroids = update_centroids(centroids, clusters)
        
        # if the stop criteria are met, stop the algorithm
        if check_converge(centroids, old_centroids, iterations, threshold=0):
            break
    
    return centroids
```

Next, we will test our algorithm on [Fisher's Iris dataset](http://en.wikipedia.org/wiki/Iris_flower_data_set), and plot the resulting clusters in 3D.

<div class='anchor' ></div>

#### Question 1.7
The code below can be used to test your algorithm with three different datasets: Iris, Moon and Blob.
Run your algorithm to cluster datapoints in these datasets, plot the results and discuss about them. Do you think that our algorithm works well? Why?


```python
%matplotlib inline

from sklearn import datasets, cluster
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# load data
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target
# do the clustering
centers = kmeans(X_iris, k=3)
labels = [find_closest_centroid(p, centers) for p in X_iris]

#plot the clusters in color
fig = plt.figure(1, figsize=(8, 8))
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=8, azim=200)
plt.cla()
ax.scatter(X_iris[:, 3], X_iris[:, 0], X_iris[:, 2], c=labels)
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
plt.show()

# moon
plt.figure(figsize=(8,8))
X, y = datasets.make_moons(2000, noise=0.2)
centers = kmeans(X, k=3)
labels = [find_closest_centroid(p, centers) for p in X]
plt.scatter(X[:,0], X[:,1], s=40, c=labels, cmap=plt.cm.Spectral)
plt.show()

# blob
plt.figure(figsize=(18,8))
plt.subplot(1,2,1)
np.random.seed(0)
X, y = datasets.make_blobs(n_samples=2000, centers=3, n_features=20, random_state=0)
centers = kmeans(X, k=3)
labels = [find_closest_centroid(p, centers) for p in X]
fig = plt.figure(1, figsize=(8, 8))
plt.scatter(X[:,0], X[:,1], s=40, c=labels, cmap=plt.cm.Spectral)


plt.subplot(1,2,2)
np.random.seed(3)
X, y = datasets.make_blobs(n_samples=2000, centers=3, n_features=20, random_state=0)
centers = kmeans(X, k=3)
labels = [find_closest_centroid(p, centers) for p in X]
fig = plt.figure(1, figsize=(8, 8))
plt.scatter(X[:,0], X[:,1], s=40, c=labels, cmap=plt.cm.Spectral)

plt.show()
```

    /opt/conda/lib/python3.4/site-packages/matplotlib/collections.py:590: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if self._edgecolors == str('face'):



![png](output_23_1.png)



![png](output_23_2.png)



![png](output_23_3.png)


<div class="comment"><br>
We can see that in the first plot the clustering works quite well indeed it is divided in three groups where the elements inside are actually close to each other. This is not true always because in certain iterations some elementes that should belong to the blue group are instead associated to the green one. This is due to the randomisation of the centroids and the outcome of our algorithm it's strongly related to the set of initial centroids and data itself.
<br><br>
In the second one the elements are really close to each other but still the algorithm is able to evaluate 3 well separated groups
<br><br>
For the last dataset the outcome it's strongly correleted to the initial centroids. Here, if we try to plot data we can see 3 different groups but if we try to cluster this set, the outcome most of iterations is really bad. As you can see in the leften plot of the last two.
</div>

That's enough about K-means for now. In the next section, we will apply MMLIB's K-means on Spark to deal with a large data in the real usecase.


# 2. Usecase: Network Intrusion 

Some attacks attempt to flood a computer with network traffic. In some other cases, attacks attempt to exploit flaws in networking software in order to gain unauthorized access to a computer. Detecting an exploit in an incredibly large haystack of network requests is not easy.

Some exploit behaviors follow known patterns such as scanning every port in a short of time, sending a burst of request to a port... However, the biggest threat may be the one that has never been detected and classified yet. Part of detecting potential network intrusions is detecting anomalies. These are connections that aren't known to be attacks, but, do not resemble connections that have been observed in the past.

In this notebook, K-means is used to detect anomalous network connections based on statistics about each of them.


## 2.1. Data
The data comes from [KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html). The dataset is about 708MB and contains about 4.9M connections. For each connection, the data set contains information like the number of bytes sent, login attempts, TCP errors, and so on. Each connection is one line of CSV-formatted data, containing 38 features: back, buffer_overflow, ftp_write, guess_passwd, imap, ipsweep, land, loadmodule, multihop, neptune, nmap, normal, perl, phf, pod, portsweep, rootkit, satan, smurf, spy, teardrop, warezclient, warezmaster. For more details about each features, please follow this [link](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html).

Many features take on the value 0 or 1, indicating the presence or absence of a behavior such as `su_attempted` in the 15th column. Some features are counts, like `num_file_creations` in the 17th columns. Some others are the number of sent and received bytes.

## 2.2. Clustering without using categorical features

First, we need to import some packages that are used in this notebook.


```python
import os
import sys
import re
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql import Row
from pyspark.sql.functions import *
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyspark.sql.functions as func
import matplotlib.patches as mpatches
from pyspark.mllib.clustering import KMeans, KMeansModel

input_path = "/datasets/k-means/kddcup.data"
raw_data = sc.textFile(input_path, 12)
```

### 2.2.1. Loading data

There are two types of features: numerical features and categorical features.
Currently, to get familiar with the data and the problem, we only use numerical features. In our data, we also have pre-defined groups for each connection, which we can use later as our "ground truth" for verifying our results.

**Note 1**: we don't use the labels in the training phase !!!

**Note 2**: in general, since clustering is un-supervised, you don't have access to ground truth. For this reason, several metrics to judge the quality of clustering have been devised. For a short overview of such metrics, follow this [link](https://en.wikipedia.org/wiki/Cluster_analysis#Internal_evaluation). Note that computing such metrics, that is trying to assess the quality of your clustering results, is as computationally intensive as computing the clustering itself!

![](https://farm2.staticflickr.com/1604/24934700445_833f0a5649_t.jpg)

<div class='anchor' ></div>

#### Question 2
Write function `parseLine` to construct a tuple of `(label, vector)` for each connection, extract the data that contains only the data points (without label), then print the number of connections.

Where,

* `label` is the pre-defined label of each connection
* `vector` is a numpy array that contains values of all features, but the label and the categorial features at index `1,2,3` of each connection. Each `vector` is a data point.


```python
def parseLine(line):
    cols = line.split(',')
    # label is the last column
    label = cols[-1]
    
    # vector is every column, except the label
    vector = cols[:-1]
    
    # delete values of columns that have index 1->3 (categorical features)
    vector = [vector[0]]+vector[4:]
    
    # convert each value from string to float
    vector = np.array(vector).astype(float)
    
    return (label, vector)

labelsAndData = raw_data.map(parseLine)

# we only need the data, not the label
data = labelsAndData.map(lambda x: x[1]).cache()


# number of connections
n = data.count()
```


```python
data.take(1)
```




    [array([  0.00000000e+00,   2.15000000e+02,   4.50760000e+04,
              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   1.00000000e+00,   1.00000000e+00,
              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   0.00000000e+00])]



![](https://farm2.staticflickr.com/1604/24934700445_833f0a5649_t.jpg)

<div class='anchor' ></div>

#### Question 3
Using K-means algorithm of MLLIB, cluster the connections into two groups then plot the result. Why two groups? In this case, we are just warming up, we're testing things around, so "two groups" has no particular meaning.

You can use the following parameters:

* `maxIterations=10`
* `runs=10`
* `initializationMode="random"`

Discuss the result from your figure.


```python
clusters = KMeans.train(data, 2, maxIterations=10,
        runs=10, initializationMode="random")
```


```python
import random
# do the clustering
centers = clusters.centers
SAMPLE_SIZE = 0.3
RANDOM_SEED = 42

data2 = np.array(data.sample(False, SAMPLE_SIZE, random.seed(RANDOM_SEED)).collect())
labels = data.map(lambda p: clusters.predict(p)).sample(False, SAMPLE_SIZE,random.seed(RANDOM_SEED)).collect()

num = len(data2[:, 10])
```


```python
#plot the clusters in color
fig = plt.figure(1, figsize=(8, 8))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=8, azim=200)
plt.cla()
ax.scatter(data2[:, 0], data2[:, 1], data2[:, 7], c=labels)

ax.set_xlabel('Duration')
ax.set_ylabel('Source bytes')
ax.set_zlabel('Login failed')

plt.show()
```

    /opt/conda/lib/python3.4/site-packages/matplotlib/collections.py:590: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if self._edgecolors == str('face'):



![png](output_39_1.png)


<div class="answer">

This is the 3D representation of the clustering we decided to put on the axis some feature that we think are relevant for identify an network intrusion.

</div>


```python
plt.figure(figsize=(18,10))

plt.subplot(2, 2, 1)
plt.hist(data2[:, 0], 20)
plt.yscale('log', nonposy='clip')
plt.title("Duration feature")
plt.xlabel("Duration")
plt.ylabel("Num of connections")

plt.subplot(2, 2, 2)
plt.hist(data2[:, 7], range(6))
plt.yscale('log', nonposy='clip')
plt.title("Login failed feature")
plt.xlabel("Login request")
plt.ylabel("Num of connections")

fig = plt.subplot(2, 2, 3)
plt.hist(data2[:, 1], 100)
plt.yscale('log', nonposy='clip')
fig.set_xscale('log', basex=2)
plt.title("Outgoing bytes")
plt.xlabel("Num of bytes")
plt.ylabel("Num of connections")

fig = plt.subplot(2, 2, 4)
plt.hist(data2[:, 2], 100)
plt.yscale('log', nonposy='clip')
fig.set_xscale('log', basex=2)
plt.title("Incoming bytes")
plt.xlabel("Num of bytes")
plt.ylabel("Num of connections")

num_root = len(list(filter(lambda x: x==1, data2[:, 10])))
print("Percentage of login with root privileges: ", num_root/num*100)
num_hot = len(list(filter(lambda x: x==1, data2[:, 17])))
print("Percentage of hot login: ", num_hot/num*100)
num_guest = len(list(filter(lambda x: x==1, data2[:, 18])))
print("Percentage of guest login: ", num_guest/num*100)
num_out_data = len(list(filter(lambda x: x>0, data2[:, 1])))
print("Percentage of connection width outgoing byte greater than 0: ", num_out_data/num*100)

plt.show()
```

    Percentage of login with root privileges:  0.006467411394761942
    Percentage of hot login:  6.807801468170464e-05
    Percentage of guest login:  0.08577829849894786
    Percentage of connection width outgoing byte greater than 0:  76.49313807651016



![png](output_41_1.png)


<div class="answer">
<br>
Here we wanted to understand data distribution of some features that we considered relevant for our problem.
<br><br>
We didn't plot the firsts features because even if we think that they are very useful for the intrusion detection they are boolean feature:
<br>
- login with root privileges: the value is really low as expected, most of connection should be indeed with user privileges; <br>
- hot login: user in the hot list;<br>
- guest login: logins of not registered user, so possible attackers; <br>

Then we plotted the duration of the connections and most of them lasts less then one hour but we can see that a non negligible number of connections last 11 hours. <br>
Form login failure plot we could say that there was no attempt of brute force attack on some user password because the maximum value of failure is 5.<br>
For the incoming bytes most of connection are in the order of hundreds of kB instead the outgoing connection are in the order of MB.
<br>
<br>
After this preliminary analysis we know how the data are distribuited and we can use these informtion to notice future anomalies.
</div>

### 2.2.3. Evaluating model

![](https://farm2.staticflickr.com/1604/24934700445_833f0a5649_t.jpg)

<div class='anchor' ></div>

#### Question 4

One of the simplest method to evaluate our result is calculate the Within Set Sum of Squared Errors (WSSSE), or simply, 'Sum of Squared Errors'. An error of a data point is defined as it's distance to the closest cluster center.



```python
from operator import add

# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(clusters, point):
    closest_center = clusters.centers[clusters.predict(point)]
    return np.power(euclidean_distance(closest_center, point), 2)
```


```python
WSSSE = data.map(lambda x: error(clusters, x)).reduce(add)

print("Within Set Sum of Squared Error = " + str(WSSSE))
```

    Within Set Sum of Squared Error = 3.05254895755e+18


![](https://farm2.staticflickr.com/1604/24934700445_833f0a5649_t.jpg)

<div class='anchor' ></div>

#### Question 5
This is a good opportunity to use the given labels to get an intuitive sense of what went into these two clusters, by counting the labels within each cluster. Complete the following code that uses the model to assign each data point to a cluster, and counts occurrences of cluster and label pairs.
What do you think about the result ?


```python
clusterLabelCount = data.map(lambda p: (clusters.predict(p), 1)).reduceByKey(lambda a,b: a+b).collect()

print("Our clusters:")
for item in clusterLabelCount:
    print(item)
```

    Our clusters:
    (0, 5)
    (1, 4898426)



```python
clusterLabelCount = labelsAndData.map(lambda x: (x[0], 1)).reduceByKey(lambda a,b: a+b).collect()

print("Original clusters:")
for item in clusterLabelCount:
    print(item)
```

    Original clusters:
    ('warezclient.', 1020)
    ('teardrop.', 979)
    ('smurf.', 2807886)
    ('perl.', 3)
    ('land.', 21)
    ('imap.', 12)
    ('spy.', 2)
    ('pod.', 264)
    ('guess_passwd.', 53)
    ('phf.', 4)
    ('normal.', 972781)
    ('nmap.', 2316)
    ('rootkit.', 10)
    ('warezmaster.', 20)
    ('back.', 2203)
    ('ftp_write.', 8)
    ('satan.', 15892)
    ('loadmodule.', 9)
    ('neptune.', 1072017)
    ('multihop.', 7)
    ('ipsweep.', 12481)
    ('portsweep.', 10413)
    ('buffer_overflow.', 30)


<div class='comment'>
Here the result is really bad because all elements are concentrated in one group. We could not expect a good result trying to cluster the datasets in 2 groups because we have a lot of connecitons and features and 2 groups are not enough to describe them. But still the outcome it's particularly bad, probably those 5 value are outliers and so the algorithm converges to this result.<br>
We did the same using the given labels, here most of connections are concentrated in few groups that are the non-malicius connection.
</div>

### 2.2.4. Choosing K

How many clusters are appropriate for a dataset? In particular, for our own dataset, it's clear that there are 23 distinct behavior patterns in the data, so it seems that k could be at least 23, or likely, even more. In other cases, we even don't have any information about the number of patterns at all (remember, generally your data is not labelled!). Our task now is finding a good value of $k$. For doing that, we have to build and evaluate models with different values of $k$. A clustering could be considered good if each data point were near to its closest centroid. One of the ways to evaluate a model is calculating the Mean of Squared Errors of all data points.

![](https://farm2.staticflickr.com/1604/24934700445_833f0a5649_t.jpg)

<div class='anchor' ></div>

#### Question 6
Complete the function below to calculate the MSE of each model that is corresponding to each value of $k$.
Plot the results. From the obtained result, what is the best value for $k$ ? Why ?


```python
# k: the number of clusters
def clusteringScore(data, k):
    clusters = KMeans.train(data, k, maxIterations=10,
        runs=10, initializationMode="random")
    # calculate mean square error
    return data.map(lambda x: error(clusters, x)).reduce(add)/n
```


```python
scores = []
k_range = [60,70,80,90,100,110]
for k in k_range:
    score = clusteringScore(data, k)
    scores.append(score)
    print("With k=", k, "the score is:", score)

```

    With k= 60 the score is: 197208222946.0
    With k= 70 the score is: 197206582475.0
    With k= 80 the score is: 197206890588.0
    With k= 90 the score is: 197199590212.0
    With k= 100 the score is: 197197373536.0
    With k= 110 the score is: 197195446177.0



```python
# This block is used to not run again the previous one that is very slow
scores = [197208222946, 197206582475, 197206890588, 197199590212, 197197373536, 197195446177]
k_range = [60,70,80,90,100,110]
k_range = k_range[:len(scores)]
```


```python
# plot results
plt.plot(k_range, scores)
plt.title("MSE trend changing K")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Performance (MSE)")
plt.show()
```


![png](output_59_0.png)


<div class="answer">
As expected the error decreses as we increse the number of cluster but after a certain value the slope would decrese.
<br>
In the case of K=80 che curve is slightly higher copared with k=70 that's because the algorithm in that iteration converged in an unfortunate situation probably due to the initial centroids.
</div>

### 2.2.5 Normalizing features

K-means clustering treats equally all dimensions/directions of the space and therefore tends to produce more or less spherical (rather than elongated) clusters. In this situation, leaving variances uneven is equivalent to putting more weight on variables with smaller variance, so clusters will tend to be separated along variables with greater variance.

In our notebook, since Euclidean distance is used, the clusters will be influenced strongly by the magnitudes of the variables, especially by outliers. Normalizing will remove this bias. 

Each feature can be normalized by converting it to a standard score. This means subtracting the mean of the feature’s values from each value, and dividing by the standard deviation

$normalize_i=\frac{feature_i - \mu_i}{\sigma_i}$

Where,

* $normalize_i$ is the normalized value of feature $i$
* $\mu_i$ is the mean of feature $i$
* $\sigma_i$ is the standard deviation of feature $i$


![](https://farm2.staticflickr.com/1604/24934700445_833f0a5649_t.jpg)

<div class='anchor' ></div>
#### Question 7

Complete the code below to normalize the data. Print the first 5 lines of the new data.

HINT: If $\sigma_i = 0$ then $normalize_i=feature_i - \mu_i$


```python
from operator import add

def normalizeData(data):
    # number of connections
    n = data.count()

    # calculate the sum of each feature
    #sums = data.flatMap(lambda x: [(i, x[i]) for i in range(len(x))]).reduceByKey(add).sortByKey()
    #sums = data.sum()

    # calculate means
    #means = sums.map(lambda x: x[1]/n).collect()
    means = data.mean()

    # calculate the sum square of each feature
    #sumSquares = data.flatMap(lambda x: [(i, np.power(x[i]-means[i], 2)) for i in range(len(x))])\
    #            .reduceByKey(np.add)\
    #            .sortByKey()

    # calculate standard deviation of each feature
    #stdevs = sumSquares.map(lambda x: 1 if (x[1]==0) else x[1]/n).collect()
    stdevs = data.stdev()

    def normalize(point):
        return [( (point[i]-means[i]) if (stdevs[i]==0) else (point[i]-means[i])/stdevs[i] ) for i in range(len(point))]

    return data.map(normalize)
```


```python
normalizedData = normalizeData(data).cache()
normalizedData.take(5)

```




    [[-0.06683318542420881,
      -0.001720382282287892,
      0.068188435127319452,
      -0.002390846860871979,
      -0.015139173434022768,
      -0.0011034846181947316,
      -0.02652075995496225,
      -0.0043909155847363995,
      2.4427918743642953,
      -0.0020973278346870065,
      -0.0082577083974692041,
      -0.004546461386643566,
      -0.0032845891671851204,
      -0.0095723392157848041,
      -0.0085045784245865279,
      -0.028756112730865285,
      0.0,
      -0.00063897900545820054,
      -0.028911303443117074,
      -1.5754150744340452,
      -1.196243238093317,
      -0.46604261387255441,
      -0.46575557391312739,
      -0.24828577490849504,
      -0.24813035170523645,
      0.53973309269996084,
      -0.2560565202907733,
      -0.20105929643802661,
      -3.6391392588276594,
      -1.7865104366036269,
      -1.8330227339690821,
      -0.28293900018696772,
      -1.2579366383587069,
      -0.1566684879542736,
      -0.46640478356659043,
      -0.46545364052902638,
      -0.25083182898698642,
      -0.2496319657767273],
     [-0.06683318542420881,
      -0.0017766795586854696,
      0.0053245145203947488,
      -0.002390846860871979,
      -0.015139173434022768,
      -0.0011034846181947316,
      -0.02652075995496225,
      -0.0043909155847363995,
      2.4427918743642953,
      -0.0020973278346870065,
      -0.0082577083974692041,
      -0.004546461386643566,
      -0.0032845891671851204,
      -0.0095723392157848041,
      -0.0085045784245865279,
      -0.028756112730865285,
      0.0,
      -0.00063897900545820054,
      -0.028911303443117074,
      -1.5706978877225637,
      -1.1921780765633343,
      -0.46604261387255441,
      -0.46575557391312739,
      -0.24828577490849504,
      -0.24813035170523645,
      0.53973309269996084,
      -0.2560565202907733,
      -0.20105929643802661,
      -3.6235193672194184,
      -1.7770687033066572,
      0.59896684321046056,
      -0.28293900018696772,
      0.82111873926321388,
      -0.1566684879542736,
      -0.46640478356659043,
      -0.46545364052902638,
      -0.25083182898698642,
      -0.2496319657767273],
     [-0.06683318542420881,
      -0.0016980758142813046,
      0.00020833276043491091,
      -0.002390846860871979,
      -0.015139173434022768,
      -0.0011034846181947316,
      -0.02652075995496225,
      -0.0043909155847363995,
      2.4427918743642953,
      -0.0020973278346870065,
      -0.0082577083974692041,
      -0.004546461386643566,
      -0.0032845891671851204,
      -0.0095723392157848041,
      -0.0085045784245865279,
      -0.028756112730865285,
      0.0,
      -0.00063897900545820054,
      -0.028911303443117074,
      -1.5754150744340452,
      -1.196243238093317,
      -0.46604261387255441,
      -0.46575557391312739,
      -0.24828577490849504,
      -0.24813035170523645,
      0.53973309269996084,
      -0.2560565202907733,
      -0.20105929643802661,
      -3.6078994756111777,
      -1.7676269700096876,
      0.59896684321046056,
      -0.28293900018696772,
      -0.21840894954774653,
      -0.1566684879542736,
      -0.46640478356659043,
      -0.46545364052902638,
      -0.25083182898698642,
      -0.2496319657767273],
     [-0.06683318542420881,
      -0.0017012624525679599,
      0.0014548206801342169,
      -0.002390846860871979,
      -0.015139173434022768,
      -0.0011034846181947316,
      -0.02652075995496225,
      -0.0043909155847363995,
      2.4427918743642953,
      -0.0020973278346870065,
      -0.0082577083974692041,
      -0.004546461386643566,
      -0.0032845891671851204,
      -0.0095723392157848041,
      -0.0085045784245865279,
      -0.028756112730865285,
      0.0,
      -0.00063897900545820054,
      -0.028911303443117074,
      -1.5706978877225637,
      -1.1921780765633343,
      -0.46604261387255441,
      -0.46575557391312739,
      -0.24828577490849504,
      -0.24813035170523645,
      0.53973309269996084,
      -0.2560565202907733,
      -0.20105929643802661,
      -3.5922795840029367,
      -1.758185236712718,
      0.59896684321046056,
      -0.28293900018696772,
      -0.57184836374347303,
      -0.1566684879542736,
      -0.46640478356659043,
      -0.46545364052902638,
      -0.25083182898698642,
      -0.2496319657767273],
     [-0.06683318542420881,
      -0.0016948891759946494,
      -0.00094203295650151318,
      -0.002390846860871979,
      -0.015139173434022768,
      -0.0011034846181947316,
      -0.02652075995496225,
      -0.0043909155847363995,
      2.4427918743642953,
      -0.0020973278346870065,
      -0.0082577083974692041,
      -0.004546461386643566,
      -0.0032845891671851204,
      -0.0095723392157848041,
      -0.0085045784245865279,
      -0.028756112730865285,
      0.0,
      -0.00063897900545820054,
      -0.028911303443117074,
      -1.565980701011082,
      -1.1881129150333516,
      -0.46604261387255441,
      -0.46575557391312739,
      -0.24828577490849504,
      -0.24813035170523645,
      0.53973309269996084,
      -0.2560565202907733,
      -0.20105929643802661,
      -3.5766596923946956,
      -1.7487435034157486,
      0.59896684321046056,
      -0.28293900018696772,
      -0.73817279395322677,
      -0.1566684879542736,
      -0.46640478356659043,
      -0.46545364052902638,
      -0.25083182898698642,
      -0.2496319657767273]]



![](https://farm2.staticflickr.com/1604/24934700445_833f0a5649_t.jpg)

<div class='anchor' ></div>

#### Question 8
Using the new data, build different models with different values of $k \in [60,70,80,90,100,110]$. Evaluate the results by plotting them and choose the best value of $k$.


```python
scores_norm = []
k_range_norm = [60,70,80,90,100,110]
for k in k_range_norm:
    score_norm = clusteringScore(normalizedData, k)
    scores_norm.append(score_norm)
    print("With k=", k, "the score is:", score_norm)

```


```python
# This block is used to not run again the previous one that is very slow
scores_norm = [12.45420748, 12.40548954, 11.49139048, 10.64299971, 10.54808320, 10.30299247]
k_range_norm = [60,70,80,90,100,110]
k_range_norm = k_range_norm[:len(scores_norm)]
```


```python
# Plot resutls
plt.plot(k_range_norm,scores_norm)
plt.title("MSE trend changing K of normalized data")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Performance (MSE)")
plt.show()
```


![png](output_70_0.png)


<div class='answer'>
As in the previous case the trend of the curve it's the same but we can see that the range of values it's completly different due to the normalization of the data.
</div>

![](https://farm2.staticflickr.com/1604/24934700445_833f0a5649_t.jpg)

<div class='anchor' ></div>

#### Question 9
Plot the clustering result to see the difference between before and after normalizing features. Discuss about the difference and explain why and if normalization was useful.


```python
# plot results
plt.plot(k_range, scores, 'r', k_range_norm, scores_norm, 'b')
plt.title("MSE trend changing K")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Performance (MSE)")

plt.show()
```


![png](output_74_0.png)



```python
factor = scores[0]/scores_norm[0]
scores_fac = [x/factor for x in scores]

# plot results
plt.plot(k_range, scores_fac, 'r', k_range_norm, scores_norm, 'b')
plt.title("MSE trend changing K")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Performance (MSE)")

plt.show()
```


![png](output_75_0.png)


<div class="comment">
I we plot the scores as they are we cannot see anything because the values are too far due to the normalization.
So we multiplied all of them to a factor which is the division between the firsts two, but this is not enought to compare the two plots because the standard deviation of those scores is much different so we should apply a more complicated function in order to have a more rappresentative comparison.


</div>

## 2.3. Clustering using categorical features

### 2.3.1 Loading data

In the previous section, we ignored the categorical features of our data: this is not a good idea, since these categorical features can be important in providing useful information for clustering.
The problem is that K-means (or at least, the one we have developed and the one we use from MLLib) only work with data points in a metric space. Informally, this means that operations such as addition, substraction and computing the mean of data points are trivial and well defined. For a more formal definition of what a metric space is, follow this [link](https://en.wikipedia.org/wiki/Metric_space#Definition).

What we will do next is to transform each categorical features into one or more numerical features. This approach is very widespread: imagine for example you wanted to use K-means to cluster text data. Then, the idea is to transform text data in $d$-dimensional vectors, and a nice way to do it is to use [word2vec](http://deeplearning4j.org/word2vec). If you're interested, follow this link to a nice [blog post](http://bigdatasciencebootcamp.com/posts/Part_3/clustering_news.html) on the problem.

There are two approaches:

* **Approach 1**: mapping **one** categorial feature to **one** numerical feature. The values in each categorical feature are encoded into unique numbers of the new numerical feature. For example, ['VERY HOT','HOT', 'COOL', 'COLD', 'VERY COLD'] will be encoded into [0,1,2,3,4,5]. However, by using this method, we implicit assume that the value of 'VERY HOT' is smaller than 'HOT'... This is not generally true.

* **Approach 2**: mapping mapping **one** categorial feature to **multiple** numerical features. Basically, a single variable with $n$ observations and $d$ distinct values, to $d$ binary variables with $n$ observations each. Each observation indicating the presence (1) or absence (0) of the $d^{th}$ binary variable. For example, ['house', 'car', 'tooth', 'car'] becomes 
```
[
[1,0,0,0],
[0,1,0,0],
[0,0,1,0],
[0,0,0,1],
]
```

We call the second approach "one-hot encoding". By using this approach, we keep the same role for all values of categorical features.



![](https://farm2.staticflickr.com/1604/24934700445_833f0a5649_t.jpg)

<div class='anchor' ></div>

#### Question 10
Calculate the number of distinct categorical features value (at index `1,2,3`). Then construct a new input data using one-hot encoding for these categorical features (don't throw away numerical features!).


```python
# Different values of each feature
def get_categorical_features(line):
    cols = line.split(',')
    return cols[1:4]
dictionary = raw_data.flatMap(get_categorical_features).distinct().collect()
bDictionary = sc.broadcast(dictionary)
print(bDictionary.value)
```

    ['login', 'rje', 'OTH', 'urh_i', 'hostnames', 'ftp_data', 'RSTOS0', 'bgp', 'SH', 'ctf', 'S1', 'netbios_ns', 'SF', 'IRC', 'S3', 'S0', 'uucp', 'nntp', 'daytime', 'domain', 'nnsp', 'remote_job', 'exec', 'pm_dump', 'RSTR', 'netbios_dgm', 'tftp_u', 'pop_3', 'Z39_50', 'auth', 'http', 'whois', 'finger', 'echo', 'name', 'iso_tsap', 'efs', 'shell', 'icmp', 'http_2784', 'udp', 'time', 'red_i', 'supdup', 'tcp', 'urp_i', 'uucp_path', 'netstat', 'sunrpc', 'ssh', 'link', 'smtp', 'courier', 'gopher', 'mtp', 'RSTO', 'ecr_i', 'other', 'pop_2', 'ldap', 'sql_net', 'S2', 'klogin', 'X11', 'ftp', 'aol', 'domain_u', 'imap4', 'csnet_ns', 'eco_i', 'private', 'vmnet', 'REJ', 'telnet', 'tim_i', 'netbios_ssn', 'http_8001', 'printer', 'ntp_u', 'systat', 'http_443', 'discard', 'kshell', 'harvest']



```python
def parseLineWithHotEncoding(line):
    cols = line.split(',')
    # label is the last column
    label = cols[-1]
    
    vector = cols[0:-1]
    
    # Prepare vector of categorical features
    cat_vec = [0] * len(bDictionary.value)
    
    # the binary features that are encoded from the first categorial feature
    cat_vec[bDictionary.value.index(cols[1])] = 1
        
    # the binary features that are encoded from the second categorial feature
    cat_vec[bDictionary.value.index(cols[2])] = 1
    
    # the binary features that are encoded from the third categorial feature
    cat_vec[bDictionary.value.index(cols[3])] = 1
    
    # construct the new vector
    vector = ([vector[0]] + cat_vec + vector[4:])
    
    # convert each value from string to float
    vector = np.array(vector).astype(float)
    
    return (label, vector)

labelsAndData2 = raw_data.map(parseLineWithHotEncoding)

# we only need the data, not the label
data2 = labelsAndData2.values().cache()

normalizedData2 = normalizeData(data2).cache()
```

### 2.3.2. Building models

![](https://farm2.staticflickr.com/1604/24934700445_833f0a5649_t.jpg)

<div class='anchor' ></div>
#### Question 11
Using the new data, cluster the connections with different values of $k \in [80,90,100,110,120,130,140,150,160]$.
Evaluate the results and choose the best value of $k$ as previous questions.


```python
scores2 = []
k_range2 = [80,90,100,110,120,130,140,150,160]
for k in k_range2:
    score2 = clusteringScore(normalizedData2, k)
    scores2.append(score2)
    print("With k=", k, "the score is:", score2)

```


```python
# This block is used to not run again the previous one that is very slow
scores2 = [65.42724061, 64.43378482, 59.67113683]
k_range2 = [80,90,100,110,120,130,140,150,160]
k_range2 = k_range2[:len(scores2)]
```


```python
plt.plot(scores2)
plt.title("Incoming bytes")
plt.xlabel("N_th label")
plt.ylabel("Num item in the label")
plt.show()
```


![png](output_88_0.png)


<div class="answer">


As expected we have a better score with an higher value of k. We didn't run all the value of k due to lack of time.

</div>

## 2.4. Anomaly detection
When we have a new connection data (e.g., one that we never saw before), we simply find the closest cluster for it, and use this information as a proxy to indicate whether the data point is anomalous or not. A simple approach to decide wheter there is an anomaly or not, amounts to measuring the new data point’s distance to its nearest centroid. If this distance exceeds some thresholds, it is anomalous.

![](https://farm2.staticflickr.com/1604/24934700445_833f0a5649_t.jpg)

<div class='anchor' ></div>

#### Question 12
Build your model with the best value of $k$ in your opinion. Then, detect the anomalous connections in our data. Plot and discuss your result.

HINT: The threshold has strong impact on the result. Be careful when choosing it! A simple way to choose the threshold's value is picking up a distance of a data point from among known data. For example, the 100th-farthest data point distance can be an option.


```python
best_k_value = 110
clusters2 = KMeans.train(normalizedData2, best_k_value, maxIterations=10,
        runs=10, initializationMode="random")
```


```python
clusters2LabelCount = normalizedData2.map(lambda p: (clusters2.predict(p), 1))\
        .reduceByKey(lambda a,b: a+b).sortByKey().collect()

print("Data distribution among centroids:")
print(clusters2LabelCount)
```

    Data distribution among centroids:
    [(0, 970), (1, 5161), (2, 615), (3, 10696), (4, 11476), (6, 131795), (7, 50), (8, 223), (9, 268), (12, 1041), (13, 118874), (14, 41131), (15, 8406), (16, 50115), (17, 3742), (18, 47), (19, 123), (20, 4237), (21, 333634), (23, 3833), (24, 193893), (25, 2090), (26, 4087), (27, 192815), (29, 1929182), (30, 15561), (31, 5378), (35, 132065), (36, 5732), (37, 21867), (43, 3394), (47, 4488), (48, 4549), (49, 455605), (50, 4221), (51, 976), (52, 61083), (53, 7551), (54, 442), (55, 28984), (57, 62688), (59, 9291), (60, 3970), (61, 4912), (62, 45122), (64, 77023), (67, 1040), (68, 364383), (71, 2092), (72, 1071), (73, 6671), (74, 4948), (76, 3190), (77, 64522), (78, 13063), (79, 11330), (80, 4013), (84, 36179), (86, 1055), (90, 336), (92, 40345), (93, 5348), (95, 3189), (96, 3348), (97, 2885), (98, 1093), (100, 16338), (103, 900), (104, 606), (105, 3416), (106, 2), (108, 265610), (109, 38052)]



```python
centroids2 = clusters2.centers
distances2 = normalizedData2.map(lambda p: (euclidean_distance(centroids2[clusters2.predict(p)], p), 1))\
        .reduceByKey(lambda a,b: a+b).sortByKey(ascending=False)

print("Greater distances and their frequency:")
print(distances2.take(10))
```

    Greater distances and their frequency:
    [(2676.9270880404379, 1), (2213.1077392516186, 1), (2030.4764156905771, 1), (1939.9483792871406, 1), (1566.7316511398803, 1), (1564.8291331672858, 1), (1564.8291216796324, 1), (1564.8101559595673, 1), (1564.8100283611066, 1), (1564.8100090765511, 1)]



```python
maxDistance = distances2.take(1)[0][0] #2676.9270880404379
histogram_bins = 1000
bins_width = maxDistance/histogram_bins

plot_data2 = distances2.map(lambda x: (int(x[0]/bins_width), x[1]))\
        .reduceByKey(lambda a,b: a+b).sortByKey()
```


```python
plot_data2_vector = plot_data2.collect()
print(plot_data2_vector)
```

    [(0, 4566769), (1, 150309), (2, 85257), (3, 20810), (4, 7508), (5, 8636), (6, 2482), (7, 2448), (8, 3053), (9, 1450), (10, 1852), (11, 658), (12, 358), (13, 306), (14, 1797), (15, 383), (16, 54), (17, 81), (18, 1950), (19, 66), (20, 6732), (21, 7595), (22, 11060), (23, 7621), (24, 5321), (25, 1205), (26, 242), (27, 28), (28, 19), (29, 18), (30, 3), (31, 168), (32, 293), (33, 366), (34, 164), (35, 24), (36, 13), (37, 11), (38, 3), (39, 4), (40, 9), (41, 2), (42, 351), (43, 33), (44, 4), (45, 1), (46, 43), (47, 81), (48, 1), (49, 13), (50, 6), (51, 2), (52, 9), (53, 5), (54, 1), (55, 2), (56, 5), (58, 5), (59, 6), (62, 1), (63, 70), (64, 90), (65, 8), (66, 3), (67, 151), (68, 4), (69, 1), (70, 131), (71, 1), (73, 7), (74, 7), (75, 2), (76, 6), (77, 4), (78, 2), (79, 3), (80, 6), (81, 7), (82, 8), (85, 2), (86, 5), (87, 2), (88, 6), (89, 4), (90, 3), (91, 1), (92, 1), (93, 1), (94, 4), (95, 5), (96, 1), (97, 1), (98, 1), (99, 1), (100, 2), (101, 1), (103, 4), (104, 1), (106, 50), (107, 6), (108, 2), (109, 6), (110, 1), (112, 3), (113, 1), (114, 1), (116, 1), (118, 2), (119, 1), (121, 1), (122, 2), (123, 1), (125, 1), (131, 1), (133, 1), (136, 2), (137, 1), (138, 1), (141, 2), (146, 2), (147, 2), (148, 1), (155, 28), (156, 1), (169, 1), (182, 1), (184, 1), (197, 3), (232, 2), (234, 1), (235, 1), (238, 12), (248, 1), (256, 1), (271, 1), (275, 9), (369, 5), (449, 1), (477, 3), (533, 1), (584, 7), (585, 1), (724, 1), (758, 1), (826, 1), (1000, 1)]



```python
plt.figure(figsize=(18,5))

x = plot_data2.map(lambda x: int(x[0]*bins_width)).collect()
y = plot_data2.map(lambda x: x[1]).collect()

plt.yscale('log', nonposy='clip')
plt.xscale('log', nonposy='clip')
plt.title("Distribution of distance between points and centroids")
plt.xlabel("Distance (log)")
plt.ylabel("Num of point with that distance (log)")
barlist = plt.bar(x, y)
for i in range(0, len(barlist)):
    barlist[i].set_color('b')

plt.show()
```


![png](output_98_0.png)


<div class="answer">

For the anomaly detection analisys we decided to plot a bar graph in order to understand how the distance between points and its centroid is distributed.
As aspected most of the point are very close to the centroid and only some outliers are very far from it.


</div>


```python
# threshold evaluation
i = len(plot_data2_vector)-1
c = 0
while(c<100 and i>=0):
    c += plot_data2_vector[i][1]
    i -=1
bin_threshold = plot_data2_vector[i][0]
print(bin_threshold)
```

    122



```python
# retrive the farhest points and the relative features difference
THRESHOLD = bins_width*bin_threshold

farthest_points = normalizedData2.filter(lambda p: euclidean_distance(centroids2[clusters2.predict(p)], p) > THRESHOLD)
def calculate_feature_difference(point):
    nearest_centroid = clusters2.centers[clusters2.predict(point)]
    return [(i, np.abs(nearest_centroid[i] - point[i])) for i in range(len(point))]

farthest_features = farthest_points.flatMap(calculate_feature_difference)\
        .reduceByKey(lambda a,b: a+b).sortBy(lambda x: x[1], ascending=False)

```

<div class="note">
Here we are trying to understand which features influece more the distance from the centroids.
This feature can be more relevant during the clustering.
</div>


```python
feature_names = ['duration'] + dictionary + ['src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate']
bFeature_names = sc.broadcast(feature_names)
```


```python
x = farthest_features.map(lambda x: bFeature_names.value[x[0]]).collect()
y = farthest_features.map(lambda x: x[1]).collect()

plt.figure(figsize=(18,5))
plt.title("Feature importance for outlier points")
plt.xlabel("Features")
plt.ylabel("Distance of that feature form centroid")
barlist = plt.bar(range(len(y)), y)
for i in range(0, len(barlist)):
    barlist[i].set_color('b')
plt.show()
```


![png](output_104_0.png)



```python
x2 = x[:11]
y2 = y[:11]

plt.figure(figsize=(15,5))
plt.title("Feature importance for outlier points")
plt.xlabel("Features")
plt.ylabel("Distance of that feature form centroid")
barlist = plt.bar(range(1,len(y2)+1), y2, align="center")
for i in range(0, len(barlist)):
    barlist[i].set_color('b')
plt.xticks(range(1,len(y2)+1), x2)
plt.show()

```


![png](output_105_0.png)


<div class="answer">

Based on the threadshold that we used (100 farthest points) we obtained that the features that have more influence on the distance are the following<br>
(land, tim_i, red_i, pm_dump, num_failed_logins, num_root, num_compromised, src_bytes, tftp_u, urgent, dst_bytes)
<br>
- land: 1 if connection is from/to the same host/port; 0 otherwise.
<br>
- tim_i: one of the possible value of service (categorical feature)
<br>
- red_i: one of the possible value of service (categorical feature)

</div>

![](https://farm2.staticflickr.com/1604/24934700445_833f0a5649_t.jpg)

<div class='anchor' ></div>

#### Question 13
Try other methods to find the best value for $k$ such as `silhouette`, `entropy`... In particular, with this data, you can take advantage of predefined labels to calculate the quality of model using entropy... However, we suggest you to try with `silhouette`. It's more general and can work with any dataset (with and without predefined labels).

Here are some additional information about the metrics we suggest to use:
- [Silhouette](https://en.wikipedia.org/wiki/Silhouette_(clustering))

- [Hack approach to Silhouette](http://scikitlearn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)

- [Entropy](http://scikit-learn.org/stable/modules/clustering.html) [Lookup for entropy]

Note: you are free to play with any relevant evaluation metric you think appropriate for your work!


```python
# Silhouette

def Silhouette(pair):
    point_cluster = clusters.predict(pair[0])
    
    groups = pair[1].groupBy(lambda x: clusters.predict(x))
    
    aAndC = groups.map(lambda x: x[1].map(lambda y: eucliderian_distance(pair[0], y)).reduceByKey(np.mean))
    a = aAndC.filter(lambda x: x[0]==point_cluster)[0][1]
    c = aAndC.filter(lambda x: x[0]!=point_cluster)[:][1]
    b = c.min()
    
    return 1-a/b if a<b else a/b-1

#s = data.map(lambda point: Silhouette(point)).collect()
s = data.cartesian(data).groupBy(lambda x: tuple(x[0])).map(lambda pair: Silhouette(pair))
ss = s.collect()

plt.hist(s, 100)
plt.title("Silhouette distridution")
plt.xlabel("S index")
plt.ylabel("Num of items")
plt.show()

```

<div class="answer">


We tried to implement Silhouette algorithm because compared with the previous one gives a better eximation of how good data is splitted amoung different groups. But it takes a time that is the square of the previous one because the distance it's evaluated between one point and all the others and this must be done for all the points.
<br><br>
For lack of time we didn't run it. 

</div>

![](https://farm2.staticflickr.com/1604/24934700445_833f0a5649_t.jpg)

<div class='anchor' ></div>

#### Question 14
Implement K-means on Spark so that It can work with large datasets in parallel. Test your algorithm with our dataset in this notebook. Compare our algorithm with the algorithm from MLLIB.

Let's clarify the meaning of this question: what we want is for students to design the K-means algorithm for the parallel programming model exposed by Spark. You are strongly invited to use the Python API (pyspark). So, at the end of the day, you will operate on RDDs, and implement a "map/reduce" algorithm that performs the two phases of the standard K-means algorithm, i.e. the assignment step and the update step.


```python

```

<div class="answer">


PUT YOUR ANSWER HERE !!!


</div>
