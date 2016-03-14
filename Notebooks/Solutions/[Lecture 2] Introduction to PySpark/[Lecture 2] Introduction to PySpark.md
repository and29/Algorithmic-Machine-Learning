# Goals of the Laboratory
In this introductory laboratory, we expect students to:

1. Acquire basic knowledge about Python and Matplotlib
2. Gain familiarity with Juypter Notebooks
3. Gain familiarity with the PySpark API and SparkSQL

To achieve such goals, we will go through the following steps:

1. In section 1, **IPython** and **Jupyter Notebooks** are introduced to help students understand the environment used to work on Data Science projects.

2. In section 2, we briefly overview **Python** and its syntax. In addition, we cover **Matplotlib**, a very powerful library to plot figures in Python, which you can use for your Data Science projects. Finally, we introduce **Pandas**, a python library that is very helpful when working on Data Science projects.

3. In section 3 we cover the **PySpark** and **SparkSQL** APIs

4. In section 4, we conclude the introductory laboratory with a simple use case.


# 1. Python, IPython and Jupyter Notebooks

**Python** is a high-level, dynamic, object-oriented programming language. It is a general purpose language, which means that many things are made easy. It's designed to be easy to program and easy to read.

**IPython** (Interactive Python) is orignally developed for Python. Now, it is a command shell for interactive computing in multiple programming languages. It offers rich media, shell syntax, tab completion, and history. IPython is based on an architecture that provides parallel and distributed computing. IPython enables parallel applications to be developed, executed, debugged and monitored interactively.

**Jupyter Notebooks** are a web-based interactive computational environment for creating IPython notebooks. An IPython notebook is a JSON document containing an ordered list of input/output cells which can contain code, text, mathematics, plots and rich media.  That makes data analysis easier to perform, understand and reproduce. All laboratories in this course are prepared as Notebooks. As you can see, in this Notebook, we can put text, images, hyperlinks, source code... The Notebooks can be converted to a number of open standard output formats (HTML, HTML presentation slides, LaTeX, PDF, ReStructuredText, Markdown, Python) through `File` -> `Download As` in the web interface. Beside, Jupyter manages the notebooks' versions through a `checkpoint` mechanism. You can create checkpoint anytime via `File -> Save and Checkpoint`.

Let's go throught the features of Jupyter Notebooks.

## 1.1. Tab completion

Tab completion is a convenient way to explore the structure of any object you're dealing with. Simply type object_name.<TAB> to view the suggestion for object's attributes. Besides Python objects and keywords, tab completion also works on file and directory names.


```python
s = "test function of tab completion"

# type s.<TAB> to see the suggestions
# For example, you can show your tests to work on a string. Try splitting a string into its constituent words!
```

## 1.2. System shell commands

To run any command in the system shell, simply prefix it with `!`. For example:


```python
# list all file and directories in the current folder
!ls
!pwd
!ls -l
```

    [Lecture 2] Introduction to PySpark.ipynb
    /home/group06/work
    total 100
    -rw-r--r-- 1 group06 users 98343 Mar  4 13:38 [Lecture 2] Introduction to PySpark.ipynb


## 1.3. Magic functions

IPython has a set of predefined `magic functions` that you can call with a command line style syntax. There are two types of magics, line-oriented and cell-oriented. 

**Line magics** are prefixed with the `%` character and work much like OS command-line calls: they get as an argument the rest of the line, *where arguments are passed without parentheses or quotes*. 

**Cell magics** are prefixed with a double `%%`, and they are functions that get as an argument not only the rest of the line, but also the lines below it in a separate argument.


```python
%timeit range(1000)
```

    The slowest run took 13.66 times longer than the fastest. This could mean that an intermediate result is being cached.
    1000000 loops, best of 3: 283 ns per loop



```python
%%timeit x = range(10000)
max(x)
```

    1000 loops, best of 3: 348 µs per loop


For more information, you can follow this [link](http://nbviewer.jupyter.org/github/ipython/ipython/blob/1.x/examples/notebooks/Cell%20Magics.ipynb)

## 1.4. Debugging

Whenever an exception occurs, the call stack is print out to help you to track down the true source of the problem. It is important to gain familiarity with the call stack, especially when using the PySpark API.


```python
for i in [4,3,2,0]:
    print(5/i)
```

    1.25
    1.6666666666666667
    2.5



    ---------------------------------------------------------------------------

    ZeroDivisionError                         Traceback (most recent call last)

    <ipython-input-15-8b3700012e23> in <module>()
          1 for i in [4,3,2,0]:
    ----> 2     print(5/i)
    

    ZeroDivisionError: division by zero


## 1.5. Additional features

Jupyter also supports viewing the status of the cluster and interact with the real shell environment.

To do that, you can click on the Logo Jupyter in the up-left coner of each notebook to go to the dashboard:

<img src="https://farm2.staticflickr.com/1488/24681339931_733acb3494_b.jpg" width="600px" />

You can easily find out how to use these features, so you're invited to play around!!

# 2. Python + Pandas + Matplotlib: A great environment for Data Science

This section aims to help the students have a basic understanding of the python programming language and its wonderful libraries. It also helps whose who are not familiar with `Pandas` or `Matplotlib` to have a first glance at basic use of such libraries. 

When working with a small dataset (one that can comfortably fit into a single machine), Pandas and Matplotlib, together with Python are valid alternatives to other popular tools such as R and Matlab. Using such libraries allows to inherit from the simple and clear Python syntax, achieve very good performance, enjoy a better memory management, better error handling, and good package management \[[1](http://ajminich.com/2013/06/22/9-reasons-to-switch-from-matlab-to-python/)\].


## 2.1. Python syntax

(This section is for students who did not program in Python before. If you're familiar with Python, please move to the next section: 1.2. Numpy)

When working with Python, the code seems to be simpler than (many) other languages. In this laboratory, we compare the Python syntax to that of Java - another very common language.

```java
// java syntax
int i = 10;
string s = "advanced machine learning";
System.out.println(i);
System.out.println(s);
// you must not forget the semicolon at the end of each sentence
```


```python
# python syntax
i = 10
s = "advanced machine learning"
print(i)
print(s)
# forget about the obligation of commas
```

    10
    advanced machine learning


### Indentation & If-else syntax
In python, we don't use `{` and `}` to make blocks of codes. Instead, we use indentation to do that. **The codes in the same block must have the same indentation**. For example, in java, we write:
```java
string language = "Python";

// the block is surrounded by { and }
// the condition is in ( and )
if (language == "Python") {
    int x = 1;
    x += 10;
       int y = 5; // a wrong indentation isn't problem
    y = x + y;
    System.out.println(x + y);
    
    // a statement is broken into two line
    x = y
        + y;
    
    // do some stuffs
}
else if (language == "Java") {
    // another block
}
else {
    // another block
}
```


```python
language = "Python"
if language == "Python":
    x = 10
    x += 10
    y = 5 # all statements in the same block must has the same indentation
    y = (
        x + y
    ) # a statement can be in multiple line with ( )
    print (x 
           + y)
    
    # statement can also be divided by using \ at the END of each line
    x = y \
        + y
    
    # do some other stuffs
elif language == "Java":
    # another block
    pass
else:
    # another block
    pass
```

    45


### Ternary conditional operator
In python, we often see ternary conditional operator when reading code of labraries. It is an operator to assign a value for a variable based on some condition. For example, in java, we write:

```java
int x = 10;
// if x > 10, assign y = 5, otherwise, y = 15
int y = (x > 10) ? 5 : 15;

int z;
if (x > 10)
    z = 5; // it's not necessary to have { } when the block has only one statement
else
    z = 15;
```

Of course, although we can easily write these lines of code in an `if else` block to get the same result, people prefer ternary conditioinal operator because of its simplicity.

In python, we write:


```python
x = 10
# a very natural way
y = 5 if x > 10 else 15
print(y)

# another way
y = x > 10 and 5 or 15
print(y)
```

    15
    15


### List & For loop
Another syntax that we should revisit is the `for loop`. In java, we can write:

```java
// init an array with 10 integer numbers
int[] array = new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
for (int i = 0; i < array.length; i++){
    // print the i-th element of array
    System.out.println(array[i]);
}
```

In Python, instead of using an index to help indicating an element, we can access the element directly:


```python
array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Python has no built-in array data structure
# instead, it uses "list" which is much more general 
# and can be used as a multidimensional array quite easily.
for element in array:
    print(element)
```

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10


As we can see, the code is very clean. If you need the index of each element, it's no problem:


```python
for (index, element) in enumerate(array):
    print(index, element)
```

    0 1
    1 2
    2 3
    3 4
    4 5
    5 6
    6 7
    7 8
    8 9
    9 10


Actually, Python has no built-in array data structure. It uses `list` which is much more general and can be used as a multidimensional array quite easily. Besides, the elements in a list are retrieved in a very concise way. For example, we create a 2d-array with 4 rows. Each row has 3 elements.


```python
# 2-dimentions array with 4 rows, 3 columns
twod_array = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
for index, row in enumerate(twod_array):
    print("row ", index, ":", row)

# print row 1 until row 3
print("row 1 until row 3: ", twod_array[1:3])

# all rows from row 2
print("all rows from row 2: ", twod_array[2:])

# all rows until row 2
print("all rows until row 2:", twod_array[:2])

# all rows from the beginning with step of 2. 
print("all rows from the beginning with step of 2:", twod_array[::2])
```

    row  0 : [1, 2, 3]
    row  1 : [4, 5, 6]
    row  2 : [7, 8, 9]
    row  3 : [10, 11, 12]
    row 1 until row 3:  [[4, 5, 6], [7, 8, 9]]
    all rows from row 2:  [[7, 8, 9], [10, 11, 12]]
    all rows until row 2: [[1, 2, 3], [4, 5, 6]]
    all rows from the beginning with step of 2: [[1, 2, 3], [7, 8, 9]]


### Dictionary
Another useful data structure in Python is `dictionary`. A dictionary stores (key, value) pairs. You can use it like this:


```python
d = {'key1': 'value1', 'key2': 'value2'}  # Create a new dictionary with some data
print(d['key1'])       # Get an entry from a dictionary; prints "value1"
print('key1' in d)     # Check if a dictionary has a given key; prints "True"
d['key3'] = 'value3'    # Set an entry in a dictionary
print(d['key3'])      # Prints "wet"
# print(d['key9'])  # KeyError: 'monkey' not a key of d
print(d.get('key9', 'custom_default_value'))  # Get an element with a default; prints "custom_default_value"
print(d.get('key3', 'custom_default_value'))    # Get an element with a default; prints "value3"
del d['key3']        # Remove an element from a dictionary
print(d.get('key3', 'custom_default_value')) # "fish" is no longer a key; prints "custom_default_value"

```

    value1
    True
    value3
    custom_default_value
    value3
    custom_default_value


### Functions
In Python, we can define a function by using keyword `def`.


```python
def square(x):
    return x*x

print(square(5))
```

    25


You can apply a function on each element of a list/array by using `lambda` function. For example, we want to square elements in a list:


```python
array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# apply function "square" on each element of "array"
print(list(map(lambda x: square(x), array)))

# or using a for loop
print([square(x) for x in array])

print("orignal array:", array)
```

    [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
    [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
    orignal array: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


These two above syntaxes are used very often. We can also put a function `B` inside a function `A` (that is, we can have nested functions). In that case, function `B` is only accessed inside function `A` (the scope that it's declared). For example:


```python
# select only the prime number in array
# and square them
def filterAndSquarePrime(arr):
    
    # a very simple function to check a number is prime or not
    def checkPrime(number):
        for i in range(2, int(number/2)):
            if number % i == 0:
                return False
        return True
    
    primeNumbers = filter(lambda x: checkPrime(x), arr)
    return map(lambda x: square(x), primeNumbers)

# we can not access checkPrime from here
# checkPrime(5)

result = filterAndSquarePrime(array)
list(result)
```




    [1, 4, 9, 16, 25, 49]



### Importing modules, functions
Modules in Python are packages of code. Putting code into modules helps increasing the reusability and maintainability.
The modules can be nested.
To import a module, we simple use syntax: `import <module_name>`. Once it is imported, we can use any functions, classes inside it.


```python
# import module 'math' to uses functions for calculating
import math

# print the square root of 16
print(math.sqrt(16))

# we can create alias when import a module
import numpy as np

print(np.sqrt(16))
```

Sometimes, you only need to import some functions inside a module to avoid loading the whole module into memory. To do that, we can use syntax: `from <module> import <function>`


```python
# only import function 'sin' in package 'math'
from math import sin

# use the function
print(sin(60))
```

That's quite enough for Python. Now, let's practice a little bit.

![](https://farm2.staticflickr.com/1604/24934700445_833f0a5649_t.jpg)
<div style='border-radius: 15px; background: rgb(224,239,130);padding:10px;'/>

### Question 1
#### Question 1.1
Write a function `checkSquareNumber` to check if a integer number is a square number or not. For example, 16 and 9 are square numbers. 15 isn't square number.
Requirements:

- Input: an integer number

- Output: `True` or `False`

HINT: If the square root of a number is integer, then it is a square number.



```python
import math

def checkSquareNumber(x):
    # get the integer part of its square root
    root=math.sqrt(x)
    
    # check if square root is integer
    return (root == int(root))

print(checkSquareNumber(16))
print(checkSquareNumber(250))
```

    True
    False


<div style='border-radius: 15px; background: rgb(224,239,130);padding:10px;'/>
#### Question 1.2
A list `list_numbers` which contains the numbers from 1 to 9999 can be constructed from: 

```python
list_numbers = range(0, 10000)
```

Extract the square numbers in `list_numbers` using function `checkSquareNumber` from question 1.1. How many elements in the extracted list ?


```python
list_numbers = range(0,10000)
square_numbers = list(filter(checkSquareNumber, list_numbers))
print(square_numbers)
print(len(square_numbers))
```

    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961, 1024, 1089, 1156, 1225, 1296, 1369, 1444, 1521, 1600, 1681, 1764, 1849, 1936, 2025, 2116, 2209, 2304, 2401, 2500, 2601, 2704, 2809, 2916, 3025, 3136, 3249, 3364, 3481, 3600, 3721, 3844, 3969, 4096, 4225, 4356, 4489, 4624, 4761, 4900, 5041, 5184, 5329, 5476, 5625, 5776, 5929, 6084, 6241, 6400, 6561, 6724, 6889, 7056, 7225, 7396, 7569, 7744, 7921, 8100, 8281, 8464, 8649, 8836, 9025, 9216, 9409, 9604, 9801]
    100


<div style='border-radius: 15px; background: rgb(224,239,130);padding:10px;'/>
#### Question 1.3

Using array slicing, select the elements of the list square_numbers, whose index is from 5 to 20 (zero-based index).


```python
print(square_numbers[5:21])
```

    [25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400]


Next, we will take a quick look on Numpy - a powerful module of Python.

## 2.2. Numpy
Numpy is the core library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays.
### 2.2.1. Array
A numpy array is a grid of values, all of **the same type**, and is indexed by a tuple of nonnegative integers. Thanks to the same type property, Numpy has the benefits of [locality of reference](https://en.wikipedia.org/wiki/Locality_of_reference). Besides, many other Numpy operations are implemented in C, avoiding the general cost of loops in Python, pointer indirection and per-element dynamic type checking. So, the speed of Numpy is often faster than using built-in datastructure of Python. When working with massive data with computationally expensive tasks, you should consider to use Numpy. 

The number of dimensions is the `rank` of the array; the `shape` of an array is a tuple of integers giving the size of the array along each dimension.

We can initialize numpy arrays from nested Python lists, and access elements using square brackets:


```python
import numpy as np

# Create a rank 1 array
rank1_array = np.array([1, 2, 3])
print("type of rank1_array:", type(rank1_array))
print("shape of rank1_array:", rank1_array.shape)
print("elements in rank1_array:", rank1_array[0], rank1_array[1], rank1_array[2])

# Create a rank 2 array
rank2_array = np.array([[1,2,3],[4,5,6]])
print("shape of rank2_array:", rank2_array.shape)
print(rank2_array[0, 0], rank2_array[0, 1], rank2_array[1, 0])
```

    type of rank1_array: <class 'numpy.ndarray'>
    shape of rank1_array: (3,)
    elements in rank1_array: 1 2 3
    shape of rank2_array: (2, 3)
    1 2 4


### 2.2.2. Array slicing
Similar to Python lists, numpy arrays can be sliced. The different thing is that you must specify a slice for each dimension of the array because arrays may be multidimensional.


```python
m_array = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2
b = m_array[:2, 1:3]
print(b)

# we can only this syntax with numpy array, not python list
print("value at row 0, column 1:", m_array[0, 1])

# Rank 1 view of the second row of m_array  
print("the second row of m_array:", m_array[1, :])

# print element at position (0,2) and (1,3)
print(m_array[[0,1], [2,3]])
```

    [[2 3]
     [6 7]]
    value at row 0, column 1: 2
    the second row of m_array: [5 6 7 8]
    [3 8]


### 2.2.3. Boolean array indexing
We can use boolean array indexing to check whether each element in the array satisfies a condition or use it to do filtering.


```python
m_array = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Find the elements of a that are bigger than 2
# this returns a numpy array of Booleans of the same
# shape as a, where each value of bool_idx tells
# whether that element of a is > 3 or not
bool_idx = (m_array > 3)
print(bool_idx , "\n")

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(m_array[bool_idx], "\n")

# We can combine two statements
print(m_array[m_array > 3], "\n")

# select elements with multiple conditions
print(m_array[(m_array > 3) & (m_array % 2 == 0)])

```

    [[False False False  True]
     [ True  True  True  True]
     [ True  True  True  True]] 
    
    [ 4  5  6  7  8  9 10 11 12] 
    
    [ 4  5  6  7  8  9 10 11 12] 
    
    [ 4  6  8 10 12]


### 2.2.4. Datatypes
Remember that the elements in a numpy array have the same type. When constructing arrays, Numpy tries to guess a datatype when you create an array However, we can specify the datatype explicitly via an optional argument.


```python
# let Numpy guess the datatype
x1 = np.array([1, 2])
print(x1.dtype)

# force the datatype be float64
x2 = np.array([1, 2], dtype=np.float64)
print(x2.dtype)
```

    int64
    float64


### 2.2.5. Array math
Similar to Matlab or R, in Numpy, basic mathematical functions operate elementwise on arrays, and are available both as operator overloads and as functions in the numpy module.


```python
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)
# mathematical function is used as operator
print("x + y =", x + y, "\n")

# mathematical function is used as function
print("np.add(x, y)=", np.add(x, y), "\n")

# Unlike MATLAB, * is elementwise multiplication
# not matrix multiplication
print("x * y =", x * y , "\n")
print("np.multiply(x, y)=", np.multiply(x, y), "\n")
print("x*2=", x*2, "\n")

# to multiply two matrices, we use dot function
print("x.dot(y)=", x.dot(y), "\n")
print("np.dot(x, y)=", np.dot(x, y), "\n")

# Elementwise square root
print("np.sqrt(x)=", np.sqrt(x), "\n")
```

    x + y = [[  6.   8.]
     [ 10.  12.]] 
    
    np.add(x, y)= [[  6.   8.]
     [ 10.  12.]] 
    
    x * y = [[  5.  12.]
     [ 21.  32.]] 
    
    np.multiply(x, y)= [[  5.  12.]
     [ 21.  32.]] 
    
    x*2= [[ 2.  4.]
     [ 6.  8.]] 
    
    x.dot(y)= [[ 19.  22.]
     [ 43.  50.]] 
    
    np.dot(x, y)= [[ 19.  22.]
     [ 43.  50.]] 
    
    np.sqrt(x)= [[ 1.          1.41421356]
     [ 1.73205081  2.        ]] 
    


Note that unlike MATLAB, `*` is elementwise multiplication, not matrix multiplication. We instead use the `dot` function to compute inner products of vectors, to multiply a vector by a matrix, and to multiply matrices.


```python
# declare two vectors
v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors
print("v.dot(w)=", v.dot(w))
print("np.dot(v, w)=", np.dot(v, w))

# Matrix / vector product
print("x.dot(v)=", x.dot(v))
print("np.dot(x, v)=", np.dot(x, v))

# Matrix / matrix product
print("x.dot(y)=", x.dot(y))
print("np.dot(x, y)=", np.dot(x, y))
```

    v.dot(w)= 219
    np.dot(v, w)= 219
    x.dot(v)= [ 29.  67.]
    np.dot(x, v)= [ 29.  67.]
    x.dot(y)= [[ 19.  22.]
     [ 43.  50.]]
    np.dot(x, y)= [[ 19.  22.]
     [ 43.  50.]]


Besides, we can do other aggregation computations on arrays such as `sum`, `nansum`, or `T`.


```python
x = np.array([[1,2], [3,4]])

# Compute sum of all elements
print(np.sum(x))

# Compute sum of each column
print(np.sum(x, axis=0))

# Compute sum of each row
print(np.sum(x, axis=1))

# transpose the matrix
print(x.T)

# Note that taking the transpose of a rank 1 array does nothing:
v = np.array([1,2,3])
print(v.T)  # Prints "[1 2 3]"
```

    10
    [4 6]
    [3 7]
    [[1 3]
     [2 4]]
    [1 2 3]


![](https://farm2.staticflickr.com/1604/24934700445_833f0a5649_t.jpg)
<div style='border-radius: 15px; background: rgb(224,239,130);padding:10px;'/>

### Question 2

Given a 2D array:

```
 1  2  3  4
 5  6  7  8 
 9 10 11 12
13 14 15 16
```

#### Question 2.1

Print the all odd numbers in this array using `Boolean array indexing`.



```python
array_numbers = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])

print(array_numbers[array_numbers % 2 ==1])
```

    [ 1  3  5  7  9 11 13 15]


<div style='border-radius: 15px; background: rgb(224,239,130);padding:10px;'/>
#### Question 2.2

Extract the second row and the third column in this array using `array slicing`.


```python
print(array_numbers[1, :])
print(array_numbers[:, 2])
```

    [ 9 10 11 12]
    [ 4  8 12 16]


<div style='border-radius: 15px; background: rgb(224,239,130);padding:10px;'/>
#### Question 2.3
Calculate the sum of diagonal elements.


```python
sum = 0
for i in range(0, array_numbers.shape[0]):
    sum += array_numbers[i, i]
    
print(sum)
```

    34


<div style='border-radius: 15px; background: rgb(224,239,130);padding:10px;'/>
#### Question 2.4
Print elementwise multiplication of the first row and the last row using numpy's functions.

Print the inner product of these two rows.



```python
print(np.multiply(array_numbers[0,:], array_numbers[-1,:]))
print(np.dot(array_numbers[0,:], array_numbers[-1,:]))
```

    [13 28 45 64]
    150


## 2.3. Matplotlib

As its name indicates, Matplotlib is a plotting library. It provides both a very quick way to visualize data from Python and publication-quality figures in many formats. The most important function in matplotlib is `plot`, which allows you to plot 2D data.


```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.ylabel('custom y label')
plt.show()
```


![png](output_70_0.png)


In this case, we provide a single list or array to the `plot()` command, matplotlib assumes it is a sequence of y values, and automatically generates the x values for us. Since python ranges start with 0, the default x vector has the same length as y but starts with 0. Hence the x data are [0,1,2,3].

In the next example, we plot figure with both x and y data. Besides, we want to draw dashed lines instead of the solid in default.


```python
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'r--')
plt.plot([1, 2, 3, 4], [1, 3, 5, 11], 'b--')
plt.show()

plt.bar([1, 2, 3, 4], [1, 4, 9, 16], align='center')
# labels of each column bar
x_labels = ["Type 1", "Type 2", "Type 3", "Type 4"]
# assign labels to the plot
plt.xticks([1, 2, 3, 4], x_labels)

plt.show()
```


![png](output_72_0.png)



![png](output_72_1.png)


If we want to merge two figures into a single one, subplot is the best way to do that. For example, we want to put two figures in a stack vertically, we should define a grid of plots with 2 rows and 1 column. Then, in each row, a single figure is plotted.


```python
# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'r--')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.bar([1, 2, 3, 4], [1, 4, 9, 16])

plt.show()
```


![png](output_74_0.png)


For more examples, please visit the [homepage](http://matplotlib.org/1.5.1/examples/index.html) of Matplotlib.


![](https://farm2.staticflickr.com/1604/24934700445_833f0a5649_t.jpg)

<div style='border-radius: 15px; background: rgb(224,239,130);padding:10px;'/>

###  Question 3
Given a list of numbers from 0 to 9999.

#### Question 3.1
Calculate the histogram of numbers divisible by 3, 7, 11 in the list respectively.

( Or in other word, how many numbers divisible by 3, 7, 11 in the list respectively ?)


```python
arr = np.array(range(10000))
divisors = [3, 7, 11]
histogram = [len(arr[arr % divisors[x] == 0]) for x in range(len(divisors))]
print(histogram)
```

    [3334, 1429, 910]


<div style='border-radius: 15px; background: rgb(224,239,130);padding:10px;'/>
#### Question 3.2
Plot the histogram in a line chart.


```python
%matplotlib inline
import matplotlib.pyplot as plt

# simple line chart
plt.plot(histogram)
x_indexes = range(len(divisors))
x_names = list(divisors)
plt.xticks(x_indexes, x_names)
plt.show()
```


![png](output_79_0.png)


<div style='border-radius: 15px; background: rgb(224,239,130);padding:10px;'/>
#### Question 3.3
Plot the histogram in a bar chart.


```python
# char chart with x-lables
x_indexes = range(len(divisors))
x_names = list(divisors)
plt.bar( x_indexes, histogram, align='center')
plt.xticks(x_indexes, x_names)
plt.show()
```


![png](output_81_0.png)


## 2.4. Pandas

Pandas is an open source library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language. Indeeed, it is great for data manipulation, data analysis, and data visualization.

### 2.4.1. Data structures
Pandas introduces has two useful (and powerful) structures: `Series` and `DataFrame`, both of which are built on top of NumPy.

#### Series
A `Series` is a one-dimensional object similar to an array, list, or even column in a table. It assigns a *labeled index* to each item in the Series. By default, each item will receive an index label from `0` to `N-1`, where `N` is the number items of `Series`.

We can create a Series by passing a list of values, and let pandas create a default integer index.



```python
import pandas as pd
import numpy as np

# create a Series with an arbitrary list
s = pd.Series([3, 'Machine learning', 1.414259, -65545, 'Happy coding!'])
print(s)
```

    0                   3
    1    Machine learning
    2             1.41426
    3              -65545
    4       Happy coding!
    dtype: object


Or, an index can be used explixitly when creating the `Series`.


```python
s = pd.Series([3, 'Machine learning', 1.414259, -65545, 'Happy coding!'],
             index=['Col1', 'Col2', 'Col3', 4.1, 5])
print(s)
```

    Col1                   3
    Col2    Machine learning
    Col3             1.41426
    4.1               -65545
    5          Happy coding!
    dtype: object


A `Series` can be constructed from a dictionary too.


```python
s = pd.Series({
        'Col1': 3, 'Col2': 'Machine learning', 
        'Col3': 1.414259, 4.1: -65545, 
        5: 'Happy coding!'
    })
print(s)
```

    4.1               -65545
    Col1                   3
    Col2    Machine learning
    5          Happy coding!
    Col3             1.41426
    dtype: object


We can access items in a `Series` in a same way as `Numpy`.


```python
s = pd.Series({
        'Col1': 3, 'Col2': -10, 
        'Col3': 1.414259, 
        4.1: -65545, 
        5: 8
    })

# get element which has index='Col1'
print("s['Col1']=", s['Col1'], "\n")

# get elements whose index is in a given list
print("s[['Col1', 'Col3', 4.5]]=", s[['Col1', 'Col3', 4.5]], "\n")

# use boolean indexing for selection
print(s[s > 0], "\n")

# modify elements on the fly using boolean indexing
s[s > 0] = 15

print(s, "\n")

# mathematical operations can be done using operators and functions.
print(s*10,  "\n")
print(np.square(s), "\n")
```

    s['Col1']= 3.0 
    
    s[['Col1', 'Col3', 4.5]]= Col1    3.000000
    Col3    1.414259
    4.5          NaN
    dtype: float64 
    
    Col1    3.000000
    5       8.000000
    Col3    1.414259
    dtype: float64 
    
    4.1    -65545
    Col1       15
    Col2      -10
    5          15
    Col3       15
    dtype: float64 
    
    4.1    -655450
    Col1       150
    Col2      -100
    5          150
    Col3       150
    dtype: float64 
    
    4.1     4296147025
    Col1           225
    Col2           100
    5              225
    Col3           225
    dtype: float64 
    


#### DataFrame
A DataFrame is a tablular data structure comprised of rows and columns, akin to database table, or R's data.frame object. In a loose way, we can also think of a DataFrame as a group of Series objects that share an index (the column names).

We can create a DataFrame by passing a dict of objects that can be converted to series-like.


```python
data = {'year': [2013, 2014, 2015, 2013, 2014, 2015, 2013, 2014],
        'team': ['Manchester United', 'Chelsea', 'Asernal', 'Liverpool', 'West Ham', 'Newcastle', 'Machester City', 'Tottenham'],
        'wins': [11, 8, 10, 15, 11, 6, 10, 4],
        'losses': [5, 8, 6, 1, 5, 10, 6, 12]}
football = pd.DataFrame(data, columns=['year', 'team', 'wins', 'losses'])
football
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>team</th>
      <th>wins</th>
      <th>losses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013</td>
      <td>Manchester United</td>
      <td>11</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014</td>
      <td>Chelsea</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015</td>
      <td>Asernal</td>
      <td>10</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013</td>
      <td>Liverpool</td>
      <td>15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014</td>
      <td>West Ham</td>
      <td>11</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015</td>
      <td>Newcastle</td>
      <td>6</td>
      <td>10</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2013</td>
      <td>Machester City</td>
      <td>10</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2014</td>
      <td>Tottenham</td>
      <td>4</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>



We can store data as a CSV file, or read data from a CSV file.


```python
# save data to a csv file without the index
football.to_csv('football.csv', index=False)

from_csv = pd.read_csv('football.csv')
from_csv.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>team</th>
      <th>wins</th>
      <th>losses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013</td>
      <td>Manchester United</td>
      <td>11</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014</td>
      <td>Chelsea</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015</td>
      <td>Asernal</td>
      <td>10</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013</td>
      <td>Liverpool</td>
      <td>15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014</td>
      <td>West Ham</td>
      <td>11</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



To read a CSV file with a custom delimiter between values and custom columns' names, we can use parameters `sep` and `names` relatively.
Moreover, Pandas also supports to read and write to [Excel file](http://pandas.pydata.org/pandas-docs/stable/io.html#io-excel) , sqlite database file, URL,  or even clipboard.

We can have an overview on the data by using functions `info` and `describe`.


```python
print(football.info(), "\n")
football.describe()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 8 entries, 0 to 7
    Data columns (total 4 columns):
    year      8 non-null int64
    team      8 non-null object
    wins      8 non-null int64
    losses    8 non-null int64
    dtypes: int64(3), object(1)
    memory usage: 320.0+ bytes
    None 
    





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>wins</th>
      <th>losses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2013.875000</td>
      <td>9.375000</td>
      <td>6.625000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.834523</td>
      <td>3.377975</td>
      <td>3.377975</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2013.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2013.000000</td>
      <td>7.500000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2014.000000</td>
      <td>10.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2014.250000</td>
      <td>11.000000</td>
      <td>8.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2015.000000</td>
      <td>15.000000</td>
      <td>12.000000</td>
    </tr>
  </tbody>
</table>
</div>



Numpy's regular slicing syntax works as well.


```python
print(football[0:2], "\n")

# query only the teams that win more than 10 matches from 2014
print(football[(football.year >= 2014) & (football.wins >= 10)])
```

       year               team  wins  losses
    0  2013  Manchester United    11       5
    1  2014            Chelsea     8       8 
    
       year      team  wins  losses
    2  2015   Asernal    10       6
    4  2014  West Ham    11       5


An important feature that Pandas supports is `JOIN`. Very often, the data comes from multiple sources, in multiple files. For example, we have 2 CSV files, one contains the information of Artists, the other contains information of Songs. If we want to query the artist name and his/her corresponding songs, we have to do joining two dataframe.

Similar to SQL, in Pandas, you can do inner join, left outer join, right outer join and full outer join. Let's see a small example. Assume that we have two dataset of singers and songs. The relationship between two datasets is maintained by a constrain on `singer_code`.


```python
singers = pd.DataFrame({'singer_code': range(5), 
                           'singer_name': ['singer_a', 'singer_b', 'singer_c', 'singer_d', 'singer_e']})
songs = pd.DataFrame({'singer_code': [2, 2, 3, 4, 5], 
                           'song_name': ['song_f', 'song_g', 'song_h', 'song_i', 'song_j']})
print(singers)
print('\n')
print(songs)
```

       singer_code singer_name
    0            0    singer_a
    1            1    singer_b
    2            2    singer_c
    3            3    singer_d
    4            4    singer_e
    
    
       singer_code song_name
    0            2    song_f
    1            2    song_g
    2            3    song_h
    3            4    song_i
    4            5    song_j



```python
# inner join
pd.merge(singers, songs, on='singer_code', how='inner')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>singer_code</th>
      <th>singer_name</th>
      <th>song_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>singer_c</td>
      <td>song_f</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>singer_c</td>
      <td>song_g</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>singer_d</td>
      <td>song_h</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>singer_e</td>
      <td>song_i</td>
    </tr>
  </tbody>
</table>
</div>




```python
# left join
pd.merge(singers, songs, on='singer_code', how='left')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>singer_code</th>
      <th>singer_name</th>
      <th>song_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>singer_a</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>singer_b</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>singer_c</td>
      <td>song_f</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>singer_c</td>
      <td>song_g</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>singer_d</td>
      <td>song_h</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>singer_e</td>
      <td>song_i</td>
    </tr>
  </tbody>
</table>
</div>




```python
# right join
pd.merge(singers, songs, on='singer_code', how='right')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>singer_code</th>
      <th>singer_name</th>
      <th>song_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>singer_c</td>
      <td>song_f</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>singer_c</td>
      <td>song_g</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>singer_d</td>
      <td>song_h</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>singer_e</td>
      <td>song_i</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>NaN</td>
      <td>song_j</td>
    </tr>
  </tbody>
</table>
</div>




```python
# outer join (full join)
pd.merge(singers, songs, on='singer_code', how='outer')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>singer_code</th>
      <th>singer_name</th>
      <th>song_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>singer_a</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>singer_b</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>singer_c</td>
      <td>song_f</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>singer_c</td>
      <td>song_g</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>singer_d</td>
      <td>song_h</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>singer_e</td>
      <td>song_i</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5</td>
      <td>NaN</td>
      <td>song_j</td>
    </tr>
  </tbody>
</table>
</div>



We can also concat two dataframes vertically or horizontally via function `concat` and parameter `axis`. This function is useful when we need to append two similar datasets or to put them side by site


```python
# concat vertically
pd.concat([singers, songs])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>singer_code</th>
      <th>singer_name</th>
      <th>song_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>singer_a</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>singer_b</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>singer_c</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>singer_d</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>singer_e</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>NaN</td>
      <td>song_f</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>NaN</td>
      <td>song_g</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>NaN</td>
      <td>song_h</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>NaN</td>
      <td>song_i</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>NaN</td>
      <td>song_j</td>
    </tr>
  </tbody>
</table>
</div>




```python
# concat horizontally
pd.concat([singers, songs], axis=1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>singer_code</th>
      <th>singer_name</th>
      <th>singer_code</th>
      <th>song_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>singer_a</td>
      <td>2</td>
      <td>song_f</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>singer_b</td>
      <td>2</td>
      <td>song_g</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>singer_c</td>
      <td>3</td>
      <td>song_h</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>singer_d</td>
      <td>4</td>
      <td>song_i</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>singer_e</td>
      <td>5</td>
      <td>song_j</td>
    </tr>
  </tbody>
</table>
</div>



When doing statistic, we usually need to aggregate data by each group. For example, to anwser the question "how many songs each singer has?", we have to group data by each singer, and then calculate the number of songs in each group. Not that the result must contain the statistic of all singers in database (even if some of them have no song)


```python
data = pd.merge(singers, songs, on='singer_code', how='left')

# count the values of each column in group
print(data.groupby('singer_code').count())

print("\n")

# count only song_name
print(data.groupby('singer_code').song_name.count())

print("\n")

# count song name but ignore duplication, and order the result
print(data.groupby('singer_code').song_name.nunique().sort_values(ascending=True))
```

                 singer_name  song_name
    singer_code                        
    0                      1          0
    1                      1          0
    2                      2          2
    3                      1          1
    4                      1          1
    
    
    singer_code
    0    0
    1    0
    2    2
    3    1
    4    1
    Name: song_name, dtype: int64
    
    
    singer_code
    0    0
    1    0
    3    1
    4    1
    2    2
    Name: song_name, dtype: int64


![](https://farm2.staticflickr.com/1604/24934700445_833f0a5649_t.jpg)
<div style='border-radius: 15px; background: rgb(224,239,130);padding:10px;'/>


### Question 4

We have two datasets about music: [song](https://github.com/michiard/AML-COURSE/blob/master/data/song.tsv) and [album](https://github.com/michiard/AML-COURSE/blob/master/data/album.tsv).

In the following questions, you **have to** use Pandas to load data and write code to answer these questions.

#### Question 4.1
Load both dataset into two dataframes and print the information of each dataframe

**HINT**: 

- You can click button `Raw` on the github page of each dataset and copy the URL of the raw file.
- The dataset can be load by using function `read_table`. For example: `df = pd.read_table(raw_url, sep='\t')`



```python
import pandas as pd

songdb_url = 'https://raw.githubusercontent.com/DistributedSystemsGroup/Algorithmic-Machine-Learning/master/data/song.tsv'
albumdb_url = 'https://raw.githubusercontent.com/DistributedSystemsGroup/Algorithmic-Machine-Learning/master/data/album.tsv'
song_df = pd.read_table(songdb_url, sep='\t')
album_df = pd.read_table(albumdb_url, sep='\t')

print(song_df.info)
print()
print(album_df.info)
```

    <bound method DataFrame.info of             Singer                  Song  Album Length
    0  Michael Jackson                 2 bad      1   4:07
    1  Michael Jackson           Unbreakable      2   6:26
    2  Michael Jackson       Don't Walk Away      2   4:25
    3  Michael Jackson         Break of Dawn      2   5:33
    4     Taylor Swift          All Too Well      5   5:29
    5     Taylor Swift             Bad Blood      7   3:19
    6     Taylor Swift      Back to December      6   4:54
    7  Michael Jackson          Human Nature      4   4:06
    8  Michael Jackson          Baby Be Mine      4   4:20
    9  Michael Jackson  What More Can I Give    NaN   3:36>
    
    <bound method DataFrame.info of    Album code                                 Album name  Year
    0           1  HIStory: Past, Present and Future, Book I  1995
    1           2                                 Invincible  2001
    2           3                                        Bad  1986
    3           4                                   Thriller  1982
    4           5                                        Red  2012
    5           6                                  Speak Now  2010
    6           7                                       1989  2014>


<div style='border-radius: 15px; background: rgb(224,239,130);padding:10px;'/>

#### Question 4.2
How many albums in this datasets ?

How many songs in this datasets ?


```python
print("number of albums:", album_df['Album code'].count())
print("number of songs:", song_df.Song.count())
```

    number of albums: 7
    number of songs: 10


<div style='border-radius: 15px; background: rgb(224,239,130);padding:10px;'/>
#### Question 4.3
How many distinct singers in this dataset ?


```python
print("number distinct singers:", len(song_df.groupby('Singer')))
```

    number distinct singers: 2


<div style='border-radius: 15px; background: rgb(224,239,130);padding:10px;'/>
#### Question 4.4
Is there any song that doesn't belong to any album ?

Is there any album that has no song ?

**HINT**: 

- To join two datasets on different key names, we use `left_on=` and `right_on=` instead of `on=`.
- Funtion `notnull` and `isnull` help determining the value of a column is missing or not. For example:
`df['song'].isnull()`.


```python
fulldf = pd.merge(song_df, album_df, how='outer', left_on='Album', right_on='Album code')
fulldf[fulldf['Song'].notnull() & fulldf['Album'].isnull()]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Singer</th>
      <th>Song</th>
      <th>Album</th>
      <th>Length</th>
      <th>Album code</th>
      <th>Album name</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>Michael Jackson</td>
      <td>What More Can I Give</td>
      <td>NaN</td>
      <td>3:36</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
fulldf[fulldf['Song'].isnull() & fulldf['Album code'].notnull()]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Singer</th>
      <th>Song</th>
      <th>Album</th>
      <th>Length</th>
      <th>Album code</th>
      <th>Album name</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>Bad</td>
      <td>1986</td>
    </tr>
  </tbody>
</table>
</div>



<div style='border-radius: 15px; background: rgb(224,239,130);padding:10px;'/>

#### Question 4.5
How many songs in each albums of Michael Jackson ?


```python
# Try thinking like as for map reduce word count!!

fulldf[fulldf['Singer']=='Michael Jackson'].groupby('Album code').Song.count()
```




    Album code
    1    1
    2    3
    4    2
    Name: Song, dtype: int64



# 3. PySpark

Spark is an open source alternative to MapReduce designed to make it easier to build and run fast data manipulation on Hadoop. Spark comes with a library of machine learning (ML) and graph algorithms, and also supports real-time streaming and SQL apps, via Spark Streaming and Shark, respectively. Spark exposes the Spark programming model to Java, Scala, or Python. In Python, we use PySpark API to interact with Spark.

As discussed in the introductory lecture, every Spark application has a Spark driver. It is the program that declares the transformations and actions on RDDs of data and submits such requests to the cluster manager. Actually, the driver is the program that creates the `SparkContext`, connecting to a given cluster manager such as  Spark Master, YARN cluster manager\[[2](http://blog.cloudera.com/blog/2014/05/apache-spark-resource-management-and-yarn-app-models/)\]... The executors run user code, run computations and can cache data for your application. The `SparkContext` will create a job that is broken into stages. The stages are broken into tasks which are scheduled by the SparkContext on an executor.

![](http://blogs.msdn.com/cfs-file.ashx/__key/communityserver-blogs-components-weblogfiles/00-00-01-61-78-metablogapi/3566.091415_5F00_1429_5F00_Understandi1.png)

When starting PySpark with command `pyspark` or using a well configurated notebook (such as this one), `SparkContext` is created automatically in variable `sc`. 



```python
sc
```




    <pyspark.context.SparkContext at 0x7f2ad8d02f98>



PySpark uses PySpark RDDs which  are just RDDs of Python objects: like Python lists, they can store objects with mixed types (actually all the objects are instances of `PyObject`).

When PySpark is started, it also starts a JVM, which is accessible through a socket. PySpark uses the `Py4J` project to handle this communication. The JVM works as the actual Spark driver, and loads a `JavaSparkContext` that communicates with the Spark executors across the cluster. Python API calls to the Spark Context object are then **translated into Java API calls** to the JavaSparkContext. For example, the implementation of PySpark's `sc.textFile()` dispatches a call to the `.textFile` method of the `JavaSparkContext`, which ultimately communicates with the Spark executor JVMs to load the text data from HDFS. 

![](http://i.imgur.com/YlI8AqEl.png)

The Spark executors on the cluster start a Python interpreter for each core, with which they communicate data through a pipe when they need to execute user-code. A Python RDD in the local PySpark client corresponds to a `PythonRDD` object in the local JVM. The data associated with the RDD actually lives in the Spark JVMs as Java objects. For example, running `sc.textFile()` in the Python interpreter will call the `JavaSparkContexts` `textFile` method, which loads the data as Java String objects in the cluster.


When an API call is made on the `PythonRDD`, any associated code (e.g., Python lambda function) **is serialized and distributed to the executors**. The data is then converted from Java objects to a Python-compatible representation (e.g., pickle objects) and streamed to executor-associated Python interpreters through a pipe. Any necessary Python processing is executed in the interpreter, and the resulting data is stored back as an RDD (as pickle objects by default) in the JVMs. 


The data is read easily by using functions of Spark Context. For example, to read a text file and count the number of lines, we can write:



```python
# each line is stored as an element in 'words' - a PythonRDD.
words = sc.textFile("/datasets/textfile")
num_lines = words.count()
print("the number of lines in file", num_lines)
```

    the number of lines in file 31


## 3.1. Wordcount example
In the below example, we try to select top 10 words which has occurred the most in a text file and plot them using Matplotlib.

To do this exercise, we go through the workflow to understand what we should do. First, using method `textFile` of SparkContext `sc`, we create a RDD of string. Each string in RDD is representative for a line in the text file. In a loose way, we can think the first RDD is a RDD of lines. 

Because we work on the scope of words, we have to transform **a line** in the current RDD into **multiple words**, each word is an object of the new RDD. This is done by using `flatMap` function. 

Then, a `map` function will transform **each word** in RDD into **a** tuple with 2 components: the word itselft and 1. At this time, each object of the RDD is actually a key-value pair. Number 1 here mean that we have encountered one time. 

We can take advantage of function `reduceByKey` to sum all frequencies of the same word. Now, each element in the RDD is in form of: (word, total_frequency). To sort the words by its frequency, we can have many ways. One of the simplest approach is swap each tuple such that the frequency will become the key and then use `sortByKey` function.


```python
words = (
            # read the text file
            sc.textFile("/datasets/textfile")
            
            # construct words from lines
            .flatMap(lambda line: line.split())
            
            # map each word to (word, 1)
            .map(lambda x: (x, 1))
    
            # reduce by key: accumulate sum the freq of the same word
            .reduceByKey(lambda freq1, freq2: freq1 + freq2)
            
            # swap (word, freq) to (freq, word)
            .map(lambda x: (x[1], x[0]))
    
            # sort result by key DESC
            .sortByKey(False)
         )
```

Now the top-10 words are collected and sent back to the driver by using function `take`.


```python
# top 10 words:
top10 = words.take(10)
print(top10)
```

    [(13, 'the'), (9, 'on'), (9, 'in'), (8, 'Notebooks'), (8, 'we'), (8, 'of'), (7, 'and'), (7, 'to'), (6, 'are'), (5, 'for')]


The function `collect` will sent all elements in the RDD to the driver as an list.


```python
# collect results from executors to the driver
results = words.collect()
print(results)
```

    [(13, 'the'), (9, 'on'), (9, 'in'), (8, 'Notebooks'), (8, 'we'), (8, 'of'), (7, 'and'), (7, 'to'), (6, 'are'), (5, 'for'), (5, 'a'), (5, 'use'), (5, 'is'), (4, 'by'), (4, 'This'), (4, '*'), (4, 'this'), (3, 'our'), (3, 'be'), (3, 'Data'), (3, '##'), (3, 'work'), (2, 'course'), (2, 'students'), (2, 'could'), (2, 'will'), (2, 'from'), (2, 'that'), (2, 'repository'), (2, 'Our'), (2, 'which'), (2, 'contains'), (2, 'Eurecom.'), (2, 'can'), (2, 'Working'), (2, 'concepts'), (2, 'Students'), (2, 'The'), (2, 'run'), (2, 'cases'), (2, 'based'), (2, 'Finally,'), (2, 'at'), (2, 'Spark'), (2, 'statistical'), (1, 'companies'), (1, 'learning'), (1, 'writing'), (1, 'Technical'), (1, 'commenting'), (1, 'form'), (1, 'Course'), (1, 'Machine'), (1, 'progress!!**'), (1, 'instead'), (1, '[Zoe](http://zoe-analytics.eu/),'), (1, 'It'), (1, 'systems.'), (1, 'challenges,'), (1, 'book'), (1, 'Notebooks.'), (1, 'descriptive'), (1, 'Building'), (1, 'several'), (1, 'notes'), (1, 'course,'), (1, 'hands-on'), (1, 'illustrated'), (1, 'algorithm,'), (1, 'theoretical'), (1, 'student'), (1, 'clusters'), (1, 'code,'), (1, 'call'), (1, 'kernel.'), (1, 'private,'), (1, 'have'), (1, 'version'), (1, 'parallel'), (1, '"solved'), (1, 'whereas'), (1, 'platform'), (1, 'managed'), (1, '#'), (1, 'Python'), (1, 'group'), (1, 'motivated'), (1, 'Sources'), (1, "students'"), (1, 'defined'), (1, 'repository.'), (1, 'cloud'), (1, 'Wills.'), (1, 'selected'), (1, 'validation'), (1, 'Josh'), (1, 'cleaning'), (1, 'container,'), (1, 'preparation'), (1, 'machine'), (1, '[Duc-Trung'), (1, 'pushed'), (1, 'kernel'), (1, 'complete.'), (1, 'systems'), (1, 'goal'), (1, 'Uri'), (1, 'Algorithmic'), (1, 'Owen'), (1, 'Analytics'), (1, 'Spark](http://shop.oreilly.com/product/0636920035091.do),'), (1, 'publicly'), (1, 'Science'), (1, 'cannot'), (1, 'Apache'), (1, 'tasks'), (1, 'different'), (1, 'results.'), (1, '[Advanced'), (1, 'still'), (1, 'Monte'), (1, 'heterogeneity'), (1, 'cluster'), (1, 'choice'), (1, 'notebook,'), (1, 'require'), (1, 'gain'), (1, 'find'), (1, 'taking'), (1, 'computing'), (1, 'answering'), (1, 'with.'), (1, 'Some'), (1, 'e.g.,'), (1, 'you'), (1, 'running'), (1, 'model,'), (1, 'industrial'), (1, 'some'), (1, 'Ryza,'), (1, 'For'), (1, 'acknowledgments'), (1, 'majority'), (1, 'Learning'), (1, 'teach'), (1, 'EURECOM'), (1, 'classified'), (1, 'lectures'), (1, 'courses'), (1, 'experimental'), (1, 'mainly'), (1, 'with'), (1, 'all'), (1, 'roughly'), (1, 'Carlo'), (1, 'Nguyen](http://www.eurecom.fr/en/people/nguyen-duc-trung),'), (1, 'case'), (1, 'Jupyter'), (1, 'address'), (1, 'inference,'), (1, 'without'), (1, 'statistics'), (1, 'questions,'), (1, 'data'), (1, 'Notebooks,'), (1, 'Objectives'), (1, 'in:'), (1, 'PhD'), (1, 'built.'), (1, 'what'), (1, 'not'), (1, 'merges'), (1, 'on...'), (1, 'Sandy'), (1, 'skills'), (1, 'front-end'), (1, 'available'), (1, 'AML-COURSE'), (1, 'Sean'), (1, 'each'), (1, 'system'), (1, 'Laserson,'), (1, 'data,'), (1, 'containers.'), (1, 'simulations'), (1, 'exercises".'), (1, 'user-facing'), (1, 'projects.'), (1, 'my'), (1, 'building'), (1, 'background.'), (1, 'requires'), (1, 'configured'), (1, 'API.'), (1, '**NOTE:'), (1, '&'), (1, 'achieved'), (1, 'so'), (1, 'Scientists'), (1, 'nicely'), (1, 'learn'), (1, 'notebooks'), (1, 'offer'), (1, 'distributed'), (1, 'here'), (1, 'expect'), (1, 'contact'), (1, 'experience'), (1, 'analytics-as-a-service'), (1, 'focus'), (1, 'container-based')]


There are two type of functions in Spark: **transformation** and **action**. All functions `map`, `flatMap`, `reduceByKey`, `sortByKey` are transformation functions. They are not executed right away when called. Indeed, Spark is lazy, so nothing will get executed unless you call some actions such as `count`, `take`, `collect`...

RDD transformations allow us to create dependencies between RDDs. Dependencies are only steps for producing results. Each RDD in lineage chain (string of dependencies) has a function for calculating its data and has a pointer (dependency) to its parent RDD. Everytime we use an RDD, its dependencies is calculated again from beginning. In many cases, that does not take advantage of the pre-computed results. Fortunatly, we can use function `cache` to make a checkpoint for a RDD. Actually, the data of cached RDD can be stored in memory, or disk.

We have a result for our Word Count example. Now, it's time for plotting!


```python
%matplotlib inline
import matplotlib.pyplot as plt

# extract the frequencies from the result
frequencies = [x[0] for x in top10]

# plot the frequencies
plt.plot(frequencies)
plt.show()
```


![png](output_133_0.png)


## 3.2. Night flights example
We have a CSV file which contains the information about flights that took place in the US in 1994.
The data in this file has 29 columns such as `year`, `month`, `day_of_month`, `scheduled_departure_time`,...
We can have a quick look on the data:


```python
! hdfs dfs -cat /datasets/airline/1994.csv | head -n 10
```

    Year,Month,DayofMonth,DayOfWeek,DepTime,CRSDepTime,ArrTime,CRSArrTime,UniqueCarrier,FlightNum,TailNum,ActualElapsedTime,CRSElapsedTime,AirTime,ArrDelay,DepDelay,Origin,Dest,Distance,TaxiIn,TaxiOut,Cancelled,CancellationCode,Diverted,CarrierDelay,WeatherDelay,NASDelay,SecurityDelay,LateAircraftDelay
    1994,1,7,5,858,900,954,1003,US,227,NA,56,63,NA,-9,-2,CLT,ORF,290,NA,NA,0,NA,0,NA,NA,NA,NA,NA
    1994,1,8,6,859,900,952,1003,US,227,NA,53,63,NA,-11,-1,CLT,ORF,290,NA,NA,0,NA,0,NA,NA,NA,NA,NA
    1994,1,10,1,935,900,1023,1003,US,227,NA,48,63,NA,20,35,CLT,ORF,290,NA,NA,0,NA,0,NA,NA,NA,NA,NA
    1994,1,11,2,903,900,1131,1003,US,227,NA,148,63,NA,88,3,CLT,ORF,290,NA,NA,0,NA,0,NA,NA,NA,NA,NA
    1994,1,12,3,933,900,1024,1003,US,227,NA,51,63,NA,21,33,CLT,ORF,290,NA,NA,0,NA,0,NA,NA,NA,NA,NA
    1994,1,13,4,NA,900,NA,1003,US,227,NA,NA,63,NA,NA,NA,CLT,ORF,290,NA,NA,1,NA,0,NA,NA,NA,NA,NA
    1994,1,14,5,903,900,1005,1003,US,227,NA,62,63,NA,2,3,CLT,ORF,290,NA,NA,0,NA,0,NA,NA,NA,NA,NA
    1994,1,15,6,859,900,1004,1003,US,227,NA,65,63,NA,1,-1,CLT,ORF,290,NA,NA,0,NA,0,NA,NA,NA,NA,NA
    1994,1,17,1,859,900,955,1003,US,227,NA,56,63,NA,-8,-1,CLT,ORF,290,NA,NA,0,NA,0,NA,NA,NA,NA,NA
    cat: Unable to write to output stream.


In this example, we only take care about columns `CRSDepTime` (scheduled departure time) and `UniqueCarrier` (carrier of flight). The values of `CRSDepTime` is in format of: hhmm (hour-minute).
Assume that a flight is considered as 'night flight' if its scheduled departured time is late than 18:00.

Questions:

- How many night flights in the data ?
- How many night flights of each unique carrier ? Plot top 5 of them.

First, we read the data and remove the header. Then, from the lines, we extract the information of scheduled departure time and carrier.


```python
# read the data
data = sc.textFile('/datasets/airline/1994.csv')

# extract information about scheduled departure time and carrier
# note that the scheduled time must be convert from string to interger number
def extract_CRSDepTime_Carier(line):
    cols = line.split(",")
    return (int(cols[5]), cols[8])

header = data.first()

# remove header
data_without_header = data.filter(lambda line: line != header)

# screate RDD with only scheduled departure time and carrier information
# cache it for later usages
newdata = (
            data_without_header
               .map(extract_CRSDepTime_Carier)
               .cache()
          )

```

Function `filter` helps us select only the objects that satisfy a condition. In this case, it creates a new RDD by filtering out the header. We can also use it to select the night flights.


```python
night_flights = newdata.filter(lambda f: f[0] > 1800).cache()
night_flights.take(3)
```




    [(2030, 'US'), (2030, 'US'), (2030, 'US')]



We use `cache` because we dont want to recalculate `night_flights` from the beginning everytime of using it.


```python
# filter and count the night flights
num_night_flights = night_flights.count()
print(num_night_flights)
```

    1078203



```python
# group by carrier
night_flights_by_carrier = night_flights.groupBy(lambda x: x[1]).mapValues(lambda flights: len(flights))

# take top 5 carriers
top5_carriers = night_flights_by_carrier.takeOrdered(5, key=lambda x: -x[1])

print(top5_carriers)
```

    [('DL', 208224), ('US', 170397), ('AA', 142832), ('WN', 124024), ('UA', 113640)]


We use `groupBy` to put all flights which belong to the same carrier into a group. In this example, to select top 5 carriers, we don't swap key-value pairs anymore. Alternatively, `takeOrder` can handle that. This function will take top `k` objects ordered by the index. The trick is that we ask it to use the new key, instead of the current one (the carrier).

Let's plot a bar char from the result by Matplotlib. To draw a bar char, we use function `bar` which requires two parameters. Each parameter is a list of float values in each dimension.


```python
%matplotlib inline
import matplotlib.pyplot as plt

# extract the number of flights which will be used as y-values
num_flights = [ x[1] for x in top5_carriers]

# extract the carriers' names
carrier_names = [x[0] for x in top5_carriers]

# create `virtual indexes for carriers which will be used as x-values`
carrier_indexes = range(0, len(carrier_names))

# plot
plt.bar(carrier_indexes, num_flights, align="center")

# put x-labels for the plot
plt.xticks(carrier_indexes, carrier_names)
plt.show()
```


![png](output_144_0.png)


![](https://farm2.staticflickr.com/1604/24934700445_833f0a5649_t.jpg)
<div style='border-radius: 15px; background: rgb(224,239,130);padding:10px;'/>


## Question 5


### Question 5.1
Calculate how many flights have the scheduled departure time after 09:00 and before 14:00.


```python
# read the data
data = sc.textFile('/datasets/airline/1994.csv')

# extract information about scheduled departure time and origin ariport
# note that the scheduled time must be convert from string to interger number
def extract_CRSDepTime_Carier(line):
    cols = line.split(",")
    return (int(cols[5]), cols[16])

header = data.first()

# remove header
data_without_header = data.filter(lambda line: line != header)

# create RDD with only scheduled departure time and origin ariport information
# cache it for later usages
newdata = (
            data_without_header
               .map(extract_CRSDepTime_Carier)
               .cache()
          )

flights = newdata.filter(lambda f: (f[0] > 900 and f[0] < 1400)).cache()

print(flights.count())
```

    1625828


<div style='border-radius: 15px; background: rgb(224,239,130);padding:10px;'/>
### Question 5.2
Calculate the number flights that have a scheduled departure time after 09:00 and before 14:00, for each source airport (origin). Plot top 5 of them.


```python
flights_per_origin = flights.groupBy(lambda x: x[1]).mapValues(lambda val: len(val))

# take top 5 source airports
top5_source_airport = flights_per_origin.takeOrdered(5, key=lambda x: -x[1])

%matplotlib inline
import matplotlib.pyplot as plt

# extract the number of flights which will be used as y-values
# This is called list comprehension
num_flights = [ x[1] for x in top5_source_airport]

# create `virtual indexes for carriers which will be used as x-values`
airport_indexes = range(0, len(top5_source_airport))

# plot
plt.bar(airport_indexes, num_flights, align="center")

# extract the carriers' names
airport_names = [ x[0] for x in top5_source_airport]

# put x-labels for the plot
plt.xticks(airport_indexes, airport_names)
plt.show()
```


![png](output_148_0.png)


# 4. Spark SQL and DataFrames

One of the main modules that we suggest to use when analyzing data with Spark is `Spark SQL` - a module for structured data processing. Unlike the basic Spark RDD API, the interfaces provided by Spark SQL provide Spark with more information about the structure of both the data and the computation being performed. Internally, this extra information is used to perform extra optimizations. There are several ways to interact with Spark SQL including SQL, the DataFrames API and the Datasets API. In this course, we mainly focus on `DataFrame API`. 

A `DataFrame` is a distributed collection of data organized into named columns. It is based on the data frame concept in R language or in Pandas. So, it is similar to a database table in a relational database.

`DataFrames` can be constructed from a wide array of sources such as: structured data files, tables in Hive, external databases, or existing RDDs.

During the lectures in this course, we will mainly work with CSV data file. So, in the next sections, we only focus on constructing dataframes from structured data file directly and from existing RDD.

## 4.1. Constructing directly from structured data file

To construct DataFrame from a structured file directly, the file type must be supported such as csv, json, avro...
Among these types, csv type is one of the most popular in data analytic. A DataFrame is constructed from csv files by using package `spark-csv` from Databrick.


```python
from pyspark.sql import SQLContext
from pyspark.sql.types import *

sqlContext = SQLContext(sc)

df = sqlContext.read.load('/datasets/airline/1994.csv', 
                          format='com.databricks.spark.csv', 
                          header='true', 
                          inferSchema='true',
                          nullValue='NA'
                        )
```

With function `load` and assigning value `com.databricks.spark.csv` for parameter `format`, we ask SqlContext to use the parser from DataBrick's package. Besides, we can specify whether the file has header, or ask the parser to guess the data type of columns automatically. The parsed data types is viewed by function `printSchema`.


```python
# print(df.dtypes)
df.printSchema()
```

    root
     |-- Year: integer (nullable = true)
     |-- Month: integer (nullable = true)
     |-- DayofMonth: integer (nullable = true)
     |-- DayOfWeek: integer (nullable = true)
     |-- DepTime: string (nullable = true)
     |-- CRSDepTime: integer (nullable = true)
     |-- ArrTime: string (nullable = true)
     |-- CRSArrTime: integer (nullable = true)
     |-- UniqueCarrier: string (nullable = true)
     |-- FlightNum: integer (nullable = true)
     |-- TailNum: string (nullable = true)
     |-- ActualElapsedTime: string (nullable = true)
     |-- CRSElapsedTime: integer (nullable = true)
     |-- AirTime: string (nullable = true)
     |-- ArrDelay: string (nullable = true)
     |-- DepDelay: string (nullable = true)
     |-- Origin: string (nullable = true)
     |-- Dest: string (nullable = true)
     |-- Distance: string (nullable = true)
     |-- TaxiIn: string (nullable = true)
     |-- TaxiOut: string (nullable = true)
     |-- Cancelled: integer (nullable = true)
     |-- CancellationCode: string (nullable = true)
     |-- Diverted: integer (nullable = true)
     |-- CarrierDelay: string (nullable = true)
     |-- WeatherDelay: string (nullable = true)
     |-- NASDelay: string (nullable = true)
     |-- SecurityDelay: string (nullable = true)
     |-- LateAircraftDelay: string (nullable = true)
    


Actually, in this case, the inferred data types are not as expected. For example, we expect that `CRSDepTime` has interger type. Fortunately, the type and the name of each column can be changed by using function `withColumn` and `withColumnRename` respectively. Besides, we can also view the basic statistic of numerical columns via function `describe` (similar to Pandas).


```python

df = (df
          # change type of column CRSDepTime by casting its values to interger type
          .withColumn('CRSDepTime', df.CRSDepTime.cast('int'))
      
          # rename the column
          .withColumnRenamed('CRSDepTime', 'scheduled_departure_time')
    )

# print schema of the current data
df.printSchema()

# run jobs to calculate basic statistic information and show it
df.describe().show()
```

    root
     |-- Year: integer (nullable = true)
     |-- Month: integer (nullable = true)
     |-- DayofMonth: integer (nullable = true)
     |-- DayOfWeek: integer (nullable = true)
     |-- DepTime: string (nullable = true)
     |-- scheduled_departure_time: integer (nullable = true)
     |-- ArrTime: string (nullable = true)
     |-- CRSArrTime: integer (nullable = true)
     |-- UniqueCarrier: string (nullable = true)
     |-- FlightNum: integer (nullable = true)
     |-- TailNum: string (nullable = true)
     |-- ActualElapsedTime: string (nullable = true)
     |-- CRSElapsedTime: integer (nullable = true)
     |-- AirTime: string (nullable = true)
     |-- ArrDelay: string (nullable = true)
     |-- DepDelay: string (nullable = true)
     |-- Origin: string (nullable = true)
     |-- Dest: string (nullable = true)
     |-- Distance: string (nullable = true)
     |-- TaxiIn: string (nullable = true)
     |-- TaxiOut: string (nullable = true)
     |-- Cancelled: integer (nullable = true)
     |-- CancellationCode: string (nullable = true)
     |-- Diverted: integer (nullable = true)
     |-- CarrierDelay: string (nullable = true)
     |-- WeatherDelay: string (nullable = true)
     |-- NASDelay: string (nullable = true)
     |-- SecurityDelay: string (nullable = true)
     |-- LateAircraftDelay: string (nullable = true)
    
    +-------+-------+------------------+------------------+------------------+------------------------+-----------------+-----------------+------------------+--------------------+--------------------+
    |summary|   Year|             Month|        DayofMonth|         DayOfWeek|scheduled_departure_time|       CRSArrTime|        FlightNum|    CRSElapsedTime|           Cancelled|            Diverted|
    +-------+-------+------------------+------------------+------------------+------------------------+-----------------+-----------------+------------------+--------------------+--------------------+
    |  count|5180048|           5180048|           5180048|           5180048|                 5180048|          5180048|          5180048|           5180048|             5180048|             5180048|
    |   mean| 1994.0| 6.579943853802127|15.723382486031017|3.9516016067804776|      1338.6794068317513|1495.874874904634|920.2637743897354|114.82526995888841|0.012884050495284986|0.002337043981059635|
    | stddev|    0.0|3.4367200422479933|  8.78827151205811|1.9902382476128326|       463.6174563974092|472.6701038621056|608.1342356931183| 64.20405571869162| 0.11277433989219326|0.048286459867029265|
    |    min|   1994|                 1|                 1|                 1|                       1|                1|                1|                -5|                   0|                   0|
    |    max|   1994|                12|                31|                 7|                    2400|             2400|             3599|               578|                   1|                   1|
    +-------+-------+------------------+------------------+------------------+------------------------+-----------------+-----------------+------------------+--------------------+--------------------+
    


## 4.2. Constructing from an existing RDD
Another way to construct DataFrame is using data from an existing RDD. The main advantage of this approach is that it does not need a third party library. However, with this method, we have to remove the header ourself and provide a clear schema. 


```python
from pyspark.sql import SQLContext
from pyspark.sql.types import *

sqlContext = SQLContext(sc)

data = sc.textFile('/datasets/airline/1994.csv')

# extract the header
header = data.first()

# replace invalid data with NULL and remove header
cleaned_data = (
        data
    
        # filter out the header
        .filter(lambda line: line != header)
    
         # remove the 'missing data' by empty value
        .map(lambda l: l.replace(',NA', ','))
    )

airline_data_schema = StructType([ \
    #StructField( name, dataType, nullable)
    StructField("year",                     IntegerType(), True), \
    StructField("month",                    IntegerType(), True), \
    StructField("day_of_month",             IntegerType(), True), \
    StructField("day_of_week",              IntegerType(), True), \
    StructField("departure_time",           IntegerType(), True), \
    StructField("scheduled_departure_time", IntegerType(), True), \
    StructField("arrival_time",             IntegerType(), True), \
    StructField("scheduled_arrival_time",   IntegerType(), True), \
    StructField("carrier",                  StringType(),  True), \
    StructField("flight_number",            StringType(),  True), \
    StructField("tail_number",              StringType(), True), \
    StructField("actual_elapsed_time",      IntegerType(), True), \
    StructField("scheduled_elapsed_time",   IntegerType(), True), \
    StructField("air_time",                 IntegerType(), True), \
    StructField("arrival_delay",            IntegerType(), True), \
    StructField("departure_delay",          IntegerType(), True), \
    StructField("src_airport",              StringType(),  True), \
    StructField("dest_airport",             StringType(),  True), \
    StructField("distance",                 IntegerType(), True), \
    StructField("taxi_in_time",             IntegerType(), True), \
    StructField("taxi_out_time",            IntegerType(), True), \
    StructField("cancelled",                StringType(),  True), \
    StructField("cancellation_code",        StringType(),  True), \
    StructField("diverted",                 StringType(),  True), \
    StructField("carrier_delay",            IntegerType(), True), \
    StructField("weather_delay",            IntegerType(), True), \
    StructField("nas_delay",                IntegerType(), True), \
    StructField("security_delay",           IntegerType(), True), \
    StructField("late_aircraft_delay",      IntegerType(), True)\
])
```


```python
# convert each line into a tuple of features (columns) with the corresponding data type
cleaned_data_to_columns = (
    cleaned_data.map(lambda l: l.split(","))
    .map(lambda cols: 
         (
            int(cols[0])  if cols[0] else None,
            int(cols[1])  if cols[1] else None,
            int(cols[2])  if cols[2] else None,
            int(cols[3])  if cols[3] else None,
            int(cols[4])  if cols[4] else None,
            int(cols[5])  if cols[5] else None,
            int(cols[6])  if cols[6] else None,
            int(cols[7])  if cols[7] else None,
            cols[8]       if cols[8] else None,
            cols[9]       if cols[9] else None,
            cols[10]      if cols[10] else None,
            int(cols[11]) if cols[11] else None,
            int(cols[12]) if cols[12] else None,
            int(cols[13]) if cols[13] else None,
            int(cols[14]) if cols[14] else None,
            int(cols[15]) if cols[15] else None,
            cols[16]      if cols[16] else None,
            cols[17]      if cols[17] else None,
            int(cols[18]) if cols[18] else None,
            int(cols[19]) if cols[19] else None,
            int(cols[20]) if cols[20] else None,
            cols[21]      if cols[21] else None,
            cols[22]      if cols[22] else None,
            cols[23]      if cols[23] else None,
            int(cols[24]) if cols[24] else None,
            int(cols[25]) if cols[25] else None,
            int(cols[26]) if cols[26] else None,
            int(cols[27]) if cols[27] else None,
            int(cols[28]) if cols[28] else None
         ))             
)
    
# create dataframe
df = sqlContext.createDataFrame(cleaned_data_to_columns, airline_data_schema)\
    .select(['year', 'month', 'day_of_month', 'day_of_week',
            'scheduled_departure_time','scheduled_arrival_time',
            'arrival_delay', 'distance', 
            'src_airport', 'dest_airport', 'carrier'])\
    .cache()
```


```python
print(df.dtypes)
df.describe().show()
```

    [('year', 'int'), ('month', 'int'), ('day_of_month', 'int'), ('day_of_week', 'int'), ('scheduled_departure_time', 'int'), ('scheduled_arrival_time', 'int'), ('arrival_delay', 'int'), ('distance', 'int'), ('src_airport', 'string'), ('dest_airport', 'string'), ('carrier', 'string')]
    +-------+-------+------------------+------------------+------------------+------------------------+----------------------+------------------+-----------------+
    |summary|   year|             month|      day_of_month|       day_of_week|scheduled_departure_time|scheduled_arrival_time|     arrival_delay|         distance|
    +-------+-------+------------------+------------------+------------------+------------------------+----------------------+------------------+-----------------+
    |  count|5180048|           5180048|           5180048|           5180048|                 5180048|               5180048|           5101202|          5157099|
    |   mean| 1994.0| 6.579943853802127|15.723382486031017|3.9516016067804776|      1338.6794068317513|     1495.874874904634| 5.662489742613603|670.7402911985982|
    | stddev|    0.0|3.4367200422479933|  8.78827151205811|1.9902382476128326|       463.6174563974092|     472.6701038621056|23.618153725857074|522.2696182530875|
    |    min|   1994|                 1|                 1|                 1|                       1|                     1|              -115|               11|
    |    max|   1994|                12|                31|                 7|                    2400|                  2400|              1313|             4502|
    +-------+-------+------------------+------------------+------------------+------------------------+----------------------+------------------+-----------------+
    


## 4.3. Night flight example
Using the contructed DataFrame, we can answer the questions about night flights in the previous section:

- How many night flights in the data ?
- How many night flights of each unique carrier ?


```python
df[df.scheduled_departure_time > 1800].count()
```




    1078203




```python
df[df.scheduled_departure_time > 1800].groupBy(df.carrier).count().orderBy('count', ascending=0).collect()
```




    [Row(carrier='DL', count=208224),
     Row(carrier='US', count=170397),
     Row(carrier='AA', count=142832),
     Row(carrier='WN', count=124024),
     Row(carrier='UA', count=113640),
     Row(carrier='NW', count=100453),
     Row(carrier='CO', count=94501),
     Row(carrier='TW', count=54771),
     Row(carrier='HP', count=44351),
     Row(carrier='AS', count=25010)]



![](https://farm2.staticflickr.com/1604/24934700445_833f0a5649_t.jpg)

<div style='border-radius: 15px; background: rgb(224,239,130);padding:10px;'/>
## Question 6


### Question 6.1
Using Spark SQL, calculate how many flights have the scheduled departure time after 09:00 and before 14:00.


```python
flights = df[(df.scheduled_departure_time > 900) & (df.scheduled_departure_time < 1400)]
flights.count()
```




    1625828



<div style='border-radius: 15px; background: rgb(224,239,130);padding:10px;'/>
### Question 6.2
Calculate the number flights that have the scheduled departure time after 09:00 and before 14:00, for each source airport (origin). Plot top 5 of them.


```python
import pandas as pd
top5_source_airport = flights.groupBy(flights.src_airport).count().orderBy('count', ascending=0).take(5)

pdf = pd.DataFrame(data=top5_source_airport)

print(pdf)

%matplotlib inline
import matplotlib.pyplot as plt


# create `virtual indexes for carriers which will be used as x-values`
airport_indexes = range(0, len(top5_source_airport))

# plot
plt.bar(airport_indexes, pdf[1], align="center")

# put x-labels for the plot
plt.xticks(airport_indexes, pdf[0])
plt.show()
```

         0      1
    0  ORD  92252
    1  DFW  81338
    2  ATL  75077
    3  STL  54937
    4  DTW  51726



![png](output_165_1.png)


# Summary

In this lecture, we gained familiarity with the Jupyter Notebook environment, the Python programming language and its modules. In particular, we covered the Python syntax, Numpy - the core library for scientific computing, Matplotlib - a module to plot graphs, Pandas - a data analysis module. Besides, we started to gain practical experience with PySpark and SparkSQL, using as an example a dataset concerning US flights.

# References
This notebook is inspired from:

- [Python Numpy tutorial](http://cs231n.github.io/python-numpy-tutorial/)
