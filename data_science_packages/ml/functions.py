__author__ = "Brett Allen (brettallen777@gmail.com)"

"""
Utility functions for re-using common processes across various scripts.
"""
import re
import subprocess
import os
import json
import boto3
from difflib import SequenceMatcher
from typing import List, Callable, Union
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

def is_s3_path(path: str) -> bool:
    """
    Determine whether a path string is a path to an s3 resource or not.

    Args:
        path (str): Path to file.

    Returns:
        bool: Whether the provided path is to an s3 resource. Returns False if path is None.
    """
    if not path:
        return False
    return re.match(r"^s3.?:\/\/.*", path)

def decompose_s3_path(s3_path: str) -> tuple:
    """
    Extract bucket and prefix from correctly formed s3 path.

    Args:
        s3_path (str): S3 path string.

    Returns:
        tuple: Extracted bucket and prefix from orignial s3 path respectively.

    Examples:
    --------
    >>> decompose_s3_path("s3://bucket-name/path/to/file/test.csv")
    ("bucket-name", "path/to/file/test.csv")
    """
    if not s3_path:
        return None, None
    
    m = re.match(r"^s3[a-z]?:\/\/(?P<bucket>.*?)\/(?P<prefix>.*?)$", s3_path)
    if not m:
        return None, None
    
    return m.group("bucket"), m.group("prefix")

def save_json(json_obj: Union[list, dict], path: str):
    """
    Save JSON object locally or to s3.

    Args:
        json_obj (Union[list, dict]): JSON object to save.
        path (str): Path to save JSON data which can be local or s3.
    """
    if not json_obj:
        print("No json data to save.")
        return
    if not path:
        print("No path specified.")
        return
    
    if is_s3_path(path):
        bucket, key = decompose_s3_path(path)
        s3_client = boto3.client("s3")
        s3_client.put_object(Body=json.dumps(json_obj), Bucket=bucket, Key=key)
    else:
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(json_obj, f)

def load_json(path: str) -> Union[list, dict]:
    """
    Load JSON data from local path or s3.

    Args:
        path (str): Path to JSON file which can be local or s3.

    Returns:
        Union[list, dict]: Data loaded from JSON file.
    """
    if not path:
        print("No path specified.")
        return
    
    json_obj = None
    if is_s3_path(path):
        bucket, key = decompose_s3_path(path)
        s3_client = boto3.client("s3")
        response = s3_client.get_object(Bucket=bucket, Key=key)
        json_content = response["Body"].read().decode("utf-8")
        json_obj = json.loads(json_content)
    else:
        with open(path, "r", encoding="utf-8") as f:
            json_obj = json.load(f)
    return json_obj

def get_commit_id(filepath: str) -> str:
    """
    Get the commit ID associated with a given file path representing the code state in which
    the file was last modified.

    Args:
        filepath (str): Path to file to get commit id for.

    Returns:
        str: Commit id hash string. Returns None if any errors when attempting to get commit id for the specified file.
    """
    if not filepath:
        return None

    if not os.path.exists(filepath):
        return None

    command = ["git", "log", '--format="%H"', "-n", "1", "--", filepath]
    commit_id = None
    try:
        commit_id = str(subprocess.check_output(command), encoding="utf-8").replace("\"", "").strip()
    except subprocess.CalledProcessError as e:
        print(f"Failed to get commit id from file path, '{filepath}'")
        print(str(e))

    return commit_id

def coalesce(vals: list, enforced_type=str):
    """
    Obtain the first non-null value in a list of values and optionally enforce
    a datatype.
    
    Args:
        vals (list): List of values to coalesce.
        enforced_type (type, optional): Data type to enforce as an additional check for non-null. Defaults to str.
        
    Returns:
        any: First non-null value in list of values. Datatype depends on either the type of data that is in the values list or the enforced type.
    """
    v = None
    if vals:
        for val in vals:
            if val is not None:
                valid = True
                if enforced_type:
                    valid = isinstance(val, enforced_type)
                if valid:
                    v = val
                    break
    return v

def cosine_similarity(doc1: str, doc2: str) -> float:
    # Source: https://www.geeksforgeeks.org/python-measure-similarity-between-two-sentences-using-cosine-similarity/#
    # tokenization 
    X_list = word_tokenize(doc1)  
    Y_list = word_tokenize(doc2) 
    
    # sw contains the list of stopwords 
    sw = stopwords.words('english')  
    l1 =[];l2 =[] 
    
    # remove stop words from the string 
    X_set = {w for w in X_list if not w in sw}  
    Y_set = {w for w in Y_list if not w in sw} 
    
    # form a set containing keywords of both strings  
    rvector = X_set.union(Y_set)  
    for w in rvector: 
        if w in X_set: l1.append(1) # create a vector 
        else: l1.append(0) 
        if w in Y_set: l2.append(1) 
        else: l2.append(0) 
    c = 0
    
    # cosine formula  
    for i in range(len(rvector)): 
        c+= l1[i]*l2[i]
    d = float((sum(l1)*sum(l2))**0.5)
    similarity = 0 if d == 0 else c / d
    return similarity

def similar(doc_a: str, doc_b: str, fn: Callable=None) -> float:
    """
    Compute a similarity score (float in [0,1]) between two strings. 
    Note that this is 1 if the sequences are identical, and 0 if
    they have nothing in common.

    Uses the Ratcliff/Obershelp string matching algorithm which calculates 
    the similarity metric between two strings as: Twice the number of matching (overlapping) 
    characters between the two strings divided by the total number of characters in the two strings.

    Where T is the total number of elements in both sequences, and M is the number of matches, this is 2.0*M / T.
    
    Args:
        doc_a (str): First document.
        doc_b (str): Second document.
        fn (Callable, optional): Optional custom similarity function to use in place of difflib Ratcliff/Obershelp 
                                 string matching algorithm. Function must take two strings 
                                 in as arguments representing document 1 and document 2 respectively. For example,
                                 the cosine_similarity function as part of this library can be used.
        
    Returns:
        float: Similarity score between first and second documents between 0 and 1.
    """
    if fn is not None:
        return fn(doc_a, doc_b)
    return SequenceMatcher(None, doc_a, doc_b).ratio()

def most_similar(doc: str, docs: list, case_sensitive: bool=True, top_n: int=1, fn: Callable=None, sort: bool=True) -> List[tuple]:
    """
    Identify the most similar document(s) compared to each document within a list
    of existing documents. 
    
    Args:
        doc (str): Document to compare with.
        docs (list): List of existing documents to compare to the provided document.
        case_sensitive (bool, optional): Whether the document and list of documents should be case sensitive. Default True.
        top_n (int, optional): Top N scores to return along with respective similar document.
        fn (Callable, optional): Optional custom similarity function to use in place of difflib Ratcliff/Obershelp 
                                 string matching algorithm. Function must take two strings 
                                 in as arguments representing document 1 and document 2 respectively. For example,
                                 the cosine_similarity function as part of this library can be used.
        sort (bool, optional): Whether to sort the resulting list of most similar matches. Defaults to True and sorts in descending order (highest score first).
        
    Returns:
        List[tuple]: List of tuples where each tuple contains the document and similarity score (value between 0 and 1). Ordered by most similar document.
    """
    if not case_sensitive:
        doc = doc.upper()

    scores = [ (sent, similar(doc, sent if case_sensitive else sent.upper(), fn=fn)) for sent in docs ]

    if sort:
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

    if top_n < 0:
        top_n = 1

    return scores[:top_n]

