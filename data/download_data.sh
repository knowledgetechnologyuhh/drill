#!/bin/bash

# Download AGNews
DIR="ag_news_csv"
if [ -d "$DIR" ]; then
  echo "Directory '$DIR' already exists. If you want the latest version of the dataset, delete the directory and rerun the script."
else
  wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms" -O ag_news_csv.tar.gz && rm -rf /tmp/cookies.txt
  tar xvfz ag_news_csv.tar.gz
  rm ag_news_csv.tar.gz
fi

# Download AmazonReviews
DIR="amazon_review_full_csv"
if [ -d "$DIR" ]; then
  echo "Directory '$DIR' already exists. If you want the latest version of the dataset, delete the directory and rerun the script."
else
  wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA" -O amazon_review_full_csv.tar.gz && rm -rf /tmp/cookies.txt
  tar xvfz amazon_review_full_csv.tar.gz
  rm amazon_review_full_csv.tar.gz
fi

# Download DBPedia
DIR="dbpedia_csv"
if [ -d "$DIR" ]; then
  echo "Directory '$DIR' already exists. If you want the latest version of the dataset, delete the directory and rerun the script."
else
  wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k" -O dbpedia_csv.tar.gz && rm -rf /tmp/cookies.txt
tar xvfz dbpedia_csv.tar.gz
rm dbpedia_csv.tar.gz
fi

# Download YahooAnswers
DIR="yahoo_answers_csv"
if [ -d "$DIR" ]; then
  echo "Directory '$DIR' already exists. If you want the latest version of the dataset, delete the directory and rerun the script."
else
  wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU" -O yahoo_answers_csv.tar.gz && rm -rf /tmp/cookies.txt
  tar xvfz yahoo_answers_csv.tar.gz
  rm yahoo_answers_csv.tar.gz
fi

# Download YelpReview
DIR="yelp_review_full_csv"
if [ -d "$DIR" ]; then
  echo "Directory '$DIR' already exists. If you want the latest version of the dataset, delete the directory and rerun the script."
else
  wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0" -O yelp_review_csv.tar.gz && rm -rf /tmp/cookies.txt
  tar xvfz yelp_review_csv.tar.gz
  rm yelp_review_csv.tar.gz
fi