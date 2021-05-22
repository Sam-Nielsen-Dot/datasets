import csv

#with open('tweets.csv', 'r') as f:

f = open('tweets.csv', 'r', encoding='utf-8')
reader = csv.DictReader(f, delimiter=",")

fd = open('trumpinsults.txt','a', encoding='utf-8')
fd.write("hello")

for row in reader:
    fd.write(f"{row['tweet']}\n")