# Note: This example code is written for Python 3.6+!
import json
from tqdm import tqdm
from pathlib import Path

reviews_data = Path("D:/Dataset") / "review.json"
fasttext_data = Path("D:/Dataset/fasttext_dataset.txt")

# with reviews_data.open() as input, fasttext_data.open("w") as output:
with open("./data/yelp/review.json", "r", encoding='utf-8') as input:
    with open("./data/yelp/fasttext_dataset.txt", "w", encoding='utf-8') as output:
        input = input.readlines()
        for line in tqdm(input):
            review_data = json.loads(line)

            rating = review_data['stars']
            text = review_data['text'].replace("\n", " ")

            fasttext_line = "__label__{} {}".format(rating, text)

            output.write(fasttext_line + "\n")