"""
Generates a synthetic but realistic product-review dataset so the dashboard
works out-of-the-box without needing an external download.
Run this once: python generate_sample_data.py
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

products = ["Aurora Wireless Earbuds", "Nimbus Smart Watch", "Zephyr Bluetooth Speaker",
            "Pulse Fitness Tracker", "Halo Desk Lamp", "Comet Phone Case"]

positive_templates = [
    "Absolutely love the {product}! The battery life is amazing and it works perfectly.",
    "Best purchase this year. The {product} exceeded my expectations in every way.",
    "The sound quality on the {product} is incredible, worth every penny.",
    "Fast shipping and the {product} feels premium. Highly recommend!",
    "The {product} has great battery life and the app integration is seamless.",
    "I'm impressed with the build quality of the {product}. Feels sturdy and well made.",
    "Customer service was excellent when I had a question about the {product}.",
    "The {product} is comfortable to use for hours, no complaints at all.",
]

negative_templates = [
    "The {product} stopped working after two weeks. Very disappointed.",
    "Battery life on the {product} is terrible, dies within a few hours.",
    "The {product} feels cheap and the build quality is poor for the price.",
    "Had a bad experience with customer support regarding my {product}.",
    "The {product} disconnects randomly, connectivity issues are frustrating.",
    "Not worth the money, the {product} broke within a month.",
    "The app for the {product} is buggy and crashes constantly.",
    "Shipping was delayed and the {product} arrived with a scratch.",
]

neutral_templates = [
    "The {product} is okay, does what it says but nothing special.",
    "Average experience with the {product}, meets basic expectations.",
    "The {product} works fine, though the design could be improved.",
    "It's a decent {product} for the price, no major complaints.",
    "The {product} arrived on time, still testing it out.",
    "Mixed feelings about the {product}, some features are good, others not so much.",
]

def make_reviews(n=600):
    rows = []
    start_date = datetime(2025, 1, 1)
    for i in range(n):
        product = random.choice(products)
        sentiment_bucket = random.choices(
            ["positive", "negative", "neutral"], weights=[0.5, 0.3, 0.2]
        )[0]
        if sentiment_bucket == "positive":
            template = random.choice(positive_templates)
            rating = random.choice([4, 5])
        elif sentiment_bucket == "negative":
            template = random.choice(negative_templates)
            rating = random.choice([1, 2])
        else:
            template = random.choice(neutral_templates)
            rating = 3

        text = template.format(product=product)
        date = start_date + timedelta(days=random.randint(0, 270))
        rows.append({
            "review_id": i + 1,
            "date": date.strftime("%Y-%m-%d"),
            "product": product,
            "rating": rating,
            "review_text": text,
            "true_label": sentiment_bucket,  # ground-truth bucket used only to generate data
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = make_reviews(600)
    df.to_csv("data/reviews.csv", index=False)
    print(f"Generated {len(df)} reviews -> data/reviews.csv")
    print(df["true_label"].value_counts())
