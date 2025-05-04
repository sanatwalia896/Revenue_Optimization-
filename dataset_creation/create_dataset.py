import pandas as pd
import random
from datetime import datetime, timedelta

# Sample data pools
user_segments = ["Casual", "Hardcore", "Whales"]
items = [
    "Coin Pack - 1000",
    "Coin Pack - 5000",
    "Coin Pack - 10000",
    "Character Skin",
    "Premium Weapon",
    "Season Pass",
    "Power-Up Bundle",
    "Ultra Bundle",
    "Limited-Time Offer Pack",
    "Lifetime Membership",
]
purchase_types = ["One-time", "Subscription"]
promo_applied = ["Yes", "No"]


# Generate realistic random data
def generate_dataset(n_rows=5000):
    data = []
    base_date = datetime.strptime("15/01/2023", "%d/%m/%Y")

    for _ in range(n_rows):
        date_of_purchase = base_date + timedelta(days=random.randint(0, 30))
        date_of_install = date_of_purchase - timedelta(days=random.randint(1, 30))
        user_segment = random.choices(user_segments, weights=[0.6, 0.3, 0.1])[0]
        item = random.choice(items)
        session_minutes = random.randint(5, 180)
        session_time = f"{session_minutes // 60:02d}h {session_minutes % 60:02d}m"
        level = random.randint(1, 40)
        prior_purchases = (
            random.randint(0, 15) if user_segment != "Casual" else random.randint(0, 5)
        )
        p_type = (
            "Subscription"
            if "Membership" in item or "Season Pass" in item
            else "One-time"
        )
        promo = random.choices(promo_applied, weights=[0.3, 0.7])[0]

        # Price generation logic based on item
        price_map = {
            "Coin Pack - 1000": (79, 99),
            "Coin Pack - 5000": (199, 249),
            "Coin Pack - 10000": (349, 499),
            "Character Skin": (99, 199),
            "Premium Weapon": (199, 299),
            "Season Pass": (399, 599),
            "Power-Up Bundle": (149, 249),
            "Ultra Bundle": (499, 699),
            "Limited-Time Offer Pack": (49, 99),
            "Lifetime Membership": (799, 999),
        }
        price = random.randint(*price_map[item])

        data.append(
            [
                date_of_purchase.strftime("%d/%m/%Y"),
                date_of_install.strftime("%d/%m/%Y"),
                user_segment,
                item,
                session_time,
                level,
                prior_purchases,
                p_type,
                promo,
                price,
            ]
        )

    columns = [
        "Date of Purchase",
        "Date of Install",
        "User Segment",
        "Item-Purchased",
        "Session Time",
        "Level Reached",
        "Prior Purchases",
        "Purchase Type",
        "Promo Applied",
        "Price (₹)",
    ]

    return pd.DataFrame(data, columns=columns)


# Generate dataset
iap_df = generate_dataset(5000)

# Save to CSV
csv_path = "/Users/sanatwalia/Desktop/Assignments_applications/Revenue_Optimization-/dataset/mobile_game_iap_dataset.csv"
iap_df.to_csv(csv_path, index=False)
csv_path
