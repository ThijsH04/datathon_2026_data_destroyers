# import boto3  

# import pandas as pd
# import argparse






# # print(response["output"]["message"]["content"][0]["text"])


# # ── Your label space ──────────────────────────────────────────────────────────

# ALLOWED_TYPES = [
#     "Möblierte Wohnung", "Gewerbeobjekt", "Tiefgarage", "Wohnung", "Einzelgarage",
#     "Parkplatz, Garage", "Bastelraum", "Maisonette", "Bauernhaus", "Villa",
#     "Dachwohnung", "Einzelzimmer", "Parkplatz", "Reihenhaus", "Studio", "Haus",
#     "Diverses", "Attika", "Loft", "WG-Zimmer", "Terrassenwohnung",
#     "Doppeleinfamilienhaus", "Grundstück", "Mehrfamilienhaus", "Attikawohnung",
#     "Ladenfläche", "Gewerbe", "Büro", "Lager", "Maisonette / Duplex", "Praxis",
#     "Restaurant", "Hobbyraum", "Moto Hallenplatz", "Offener Parkplatz",
#     "Möbliertes Wohnobjekt", "Doppelgarage", "Einfamilienhaus", "Kellerabteil",
#     "Ferienimmobilie", "Terrassenhaus",
# ]

# # ── Your row logic (placeholder version of classifier) ────────────────────────

# def classify_row(row, client):
#     title = str(row.get("title") or row.get("header") or "")
#     desc = str(
#         row.get("object_description")
#         or row.get("description")
#         or row.get("ad_description")
#         or row.get("remarks")
#         or ""
#     )

#     text = (title + " " + desc).strip()

#     response = client.converse( 
#         modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0", 
#         messages=[ 
#             { 
#                 "role": "user", 
#                 "content": [{"text": "Classify the following real estate listing into one of the following categories And only PICK one. NO EXTRA TEXT: " + ", ".join(ALLOWED_TYPES) + ".\n\nListing:\n" + text}]
#             } 
#         ] 
#     )  

#     # default fallback (same as your original safety behavior)
#     return response["output"]["message"]["content"][0]["text"].strip() if response["output"]["message"]["content"] else "Unknown"

# # ── Main loop ────────────────────────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", required=True)
#     parser.add_argument("--output", required=True)
#     args = parser.parse_args()

#     df = pd.read_csv(args.input).fillna("")

#     results = []

#     client = boto3.client("bedrock-runtime", region_name="us-west-2")  

    

#     for i, row in df.iterrows():
#         print(f"Row {i+1}/{len(df)}")
#         label = classify_row(row, client)
#         print(label)
#         results.append(label)

#     df["object_type"] = results
#     df.to_csv(args.output, index=False)

#     print("Done →", args.output)

# if __name__ == "__main__":
#     main()
import boto3
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Label space ───────────────────────────────────────────────────────────────

ALLOWED_TYPES = [
    "Möblierte Wohnung", "Gewerbeobjekt", "Tiefgarage", "Wohnung", "Einzelgarage",
    "Parkplatz, Garage", "Bastelraum", "Maisonette", "Bauernhaus", "Villa",
    "Dachwohnung", "Einzelzimmer", "Parkplatz", "Reihenhaus", "Studio", "Haus",
    "Diverses", "Attika", "Loft", "WG-Zimmer", "Terrassenwohnung",
    "Doppeleinfamilienhaus", "Grundstück", "Mehrfamilienhaus", "Attikawohnung",
    "Ladenfläche", "Gewerbe", "Büro", "Lager", "Maisonette / Duplex", "Praxis",
    "Restaurant", "Hobbyraum", "Moto Hallenplatz", "Offener Parkplatz",
    "Möbliertes Wohnobjekt", "Doppelgarage", "Einfamilienhaus", "Kellerabteil",
    "Ferienimmobilie", "Terrassenhaus",
]

# ── Classification logic ──────────────────────────────────────────────────────

def classify_row(i, row, client):
    title = str(row.get("title") or row.get("header") or "")
    desc = str(
        row.get("object_description")
        or row.get("description")
        or row.get("ad_description")
        or row.get("remarks")
        or ""
    )

    # ⚡ speed optimization: limit input size
    text = (title + " " + desc).strip()[:1000]

    prompt = (
        "Classify the following real estate listing into exactly ONE category "
        "from the list below. Return ONLY the category name, nothing else.\n\n"
        f"Allowed categories:\n{', '.join(ALLOWED_TYPES)}\n\n"
        f"Listing:\n{text}"
    )

    try:
        response = client.converse(
            modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ]
        )

        return i, response["output"]["message"]["content"][0]["text"].strip()

    except Exception as e:
        return i, f"Error: {str(e)}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.input).fillna("")
    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    results = [None] * len(df)

    print(f"Processing {len(df)} rows with {args.workers} workers...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(classify_row, i, row, client)
            for i, row in df.iterrows()
        ]

        for future in as_completed(futures):
            i, label = future.result()
            results[i] = label
            print(f"Row {i+1}/{len(df)} → {label}")

    df["object_type"] = results
    df.to_csv(args.output, index=False)

    print("Done →", args.output)


if __name__ == "__main__":
    main()