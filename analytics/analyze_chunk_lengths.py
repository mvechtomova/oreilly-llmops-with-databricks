import json

import matplotlib.pyplot as plt

# Read the JSON file
with open("doc.json", "r") as f:
    data = json.load(f)

# Extract text chunk lengths
text_lengths = []
for element in data["document"]["elements"]:
    if element["type"] == "text":
        text_lengths.append(len(element["content"]))

# Create histogram
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(text_lengths, bins=30, edgecolor="black", alpha=0.7)
plt.xlabel("Chunk Length (characters)")
plt.ylabel("Frequency")
plt.title("Distribution of Text Chunk Lengths")
plt.grid(True, alpha=0.3)

# Create box plot
plt.subplot(1, 2, 2)
plt.boxplot(text_lengths, vert=True)
plt.ylabel("Chunk Length (characters)")
plt.title("Text Chunk Length Box Plot")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("text_chunk_distribution.png", dpi=300, bbox_inches="tight")
print("Graph saved as text_chunk_distribution.png")

# Print statistics
print(f"\nStatistics:")
print(f"Total text chunks: {len(text_lengths)}")
print(f"Min length: {min(text_lengths)}")
print(f"Max length: {max(text_lengths)}")
print(f"Mean length: {sum(text_lengths)/len(text_lengths):.2f}")
print(f"Median length: {sorted(text_lengths)[len(text_lengths)//2]}")

# Print distribution by bins
bins = [
    (0, 100),
    (100, 200),
    (200, 300),
    (300, 500),
    (500, 1000),
    (1000, float("inf")),
]
print(f"\nDistribution:")
for start, end in bins:
    count = sum(1 for x in text_lengths if start <= x < end)
    pct = (count / len(text_lengths)) * 100
    label = f"{start}-{end}" if end != float("inf") else f"{start}+"
    print(f"{label} chars: {count} ({pct:.1f}%)")
