import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load the data
df = pd.read_csv('human_ref_eval_by_slide.csv')

# Create bar plot comparing average performance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Average concept performance
concept_means = df[df['type'] == 'concepts'].groupby('model')['concept_jaccard'].mean().sort_values()
concept_stds = df[df['type'] == 'concepts'].groupby('model')['concept_jaccard'].std()

y_pos = np.arange(len(concept_means))
bars1 = ax1.barh(y_pos, concept_means, xerr=concept_stds, align='center', alpha=0.7, capsize=5, color='skyblue')
ax1.set_yticks(y_pos)
ax1.set_yticklabels([model.split('/')[-1] for model in concept_means.index])  # Shorten model names
ax1.set_xlabel('Jaccard Similarity', fontsize=12)
ax1.set_title('A) Diagnostic Concept Similarity Against Transcript Reference', fontweight='bold', fontsize=14)
ax1.grid(True, alpha=0.3, axis='x')
ax1.set_xlim(0, 1)

# Add value labels on bars
for i, bar in enumerate(bars1):
    width = bar.get_width()
    ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center', fontweight='bold')

# Average triple F1 performance
triple_means = df[df['type'] == 'triples'].groupby('model')['triple_f1'].mean().sort_values()
triple_stds = df[df['type'] == 'triples'].groupby('model')['triple_f1'].std()

y_pos = np.arange(len(triple_means))
bars2 = ax2.barh(y_pos, triple_means, xerr=triple_stds, align='center', alpha=0.7, capsize=5, color='lightcoral')
ax2.set_yticks(y_pos)
ax2.set_yticklabels([model.split('/')[-1] for model in triple_means.index])  # Shorten model names
ax2.set_xlabel('F1 Score', fontsize=12)
ax2.set_title('B) Diagnostic Triple Similarity Against Transcript Reference', fontweight='bold', fontsize=14)
ax2.grid(True, alpha=0.3, axis='x')
ax2.set_xlim(0, 1)

# Add value labels on bars
for i, bar in enumerate(bars2):
    width = bar.get_width()
    ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('average_performance_bars.png', dpi=300, bbox_inches='tight')
plt.savefig('average_performance_bars.pdf', bbox_inches='tight')
plt.show()

print("Average performance bars figure has been generated and saved!")
print("Files created:")
print("- average_performance_bars.png")
print("- average_performance_bars.pdf")
