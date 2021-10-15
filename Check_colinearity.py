import pandas as pd
import numpy as np
import glob, os
import matplotlib.pyplot as plt
import seaborn as sns

sns.color_palette("light:orange", as_cmap=True)
df = pd.read_csv(r"C:\Users\Shaun.fraser\Desktop\Co-linearity.csv")
df = df.drop("Sample", axis=1)
print(df)
corr = df.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
cmap = sns.color_palette("light:orange", as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,                    #mask=mask,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.show()


