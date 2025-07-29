"""To convert this script to a Jupyter notebook, use e.g. the VSCode "Export current file as Jupyter notebook" command.

This is being saved as a script so the diffs are cleaner and I don't accidentally commit a notebook with PID in it."""

# %% [markdown]
# # Ice rink user group analysis
 

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from collections import Counter
import tabulate
import plotly.express as px
import pgeocode
import geopy.distance 

# %%
data = "data/2025-07-25.csv"

# %% [markdown]
# ## Clean up column names so don't have to deal with spaces

# %%
df = pd.read_csv(data)

# %%
df.rename(
    columns={
        "Timestamp": "timestamp",
        "Which club(s) are you a member of at the Oxford Ice Rink? ": "club",
        "Which sports/activities do you primarily participate in at the rink?  ": "activity",
        "Do you use the Oxford Ice Rink for leisure or for a club/organised activity?  ": "use_type",
        "How frequently do you use the Oxford Ice Rink?  ": "frequency",
        "What is the first half of your postcode (e.g., OX1, OX2, OX3)? (This helps us understand geographical impact without identifying you. If you don't know, please provide the nearest major postcode.)  ": "postcode",
        "What is your age group?  ": "age_group",
        "Do you have any special needs or requirements that affect your travel to the rink (e.g., Blue Badge holder, mobility issues)?": "special_needs",
        "How do you currently travel to the Oxford Ice Rink?": "travel_mode",
        "If you currently travel by car, approximately how much do you typically spend on car parking per visit to the Oxford Ice Rink area?  ": "car_parking_cost",
        "The proposed congestion charge is £5 per journey into and out of the city. How would this additional £5 charge impact your ability or willingness to travel to the Oxford Ice Rink by car?  ": "congestion_charge_impact",
        "Are you aware of any alternative means or options for travel to the Oxford Ice Rink that you could use if a congestion charge were implemented (e.g., park and ride, bus routes, cycling routes)?  ": "alternative_travel_options",
        "Please provide any additional comments or concerns you have regarding the proposed congestion charge and its potential impact on your use of the Oxford Ice Rink.  ": "congestion_charge_comments",
        "Do you consider than the daily congestion charge might stop you from partaking in your club activity? Y or N or n/a": "congestion_charge_stop_club_activity",
        "Would you like us to keep in touch with you to update you of the progress of this survey? If so and you are over the age of 18, please leave your email address here: ": "email",
    },
    inplace=True,
)
df.head()

# %%
df.describe()

# %% [markdown]
# ## Graph for plotting proportions of responses in horizontal stacked bar chart

# %%
def survey(results, category_names, figsize=(9.2, 5)):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap()(np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=figsize)
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        
        # Remove 0 values from widths and labels (to avoid "0%" labels)
        widths = np.where(widths == 0, np.nan, widths)
        labels = np.where(widths == 0, "", labels) 

        starts = data_cum[:, i] - widths

        rects = ax.barh(
            labels, widths, left=starts, height=0.5, label=colname, color=color
        )

        r, g, b, _ = color
        text_color = "white" if r * g * b < 0.5 else "darkgrey"
        ax.bar_label(rects, label_type="center", color=text_color, fmt="%.0f%%")
    ax.legend(
        ncol=len(category_names),
        bbox_to_anchor=(0, -0.1),
        loc="lower left",
        fontsize="small",
    )

    return fig, ax

# %% [markdown]
# ## Get info on club membership

# %%
def split_multilabel(text):
    """Split text with mutliple labels on common separators and clean up"""
    if pd.isna(text):
        return []
    
    # Split on common separators: comma, semicolon, 'and', plus sign
    labels = re.split(r'[,;+]|\band\b', str(text))
    
    # Clean up each club name
    cleaned_labels = []
    for club in labels:
        club = club.strip()
        if club and club.lower() not in ['', 'n/a', 'none']:
            cleaned_labels.append(club)
    
    return cleaned_labels

# %%
# Split out individual clubs and count frequencies
club_type = {
"Oxford Junior Stars": "Ice Hockey",
"Fans/Spectators": "None",
"Figure Club": "Figure Skating",
"Other": "None",
"Oxford Rising Stars": "Ice Hockey",
"OXIST": "Figure Skating",
"I am not a member of any club.": "None",
"Junior Dance Club": "Dance",
"Oxford Midnight Stars": "Ice Hockey",
"RAF Blue Wings": "Ice Hockey",
"Oxford 84's": "Ice Hockey",
"Oxford City Stars": "Ice Hockey",
"Oxford Shooting Stars": "Ice Hockey",
"Oxford University Ice Hockey Club": "Ice Hockey"
}

def clean_club_name(club_name):
    """Remove the part in parentheses from the club name.
    >>> clean_club_name("Oxford Junior Stars (Ice Hockey)")
    'Oxford Junior Stars'
    """
    # Remove the part in parentheses and strip whitespace
    cleaned_name = re.sub(r'\s*\(.*?\)', '', club_name).strip()
    return cleaned_name


# %%

# Create a list of all individual club memberships
all_clubs = []
multiclub_member_count = 0. # this is a bit of a hack
multiclub_members = []
for club_text in df['club']:
    individual_clubs = split_multilabel(club_text)
    all_clubs.extend(individual_clubs)
    if len(individual_clubs) > 1:
        multiclub_member_count += 1
        multiclub_members.append(individual_clubs)
all_clubs = [clean_club_name(club) for club in all_clubs]

individual_club_counts = Counter(all_clubs)

# Create a dataframe for easier analysis
club_freq_df = pd.DataFrame(list(individual_club_counts.items()), 
                           columns=['Club', 'Frequency']).sort_values('Frequency', ascending=False)
club_freq_df['Club Type'] = club_freq_df['Club'].map(club_type).fillna("None")            
club_freq_df['Percentage'] = round(100 * (club_freq_df['Frequency'] / sum(club_freq_df['Frequency'])), 1)

# Reorder columns and reset index
club_freq_df = club_freq_df[['Club', 'Club Type', 'Frequency', 'Percentage']]
club_freq_df.reset_index(drop=True, inplace=True)
print(club_freq_df)

# %%
# Let's cleanup the original dataframe too, adding a column for club type
# This is gnarly sorry, the multilabel in the club column is a bit of a pain 
hockey_clubs = set(club_freq_df[club_freq_df['Club Type'] == 'Ice Hockey']['Club'].tolist())
figure_skating_clubs = set(club_freq_df[club_freq_df['Club Type'] == 'Figure Skating']['Club'].tolist())
dance_clubs = set("Junior Dance Club")                   

def is_hockey_club(club_names):
    """Check if the club is an ice hockey club."""
    names = split_multilabel(club_names)
    names = [clean_club_name(name) for name in names]
    return any(name in hockey_clubs for name in names)

def is_figure_skating_club(club_names):
    """Check if the club is a figure skating club."""
    names = split_multilabel(club_names)
    names = [clean_club_name(name) for name in names]
    return any(name in figure_skating_clubs for name in names)

def is_dance_club(club_names):
    """Check if the club is a dance club."""
    names = split_multilabel(club_names)
    names = [clean_club_name(name) for name in names]
    print(names, any(name == "Junior Dance Club" for name in names))
    return any(name == "Junior Dance Club" for name in names)


# %%
df['hockey_club'] = df['club'].apply(is_hockey_club)
df['figure_skating_club'] = df['club'].apply(is_figure_skating_club)
df['dance_club'] = df['club'].apply(is_dance_club)

# %%
# Want to count the number of responses that have more than one club membership
multiclub_member_count

# %%
# Get frequency of the the multiclub_members
multiclub_members_freq = Counter(tuple(sorted(members)) for members in multiclub_members)
multiclub_members_freq

# %%
markdown_table = club_freq_df.to_markdown(index=False)
print(markdown_table)

# %%
# Get frequency by club type
club_type_counts = club_freq_df.groupby('Club Type')['Frequency'].sum().reset_index()
club_type_counts['Percentage'] = round((club_type_counts['Frequency'] / sum(club_freq_df['Frequency'])) * 100, 1)

print(club_type_counts)

# %%
# Plot the club type frequencies
# (Plotting by club is a bit of a waste of time as there are too many to be readable)
# Just include a table of this instead 
custom_order = [
"Ice Hockey",
"Figure Skating",
"Dance",
"None"
]

freq_results = {
    "": club_freq_df["Club Type"]
    .value_counts(dropna=True, normalize=True)
    .reindex(custom_order)
    .tolist()
}
freq_results = {k: list(map(lambda x: x * 100, v)) for k, v in freq_results.items()}
freq_results

# Fix up labels 
labels = custom_order
labels[-1] = "None (e.g. fans, spectators, other)"

survey(freq_results, custom_order)
plt.title("Respondents by Club Type")
plt.show()

# %% [markdown]
# <!-- ## Get info on activities -->

# %% [markdown]
# ## Get info on use type (club vs public)

# %%
# This is not multilabel 
df['use_type'].value_counts()

# %%
# Plot the use type frequencies
labels = [
"Both leisure and club",
"Club/organised activity",
"Leisure",
]

freq_results = {
    "": df["use_type"]
    .value_counts(dropna=True, normalize=True)
    # .reindex(custom_order)
    .tolist()
}
freq_results = {k: list(map(lambda x: x * 100, v)) for k, v in freq_results.items()}
freq_results

survey(freq_results, labels)
plt.title("What activities do you use the Oxford Ice Rink for?")
plt.show()

# %% [markdown]
# ## Age groups

# %%
df['age_group'].value_counts()

# %%
# Plot the use type frequencies
custom_order = [
"Under 16",
"16-24",
"25-34",
"35-44",
"45-54",
"55-64",
"65+",
]

freq_results = {
    "": df["age_group"]
    .value_counts(dropna=True, normalize=True)
    .reindex(custom_order)
    .tolist()
}
freq_results = {k: list(map(lambda x: x * 100, v)) for k, v in freq_results.items()}
freq_results

survey(freq_results, custom_order)
plt.title("Age group of respondents")
plt.show()

# %%
# Let's have a look at this by club type
custom_order = [
    "Under 16",
    "16-24",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65+",
]

freq_results = {
    "All": df["age_group"]
    .value_counts(dropna=True, normalize=True)
    .reindex(custom_order)
    .tolist(),
    "Ice hockey": df[df["hockey_club"]]["age_group"]
    .value_counts(dropna=True, normalize=True)
    .reindex(custom_order)
    .tolist(),
    "Figure skating": df[df["figure_skating_club"]]["age_group"]
    .value_counts(dropna=True, normalize=True)
    .reindex(custom_order)
    .tolist(),
    "Dance": df[df["dance_club"]]["age_group"]
    .value_counts(dropna=True, normalize=True)
    .reindex(custom_order)
    .tolist(),
    "No club": df[~df["hockey_club"] & ~df["figure_skating_club"] & ~df["dance_club"]][
        "age_group"
    ]
    .value_counts(dropna=True, normalize=True)
    .reindex(custom_order)
    .tolist(),
}
freq_results = {k: list(map(lambda x: x * 100, v)) for k, v in freq_results.items()}

# Replace the NaNs with 0s for plotting (unsure why not dealt with by dropna above...)
for key in freq_results:
    freq_results[key] = [0 if np.isnan(x) else x for x in freq_results[key]]

survey(freq_results, custom_order)
plt.title("Age group of respondents, by club type")
plt.show()

# %% [markdown]
# ## Get info on frequency

# %%
# this isn't multilabel either
df['frequency'].value_counts()

# %%
custom_order = [
    "Daily",
    "4-6 times a week", 
    "2-3 times a week",
    "Once a week",
    "2-3 times a month",
    "Once a month",
    "Less than once a month"
]

freq_results = {
    "All": df["frequency"]
    .value_counts(dropna=True, normalize=True)
    .reindex(custom_order)
    .tolist(),
    "Ice hockey": df[df["hockey_club"]]["frequency"]
    .value_counts(dropna=True, normalize=True)
    .reindex(custom_order)
    .tolist(),
    "Figure skating": df[df["figure_skating_club"]]["frequency"]
    .value_counts(dropna=True, normalize=True)
    .reindex(custom_order)
    .tolist(),
    "Dance": df[df["dance_club"]]["frequency"]
    .value_counts(dropna=True, normalize=True)
    .reindex(custom_order)
    .tolist(),
    "No club": df[~df["hockey_club"] & ~df["figure_skating_club"] & ~df["dance_club"]][
        "frequency"
    ]
    .value_counts(dropna=True, normalize=True)
    .reindex(custom_order)
    .tolist(),
}
freq_results = {k: list(map(lambda x: x * 100, v)) for k, v in freq_results.items()}
for key in freq_results:
    freq_results[key] = [0 if np.isnan(x) else x for x in freq_results[key]]
freq_results

survey(freq_results, custom_order, figsize=(14, 7))
plt.title("Frequency of visits to the ice rink, by club type")
plt.show()

# %% [markdown]
# ## Plot the postcodes where people live

# %%
df['postcode'].value_counts()

# Make a table of the postcodes
postcode_counts = df['postcode'].value_counts().reset_index()
postcode_counts.columns = ['postcode', 'count']

postcode_counts

# %%
nomi = pgeocode.Nominatim('gb')

def get_lat_long(postcode):
    """Get latitude and longitude for a given postcode."""
    if pd.isna(postcode):
        return None, None
    location = nomi.query_postal_code(postcode)
    if location is not None:
        return location.latitude, location.longitude
    else:
        return None, None

# %%
postcode_counts['latitude'], postcode_counts['longitude'] = zip(*postcode_counts['postcode'].apply(get_lat_long))

# %%
postcode_counts

# %%
ICE_RINK = (51.7488, -1.2650)

# %%
# Calculate distance to the ice rink
def calculate_distance(row):
    """Calculate distance from the ice rink for a given row, in miles"""
    if pd.isna(row['latitude']) or pd.isna(row['longitude']):
        return None
    return round(geopy.distance.geodesic((row['latitude'], row['longitude']), ICE_RINK).miles, 1)

postcode_counts['distance_to_ice_rink'] = postcode_counts.apply(calculate_distance, axis=1)

# %%
df['latitude'], df['longitude'] = zip(*df['postcode'].apply(get_lat_long))
df['distance_to_ice_rink'] = df.apply(lambda row: calculate_distance(row), axis=1)

# %%
df['distance_to_ice_rink'].describe()

# %%
# I want a histogram of the distances to the ice rink
# Using matplotlib for the histogram
plt.figure(figsize=(10, 6))
plt.hist(df['distance_to_ice_rink'].dropna(), bins=30, edgecolor='black')
plt.title('Distribution of Distances to the Ice Rink')
plt.xlabel('Distance to Ice Rink (miles)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# %%
# Do horizontal box plots of the distances to the ice rink by club type
fig, ax = plt.subplots(figsize=(14, 7))

# Create separate datasets for each club type
club_data = [
    df[~df["hockey_club"] & ~df["figure_skating_club"] & ~df["dance_club"]][
        "distance_to_ice_rink"
    ].dropna(),
    df[df["dance_club"]]["distance_to_ice_rink"].dropna(),
    df[df["figure_skating_club"]]["distance_to_ice_rink"].dropna(),
    df[df["hockey_club"]]["distance_to_ice_rink"].dropna(),
]

club_labels = ["Ice Hockey", "Figure Skating", "Dance", "No Club"]

# Create horizontal box plot
ax.boxplot(club_data, labels=club_labels, vert=False)
ax.set_xlabel("Distance to Ice Rink (miles)")
ax.set_ylabel("Club Type")
ax.set_title("Distance to Ice Rink by Club Type")
ax.grid(axis="x", alpha=0.75)
plt.tight_layout()
plt.show()

# %%
fig = px.scatter_map(
    postcode_counts,
    lat="latitude",
    lon="longitude",
    size="count",
    opacity=0.8,
    zoom=8,
    height=1200,
    width=1400,
    color_discrete_sequence=['red']
    )
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.add_scattermap(
    lat=[ICE_RINK[0]],
    lon=[ICE_RINK[1]],
    mode='markers',
    marker=dict(
        size=15,
        opacity=0.8,
        symbol='star',
    ),
)

fig.show()

# %%
postcode_counts.head()

# %% [markdown]
# ## How do they travel?

# %%
# Plot the travel mode frequencies
custom_order = [
"Car (as driver)",
"Car (as passenger)",
"Other (Please specify):",
"Cycling",
"Walking",
]
freq_results = {
    "All": df["travel_mode"]
    .value_counts(dropna=True, normalize=True)
    .reindex(custom_order)
    .tolist()
}
freq_results = {k: list(map(lambda x: x * 100, v)) for k, v in freq_results.items()}
freq_results

# Fix up labels 
labels = custom_order
labels[2] = "Other"

survey(freq_results, custom_order)
plt.title("Travel mode")
plt.show()

# %%
df["travel_mode"].value_counts()

# %% [markdown]
# ## Self-reported impact
# 
# Two questions asked here:
# 
# 1. "The proposed congestion charge is £5 per journey into and out of the city. How would this additional £5 charge impact your ability or willingness to travel to the Oxford Ice Rink by car?"
# 2. "Do you consider than the daily congestion charge might stop you from partaking in your club activity? Y or N or n/a"

# %%
df["congestion_charge_impact"].value_counts()

# %%
df["congestion_charge_stop_club_activity"].value_counts()


# %%
# Plot the level of impact
custom_order = [
"Significant negative impact (e.g., would stop me from coming, would significantly reduce my visits)",
"Moderate negative impact (e.g., would make me consider alternatives, might reduce some visits)",
"Minor negative impact (e.g., an inconvenience, but wouldn't change my habits much)",
"Positive impact (e.g., encourage me to use public transport)",
"No impact",
"Unsure",
]
freq_results = {
    "All": df["congestion_charge_impact"]
    .value_counts(dropna=True, normalize=True)
    .reindex(custom_order)
    .tolist()
}
freq_results = {k: list(map(lambda x: x * 100, v)) for k, v in freq_results.items()}
freq_results

labels = [
    "Significant negative impact",
    "Moderate negative impact",
    "Minor negative impact",
    "Positive impact",
    "No impact",
    "Unsure",
]

survey(freq_results, labels, figsize=(13, 5))
plt.title("Level of impact of congestion charge on use of the ice rink")
plt.show()

# %%
custom_order = [
"Significant negative impact (e.g., would stop me from coming, would significantly reduce my visits)",
"Moderate negative impact (e.g., would make me consider alternatives, might reduce some visits)",
"Minor negative impact (e.g., an inconvenience, but wouldn't change my habits much)",
"Positive impact (e.g., encourage me to use public transport)",
"No impact",
"Unsure",
]

freq_results = {
    "All": df["congestion_charge_impact"]
    .value_counts(dropna=True, normalize=True)
    .reindex(custom_order)
    .tolist(),
    "Ice hockey": df[df["hockey_club"]]["congestion_charge_impact"]
    .value_counts(dropna=True, normalize=True)
    .reindex(custom_order)
    .tolist(),
    "Figure skating": df[df["figure_skating_club"]]["congestion_charge_impact"]
    .value_counts(dropna=True, normalize=True)
    .reindex(custom_order)
    .tolist(),
    "Dance": df[df["dance_club"]]["congestion_charge_impact"]
    .value_counts(dropna=True, normalize=True)
    .reindex(custom_order)
    .tolist(),
    "No club": df[~df["hockey_club"] & ~df["figure_skating_club"] & ~df["dance_club"]][
        "congestion_charge_impact"
    ]
    .value_counts(dropna=True, normalize=True)
    .reindex(custom_order)
    .tolist(),
}
freq_results = {k: list(map(lambda x: x * 100, v)) for k, v in freq_results.items()}
for key in freq_results:
    freq_results[key] = [0 if np.isnan(x) else x for x in freq_results[key]]
freq_results

labels = [
    "Significant negative impact",
    "Moderate negative impact",
    "Minor negative impact",
    "Positive impact",
    "No impact",
    "Unsure",
]

survey(freq_results, labels, figsize=(14, 7))
plt.title("Level of impact of congestion charge on use of the ice rink")
plt.show()

# %%
labels = ["Yes", "No"]

freq_results = {
    "All": df["congestion_charge_stop_club_activity"]
    .value_counts(dropna=True, normalize=True)
    .tolist(),
    "Ice hockey": df[df["hockey_club"]]["congestion_charge_stop_club_activity"]
    .value_counts(dropna=True, normalize=True)
    .tolist(),
    "Figure skating": df[df["figure_skating_club"]][
        "congestion_charge_stop_club_activity"
    ]
    .value_counts(dropna=True, normalize=True)
    .tolist(),
    "Dance": df[df["dance_club"]]["congestion_charge_stop_club_activity"]
    .value_counts(dropna=True, normalize=True)
    .tolist(),
    "No club": df[~df["hockey_club"] & ~df["figure_skating_club"] & ~df["dance_club"]][
        "congestion_charge_stop_club_activity"
    ]
    .value_counts(dropna=True, normalize=True)
    .tolist(),
}
freq_results = {k: list(map(lambda x: x * 100, v)) for k, v in freq_results.items()}
for key in freq_results:
    freq_results[key] = [0 if np.isnan(x) else x for x in freq_results[key]]
freq_results

survey(freq_results, labels)
plt.title("Would congestion charge stop you taking part in club activity?")
plt.show()

# %% [markdown]
# ## A dig through the comments

# %%
# Write all comments to a text file
with open('congestion_charge_comments.txt', 'w', encoding='utf-8') as f:
    f.write("Congestion Charge Comments\n")
    f.write("=" * 50 + "\n\n")

    # Set as there are some duplicates
    comments = set(df['congestion_charge_comments'].dropna().tolist())
    
    for i, comment in enumerate(comments, 1):
        f.write(f"[{i}] {comment}\n\n")
        f.write("-" * 50 + "\n\n")


