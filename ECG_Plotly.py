import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Read data from CSV
df = pd.read_csv("info-main-classes.csv")

# Assuming your dataset has columns: 'DX'
dx_counts = df['DX'].value_counts()

# Create bar chart
bar_chart = go.Figure(data=[go.Bar(
    x=dx_counts.index,
    y=dx_counts.values,
    marker_color=['green', 'blue', 'red', 'yellow']  # You can customize colors here
)])

bar_chart.update_layout(title='Diagnosis Distribution',
                        xaxis_title='Diagnosis',
                        yaxis_title='Count')

# Save bar chart as HTML
bar_chart.write_html("dx_bar_chart.html")

# Create pie chart
pie_chart = go.Figure(data=[go.Pie(labels=dx_counts.index,
                                   values=dx_counts.values)])

pie_chart.update_layout(title='Diagnosis Distribution')

# Save pie chart as HTML
pie_chart.write_html("dx_pie_chart.html")

# Group by diagnosis and age, count patients
dx_age_distribution = df.groupby(['DX', 'Age']).size().reset_index(name='Count')

# Plot line chart
line_chart = px.line(dx_age_distribution, x='Age', y='Count', color='DX', title='Diagnosis Distribution by Age')

# Save line chart as HTML
line_chart.write_html("dx_Age_line_chart.html")

# Group by diagnosis and gender, count patients
dx_gender_distribution = df.groupby(['DX', 'Gender']).size().reset_index(name='Count')

# Plot bar chart
bar_chart = px.bar(dx_gender_distribution, x='DX', y='Count', color='Gender', barmode='group', title='Diagnosis Distribution by Gender')

# Save bar chart as HTML
bar_chart.write_html("dx_Gender_bar_chart.html")

# Now you can combine these HTML files into a single dashboard using an HTML iframe in an HTML file.
# Here's a simple example of how you can do it manually:

dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arrhythmia Patients Dashboard</title>
</head>
<body>
    <h1>Arrhythmia Patients Dashboard</h1>
    <iframe src="dx_bar_chart.html" width="45%" height="400"></iframe>
    <iframe src="dx_pie_chart.html" width="45%" height="400"></iframe>
    <iframe src="dx_age_line_chart.html" width="45%" height="400"></iframe>
    <iframe src="dx_gender_bar_chart.html" width="45%" height="400"></iframe>
</body>
</html>
"""

# Save the dashboard HTML content to a file
with open("dashboard_ECG.html", "w") as file:
    file.write(dashboard_html)