import pandas as pd
import streamlit as st

# DataFrame setup
data = {
    'timePeriod': ['Week1', 'Week3', 'Week4', 'Week5'],
    'AzCS': [[12354, 78910], [6548], None, None],
    'MKMSFE': [[64654], None, None, [6465445]],
    'PinReset': [None, None, [32145], None],
    'AzCS_status': [['Mitigated', 'Resolved'], ['Resolved'], 'Resolved', 'Resolved'],
    'MKMSFE_status': [['Resolved'], 'Resolved', 'Mitigated', 'Resolved'],
    'PinReset_status': ['Resolved', 'Resolved', 'Resolved', 'Resolved'],
}

df = pd.DataFrame(data)

# Function to apply hyperlink with color according to status
def make_hyperlink(values, statuses):
    if values is not None:
        links = []
        for value, status in zip(values, statuses):
            if status == 'Resolved':
                color = 'green'
            elif status == 'Mitigated':
                color = 'orange'
            else:
                color = 'black'
            link = f'<a style="color:{color};" href="http://www.cnn.com/{value}">{value}</a>'
            links.append(link)
        return ", ".join(links)
    return None

# Applying hyperlink to specific columns with color based on status
for col, status_col in [('AzCS', 'AzCS_status'), ('MKMSFE', 'MKMSFE_status'), ('PinReset', 'PinReset_status')]:
    df[col] = df.apply(lambda row: make_hyperlink(row[col], row[status_col]), axis=1)

# Removing the status columns from display
df = df.drop(columns=['AzCS_status', 'MKMSFE_status', 'PinReset_status'])

# Show DataFrame as markdown
st.markdown(df.to_markdown(), unsafe_allow_html=True)


################# v1
# import pandas as pd
# import streamlit as st

# # DataFrame setup
# data = {
#     'timePeriod': ['Week1', 'Week3', 'Week4', 'Week5'],
#     'AzCS': [12354, 6548, None, None],
#     'MKMSFE': [64654, None, None, 6465445],
#     'PinReset': [None, None, 32145, None],
#     'AzCS_status': ['Mitigated', 'Resolved', None, None],
#     'MKMSFE_status': ['Resolved', None, 'Mitigated', None],
#     'PinReset_status': [None, None, None, 'Resolved'],
# }

# df = pd.DataFrame(data)

# # Function to apply hyperlink with color according to status
# def make_hyperlink(value, status):
#     if pd.notnull(value):
#         if status == 'Resolved':
#             color = 'green'
#         elif status == 'Mitigated':
#             color = 'orange'
#         else:
#             color = 'black'
#         return f'<a style="color:{color};" href="https://www.cnn.com/{value}">{value}</a>'
#     return None

# # Applying hyperlink to specific columns with color based on status
# for col, status_col in [('AzCS', 'AzCS_status'), ('MKMSFE', 'MKMSFE_status'), ('PinReset', 'PinReset_status')]:
#     df[col] = df.apply(lambda row: make_hyperlink(row[col], row[status_col]), axis=1)

# # Removing the status columns from display
# df = df.drop(columns=['AzCS_status', 'MKMSFE_status', 'PinReset_status'])

# # Show DataFrame as markdown
# st.markdown(df.to_markdown(), unsafe_allow_html=True)

################################################## v0
# import streamlit as st
# import pandas as pd
# import json
# import re
# from datetime import datetime, timedelta
# import alpaca_trade_api as alpaca
# import plotly.graph_objects as go

# custom_css = """
#   <style>
#     table {
#       border-collapse: collapse;
#       width: 100%
#     }
#     th, td {
#       border : 1px solid black;
#       padding: 8px;
#       text-align: left;
#       white-space: pre-wrap;
#       word-wrap: break-word
#     }
#   </style>
# """

# st.set_page_config(page_title= "Camp 1: Explore", page_icon = "🔢", layout="wide")
# st.title("🔢 My StockAI Project")

# # filename = r"C:\development\board.csv"
# # import pandas as pd
# # import streamlit as st

# # DataFrame setup
# data = {
#     'timePeriod': ['Week1', 'Week3', 'Week4', 'Week5'],
#     'AzCS': [12354, 6548, None, None],
#     'MKMSFE': [64654, None, None, 6465445],
#     'PinReset': [None, None, 32145, None],
#     'AzCS_status': ['Mitigated', 'Resolved', None, None],
#     'MKMSFE_status': ['Resolved', None, 'Mitigated', None],
#     'PinReset_status': [None, None, None, 'Resolved'],
# }

# df = pd.DataFrame(data)

# # Function to apply hyperlink
# def make_hyperlink(value):
#     if pd.notnull(value):
#         return f"[{value}](http://www.cnn.com/{value})"
#     return None

# # Applying hyperlink to specific columns
# for col in ['AzCS', 'MKMSFE', 'PinReset']:
#     df[col] = df[col].apply(make_hyperlink)

# # Function to apply color
# def color_status(val):
#     if val == 'Resolved':
#         color = 'green'
#     elif val == 'Mitigated':
#         color = 'orange'
#     else:
#         return val
#     return f'<span style="color:{color};">{val}</span>'

# # Applying color to status columns
# for col in ['AzCS_status', 'MKMSFE_status', 'PinReset_status']:
#     df[col] = df[col].apply(color_status)

# # Show DataFrame as markdown
# st.markdown(df.to_markdown(), unsafe_allow_html=True)
