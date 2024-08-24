# globals.py
df3 = []
output_str = ""

# Function to update df3 and output_str
def update_globals(predictions, prediction_str):
    global df3, output_str
    df3 = predictions
    output_str = prediction_str
