from embedding import *

text = """


Safety:
The confirmation step ensures that you don't accidentally delete directories without reviewing them first.
Running the Script:
Execute the Script: Run the script in your Python environment. It will check all author directories in bigText, flag any that match the keyword, and ask you to confirm deletion.
Review and Confirm: Carefully review the flagged directories before confirming deletion to ensure that only the intended directories are removed.


"""

vector = generateEmbedding(text)