# LSTM Vs RNN

RNN: A person remembering what they read last sentence in a book, but forgetting after a few sentences
LSTM: A person with a notebook (cell state) where they write important info, and carefully decide what to keep, add, or use when reading the book

Gates in LSTM:
    1. Forget gate: Decides what past information to throw away
    2. Input gate: Decides what new information to add
    3. Output gate: Decides what part of the cell state to output

Mathematically:
    