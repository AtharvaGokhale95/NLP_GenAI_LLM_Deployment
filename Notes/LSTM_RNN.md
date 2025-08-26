# LSTM Vs RNN

RNN: A person remembering what they read last sentence in a book, but forgetting after a few sentences
LSTM: A person with a notebook (cell state) where they write important info, and carefully decide what to keep, add, or use when reading the book

Gates in LSTM:
    1. Forget gate: Decides what past information to throw away
    2. Input gate: Decides what new information to add
    3. Output gate: Decides what part of the cell state to output

LSTM Networks: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

1. LSTM was invented to remember information for a long time while still focusing on recent information
2. LSTM introduces a memory cell (like a conveyor belt) that carries information across time steps.
    At each step, it decides: It does this using gates
    1. What to keep
    2. What to update
    3. What to throw away
3. LSTM working:
   1. At each time step (say, each word in a sentence):
      1. The LSTM looks at the current input and previous hidden state
      2. Gates decide what to forget, what new info to add, and what to output
      3. The cell state carries forward the long-term memory
      4. The hidden state is the short-term, immediate output
   2. $C_t$ is the cell state - the memory of the LSTM at time step 't'
      1. Unlike the hidden state $h_t$ (which is like short-term “working memory”), $C_t$ is designed to flow almost unchanged across many time steps so the network doesn’t forget important context
      2. Analogy:
         1. $C_t$ = Your running understanding of the whole plot so far (long-term memory).
         2. $h_t$ = What you’d summarize if someone asked you right now “what’s happening in this chapter?”
      3. So now at input 't', we have following parameters available:
         1. $x_t$ which is the input value at time 't'
         2. $h_{t-1}$ which is the previous hidden state (short term memory - working memory)
         3. $f_t$ (forget gate layer) -> Based on $x_t$ and $h_{t-1}$, it decides what to throw away from the previous cell state $C_{t-1}$: ![alt text](Images/image-8.png)
            1. The value is between 0 and 1: 1- “completely keep this” while a 0 - “completely get rid of this
            2. E.g: Language model trying to predict the next word based on all the previous ones. In such a problem, the cell state might include the gender of the present subject, so that the correct pronouns can be used. When we see a new subject, we want to forget the gender of the old subject
         4. The next step is to decide what new information we’re going to store in the cell state at time step 't' - Update the previous cell state $C_{t-1}$. This is done in 2 parts
            1. Input Gate Layer: Based on the input $x_t$, it decides which values to update
            2. tanh layer: Created a vector of new candidate values 