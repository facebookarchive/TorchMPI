# Commonalities
In both CUDA and MPI versions, each participant transmits communicatorSize - 1 times.
At each step a rank has a choice between:

  1. sending a chunk to the right
  2. receiving a chunk form the left
  3. sitting out

Each chunk can only be send/received once per step
```
Balanced example:

        rank  0 1 2
  chunk
    0         S R .
    1         . S R
    2         R . S
  //
    0         R . S
    1         S R .
    2         . S R
  //

Unbalanced example:

        rank  0 1 2 3 4 5
  chunk
    0         S R . . . .
    1         . S R . . .
    2         . . S R . .
  //
    0         . S R . . .
    1         . . S R . .
    2         . . . S R .

Single chunk:
  //
  If there is a single chunk available then each process does at most
  1 send or 1 receive, potentially none..
        rank  0 1 2 3
    0         S R . .
  //
```

# CUDA
The CUDA implementation follows a pull model, receive-centric.
This is different (and easier) to write than a bulk-synchronous model with matching send and receives.
In particular, the sendingChunk *in prev* and the receivingChunk *in current* must always match.
