# Wordle Solver

This object can be used to programatically come up with good guesses for the new hit game, wordle. The algorithm uses 
heuristics to determine eligible words based on the clues provided, then uses nlp techniques (TFIDF or TF) to score the 
eligible words. 

Please note that since there are many versions of the game out there, I am using an open source library to get a list 
of english words, but that list might not align perfectly with your version of the game. You may add/remove terms
from `f"utils/words_to_add_{add|remove}.py"` to suit your individual needs. 

**Example Usage**
1. First I create the solver object, which immediately suggests some good options for my first guess. These will be the same every time as the processes to create them are deterministic.
2. Then I provide all the clues I have (provide the algorithm clues from every previous guess) **NOTE: remember that python is 0 indexed**


>       solver = WordleSolver()  # can also play with params word_length and use_tfidf
> 
> first guess: {'loris': 2.234947858531693, 'snort': 2.2341932811762195, 'dumpy': 2.233495407979424}
> 
>       solver.provide_clues(correct=[('c', 0)], close=[('u', 3), ('u', 2), ('b', 4)], wrong=list('mrfaeslodt'))
> 
> next guess: {'cubic': 1.8862536688073503, 'cubby': 1.839545043329641}
