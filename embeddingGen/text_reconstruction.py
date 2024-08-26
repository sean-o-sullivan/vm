import stanza


stanza.download('en')  
nlp = stanza.Pipeline('en', processors='tokenize')

def recombine_sentences(text, window_size=2):
    """
    Tokenizes the text, groups sentences into chunks of the specified window size,
    and then recombines them into a single text.
    """
    
    doc = nlp(text)
    sentences = [sentence.text for sentence in doc.sentences]
    
    
    chunks = [' '.join(sentences[i:i+window_size]) for i in range(0, len(sentences), window_size)]
    
    
    recombined_text = '\n\n'.join(chunks)
    
    return recombined_text

if __name__ == "__main__":
    
    original_text = """Gordon Edgley's sudden death came as a shock to everyone—not least himself. One moment he was in his study, seven words into the twenty-fifth sentence of the final chapter of his new book, And the Darkness Rained upon Them, and the next he was dead. A tragic loss, his mind echoed numbly as he slipped away.

The funeral was attended by family and acquaintances but not many friends. Gordon hadn't been a well-liked figure in the publishing world, for although the books he wrote—tales of horror and magic and wonder—regularly reared their heads in the bestseller lists, he had the disquieting habit of insulting people without realizing it, then laughing at their shock. It was at Gordon's funeral, however, that Stephanie Edgley first caught sight of the gentleman in the tan overcoat.

He was standing under the shade of a large tree, away from the crowd, the coat buttoned up all the way despite the warmth of the afternoon. A scarf was wrapped around the lower half of his face, and even from her position on the far side of the grave, Stephanie could make out the wild and frizzy hair that escaped from the wide-brimmed hat he wore low over his gigantic sunglasses. She watched him, intrigued by his appearance. And then, like he knew he was being observed, he turned and walked back through the rows of headstones and disappeared from sight.

After the service, Stephanie and her parents traveled back to her dead uncle's house, over a humpbacked bridge and along a narrow road that carved its way through thick woodland. The gates were heavy and grand and stood open, welcoming them into the estate. The grounds were vast, and the old house itself was ridiculously big.

There was an extra door in the living room, a door disguised as a bookcase, and when she was younger Stephanie liked to think that no one else knew about this door, not even Gordon himself. It was a secret passageway, like in the stories she'd read, and she'd make up adventures about haunted houses and smuggled treasures. This secret passageway would always be her escape route, and the imaginary villains in these adventures would be dumbfounded by her sudden and mysterious dis-appearance. But now this door, this secret passageway, stood open, and there was a steady stream of people through it, and she was saddened that this little piece of magic had been taken from her.

Tea was served and drinks were poured and little sandwiches were passed around on silver trays, and Stephanie watched the mourners casually ap-praise their surroundings. The major topic of hushed conversation was the will. Gordon wasn't a man who doted, or even demonstrated any great affec-tion, so no one could predict who would inherit his substantial fortune. Stephanie could see the greed seep into the watery eyes of her father's other brother, a horrible little man called Fergus, as he nodded sadly and spoke somberly and pocketed the silverware when he thought no one was looking.
"""

    
    recombined_text = recombine_sentences(original_text, window_size=2)
    
    
    print("Recombined Text:\n")
    print(recombined_text)


