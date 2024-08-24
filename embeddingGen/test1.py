from embedding import *

def main(text):
    vector = generateEmbedding(text)
    print(vector)

if __name__ == "__main__":

    text = """In the heart of a city that never seemed to sleep, where neon lights flickered like the last, desperate gasps of a dying firefly, Skulduggery Pleasant strolled down the alley with the kind of nonchalance that only comes from being a wise-cracking, undead detective. His overcoat flapped behind him like a ragged flag of rebellion, while his eyes, sharp and knowing, scanned the shadows for trouble. "You know," he said to his companion, Valkyrie Cain, who was busy fending off a particularly persistent street vendor offering ‘mystical’ hot dogs, "if I had a penny for every time someone tried to sell me something magical, I’d be richer than the most miserly dragon you can imagine." Valkyrie shot him a look that combined exasperation with a hint of amusement. "And if I had a nickel for every time you made a joke that wasn’t terrible, I'd still be broke, but at least I'd have a better sense of humor." Skulduggery chuckled, the sound echoing off the graffiti-covered walls, as he prepared for whatever dark and absurdly dangerous adventure lay ahead."""
    # print("Please enter the text to be embedded. Type 'END' on a new line when you are finished:")
    
    # # Collect multiline input
    # lines = []
    # while True:
    #     line = input()
    #     if line == "END":
    #         break
    #     lines.append(line)
    
    # # Join all lines into a single string
    # text = "\n".join(lines)
    
    main(text)
