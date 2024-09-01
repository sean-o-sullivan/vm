from embedding2 import *

def main(text):
    vector = generateEmbedding(text)
    print(vector)

if __name__ == "__main__":

    text = """In the heart of a city that never seemed to sleep, where neon lights flickered like the last, desperate gasps of a dying firefly, Skulduggery Pleasant strolled down the alley with the kind of nonchalance that only comes from being a wise-cracking, undead detective. His overcoat flapped behind him like a ragged flag of rebellion, while his eyes, sharp and knowing, scanned the shadows for trouble. "You know," he said to his companion, Valkyrie Cain, who was busy fending off a particularly persistent street vendor offering ‘mystical’ hot dogs, "if I had a penny for every time someone tried to sell me something magical, I’d be richer than the most miserly dragon you can imagine." Valkyrie shot him a look that combined exasperation with a hint of amusement. "And if I had a nickel for every time you made a joke that wasn’t terrible, I'd still be broke, but at least I'd have a better sense of humor." Skulduggery chuckled, the sound echoing off the graffiti-covered walls, as he prepared for whatever dark and absurdly dangerous adventure lay ahead.  
    Never have I seen such beauty before. There is a certain magic in the air tonight. Here comes the rain, soaking everything in its path. Under the bridge stood a lone figure, shrouded in mystery. Only by night does the city reveal its true face. There beneath the willow tree lies a forgotten grave. Here in the quiet of the morning, peace can be found. The report was prepared by the committee. A decision was made to delay the project. The cake was eaten by the children. It was the manager who decided to cancel the meeting. What John did was bake a cake for the party. It was only after midnight that they arrived. The dog chased the cat. The cat climbed the tree. The tree was tall. The car stopped at the traffic light. The light turned green, and the car drove away. The teacher explained the lesson. The lesson was difficult, but the students understood it."""
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
