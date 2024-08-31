import re

def find_true_end(text, initial_end_pos, lookahead_range=1000):
    current_end_pos = initial_end_pos

    while True:
        
        lookahead_text = text[current_end_pos:current_end_pos + lookahead_range]
        # Search for another "end" sequence within the lookahead range
        next_end_match = re.search(r'([=]{3,}|[-]{3,}|[.]{3,}|-{10,})', lookahead_text)

        if next_end_match:
            current_end_pos += next_end_match.end()
        else:
            break

    return current_end_pos

def remove_table_from_text(text):
    cleaned_text = text
    position = 0

    while True:
        # Step 1: Find the position of the next ALL CAPS word, tables often start with all caps: TABLE.X:
        start_match = re.search(r'\b[A-Z\s.,\'\-]{3,}\b', cleaned_text[position:])
        
        if not start_match:
            # No ALL CAPS word found, exit loop
            break

        start_pos = position + start_match.start()
        print(f"start pos is: {start_pos}")

        # Step 2: Find the first occurrence of sequences of `=`, `-`, `.` or dashes after the ALL CAPS word
        first_end_match = re.search(r'([=]{3,}|[-]{3,}|[.]{3,}|-{10,})', cleaned_text[start_pos:])
        
        if not first_end_match:
            # No end sequence found after this ALL CAPS word, continue searching
            position = start_pos + len(start_match.group(0))
            continue
        
        initial_end_pos = start_pos + first_end_match.end()

        # Step 3: Find the true end of the table by looking ahead 1000 characters
        true_end_pos = find_true_end(cleaned_text, initial_end_pos)

        # Step 4: Remove the table and update position
        cleaned_text = cleaned_text[:start_pos].strip() + "\n" + cleaned_text[true_end_pos:].strip()
        
        # Continue searching after the end of the removed table
        position = start_pos

    return cleaned_text
