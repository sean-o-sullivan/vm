import pandas as pd
import re
from bs4 import BeautifulSoup
import unicodedata
from tqdm import tqdm
#a culmination of my previous efforts
def find_true_end(text, initial_end_pos, lookahead_range=1000):
    current_end_pos = initial_end_pos

    while True:
        lookahead_text = text[current_end_pos:current_end_pos + lookahead_range]
        next_end_match = re.search(r'([=]{5,}|[-]{5,}|[.]{5,}|-{5,})', lookahead_text)

        if next_end_match:
            current_end_pos += next_end_match.end()
        else:
            break

    return current_end_pos

def remove_table_from_text(text):
    cleaned_text = ""
    position = 0

    while True:
        start_match = re.search(r'\b[A-Z]{3,}\b', text[position:])

        if not start_match:
            cleaned_text += text[position:]
            break

        start_pos = position + start_match.start()
        cleaned_text += text[position:start_pos].strip() + "\n"

        first_end_match = re.search(r'([=]{5,}|[-]{5,}|[.]{5,}|-{5,})', text[start_pos:])
        
        if not first_end_match:
            position = start_pos + len(start_match.group(0))
            continue
        
        initial_end_pos = start_pos + first_end_match.end()
        true_end_pos = find_true_end(text, initial_end_pos)
        table_content = text[start_pos:true_end_pos]
        print(f"Removing table from position {start_pos} to {true_end_pos}")
        print("Table content:")
        print(table_content)
        print("----")

        position = true_end_pos

    return cleaned_text.strip()

def clean_text(text):
    
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = re.sub(r'\((?:\w+,?\s+(?:et al\.)?,?\s+)?(?:19|20)\d{2}[a-z]?(?::\d+(?:-\d+)?)?(?:\s+and\s+(?:\w+,?\s+(?:et al\.)?,?\s+)?(?:19|20)\d{2}[a-z]?(?::\d+(?:-\d+)?)?)*\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = remove_table_from_text(text)
    text = re.sub(r'\{.*?\}', '', text)
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'(?m)^\s*[\|+].*[\|+]\s*$', '', text)
    text = re.sub(r'(?m)^\s*[-+]+\s*$', '', text)
    text = re.sub(r'(?m)^\s*[a-zA-Z0-9]+\s*[-+*/^()]+.*$', '', text)
    text = re.sub(r'(?m)^\s*[∑∫∏∂∇Δ].*$', '', text)
    text = re.sub(r'[±∓×÷∙∘·°∂∇∆∑∏∫√∛∜∝∞≈≠≡≤≥≪≫⊂⊃⊄⊅⊆⊇⊈⊉⊊⊋∈∉∋∌∍∎∏∐∑−]', '', text)
    text = re.sub(r'\b(\d+(?:\s+\d+)+)\b', '', text)
    text = re.sub(r'---+', '--', text)
    text = re.sub(r'[—–]', '-', text)
    text = re.sub(r'[""''""‹›«»]', "'", text)
    text = re.sub(r'[''´`]', "'", text)
    text = re.sub(r'[•◦▪▫▸▹►▻➤➢◆◇○●]', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'(?<!\w)[\^\d+]', '', text)
    text = re.sub(r'[™®©℠]', '', text)
    fraction_map = {
        '½': '1/2', '⅓': '1/3', '⅔': '2/3', '¼': '1/4', '¾': '3/4',
        '⅕': '1/5', '⅖': '2/5', '⅗': '3/5', '⅘': '4/5', '⅙': '1/6',
        '⅚': '5/6', '⅐': '1/7', '⅛': '1/8', '⅜': '3/8', '⅝': '5/8',
        '⅞': '7/8', '⅑': '1/9', '⅒': '1/10'
    }
    for frac, repl in fraction_map.items():
        text = text.replace(frac, repl)
    
    
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'So')
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    
    
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    
    
    text = re.sub(r'\s+([,.!?:;])', r'\1', text)
    text = re.sub(r'([,.!?:;])\s+', r'\1 ', text)
    
    
    text = re.sub(r'\(\s*\)', '', text)  
    text = re.sub(r'\(\s*[a-z]\s*\)', '', text)  

    
    text = re.sub(r'\(\s*(Pl\.\s*\d+\s*,)?\s*Fig\.\s*\d+(\.\d+)?\s*\)', '', text)

    
    text = '\n'.join(line for line in text.split('\n') if len(line.split()) > 1 or len(line.strip()) < 3)
    
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text




sample_text = """
Iyenaga__T___Toyokichi_-36822_Japanandthe.txt-11,Iyenaga T  Toyokichi ,"This is the capital reason which is being ascribed for the discriminatory effort against the Japanese in California by the leaders of the movement. Congestion in California. The Japanese, moreover, manifest a strong tendency to congregate in a locality where they realize a social condition which is a poor hybrid of Japanese and American ways. The tendency to group together is not a phenomenon peculiar to Japanese immigrants alone. Such a tendency is manifested by almost all immigrants in America in different degrees. In the case of the Japanese, however, several additional factors operate to necessitate their huddling together--they are ethnologically different; English is an entirely different language from theirs; their customs are wholly different from those of Americans; their segregation offers advantages and facilities to some Americans who deal with them. The external hostile pressure naturally compresses them into small groups. Whatever the cause, it is true that this habit of collective living among themselves retards the process of assimilation, and, moreover, makes the Japanese problem loom large in the eyes of the white population living in adjoining places. Fear and Envy Incited by Japanese Progress. In addition to this, a point to be noted is the increase in number of Japanese and their rapid economic development within the State of California. The question of immigration becomes inextricably mixed up in the minds of the populace with the problem of the treatment of those who are already admitted. They act and react as causes and effects of the agitation. The apprehension of a Japanese ""influx"" expresses itself in a hostile attitude toward the Japanese already domiciled there. Conversely, the conflict arising from the presence of Japanese in California naturally prompts opposition against Japanese immigration. Now, it so happened that recently, and especially since the war, the number of Japanese coming to the United States through the California port has decidedly increased. This is due to the increased arrival of travelers, business men, officials, and students, as a consequence of the closer relationship between America and Japan, as we shall see in the next chapter. Nevertheless, it incites the fear of the Californians and induces them to adopt more stringent measures against the Japanese living in that State. On the other hand, the economic status of the Japanese in California has been steadily developing. They are entering in some directions into serious competition with the white race. Thus, in agriculture, their steady expansion through industry and thrift has caused alarm among small white farmers. Added to this is the high birth rate among the Japanese, which, because of their racial and cultural distinction, forms a problem touching the fundamental questions of the American commonwealth. Summary. By the foregoing analysis of the situation, we see that although the problem of the Japanese in California has been made the subject of political and private exploitation, and thereby rendered unnecessarily complicated and acute, it is, nevertheless, a grave problem which contains germs that are bound to develop many evils unless it is properly solved. In the following chapters we shall study the status of the Japanese in California in respect to population and birth rate, their agricultural condition, their living and culture, and their economic attainments, with a view to elucidating just wherein lie the precise causes of the difficulties. CHAPTER VII FACTS ABOUT THE JAPANESE IN CALIFORNIA--POPULATION AND BIRTH RATE A knowledge of the facts regarding the Japanese population in California is important, because it has been a point of sharp dispute between those who insist on exclusion and those who oppose it, the former arguing that the Japanese are increasing at an amazing rate through immigration, smuggling, and birth, threatening to overwhelm the white population in the State, the latter contending that they are not multiplying in a way menacing to the State of California. The fact that such a dispute prevails in the matter of the number of Japanese suggests that it is, at least, one of the crucial points on which the whole problem rests. This is true in the sense that, if the Japanese in California were decreasing in number as the American Indians are, it would be totally useless to waste energy in an attempt to quicken the final extinction. If, on the other hand, they were to multiply in a progressively higher rate so as to overwhelm the white population, it would certainly be serious both for California and for the United States. Number of Japanese in California. This being the case, it is but natural that the enemies of the Japanese should exaggerate the number of Japanese living in California. The leaders of the movement for excluding Japanese estimate their number as no less than one hundred thousand. The report of the State Board of Control of California, prepared for the specific purpose of emphasizing the gravity of the Japanese problem in California, enumerated the population of Japanese in that State at the end of December, 1919, as 87,279. This number turned out to be 13,355 higher than the number reported by the Foreign Office of Japan,[11] which was based on the Consular registrations (including American-born offspring of the Japanese) and the count made by the Japanese Association of America. Most fortunately, the preliminary publication of a part of the United States Census for 1920 removed the uncertainty arising from the discrepancy by stating the exact number of the Japanese in California to be 70,196. The possible cause of the over-estimation by the Board of Control is to be found in its method of computation. Instead of counting the actual number of residents, it simply added the number of net gain from immigration and the excess in birth over death statistics to the returns of the census of 1910, overlooking the fact that in the meantime a great number of Japanese were leaving California for Japan as well as other States of the Union. The present number of Japanese is a minor matter compared with its dynamic tendency. The rate of increase of the Japanese population in California in the past may be easily obtained by comparing the returns of the United States Census. The following table indicates the number and rate of decennial increase: NUMBER OF JAPANESE IN CALIFORNIA ACCORDING TO THE UNITED STATES CENSUS. =========================================== Year. Number. Decennial Percentage of Increase. Decennial Increase. ----- ------- --------- ------------------- 1880 86 ..... ....... 1890 1,147 1,061 1,234 % 1900 10,151 9,004 785 % 1910 41,356 31,205 307.3% 1920 70,196 28,840 69.7% =========================================== We see from the above table that after half a century of Japanese immigration to the United States, California's net gain amounts to a little over 70,000, the number having increased at an average rate of 14,025 per decade, or 1603 per annum. We also observe that the percentage of decennial increase gradually decreased from 1234 per cent. to 69.7 per cent. It is useful to compare this development of the Japanese population with that of California in general, because it gives an idea of the relative importance of the Japanese increase. This is shown in the following table, in which the decennial rates of increase between them are compared: COMPARISON OF POPULATION INCREASE OF CALIFORNIA AND OF JAPANESE IN CALIFORNIA. ================================================================== Year. Number. Decennial Rate of Rate of Percentage of Increase. Decennial Japanese Japanese to the Increase. Decennial Total Population Increase. of California. ----- ----------- ----------- --------- --------- ---------------- 1880 864,694 ......... .... .... .0099% 1890 1,213,398 348,704 40.3% 1234 % .095 % 1900 1,485,053 271,655 22.3% 785 % .68 % 1910 2,377,549 892,496 60.0% 307.3% 1.73 % 1920 3,426,861 1,049,312 44.1% 69.7% 2.04 % ================================================================== Thus we see that while the percentage of decennial increase of Japanese has been fast decreasing since the census of 1890, descending from 1234 per cent. to 785 per cent. in the next census, and to 307.3 per cent. in 1910, and 69.7 per cent. in 1920, that of California is headed, on the whole, towards an increase. We also notice that the percentage of the Japanese population to the total population of California also shows a tendency to slow growth, increasing only three tenths of one per cent. during the last decade. As a general conclusion, therefore, we may say that the rate of increase of Japanese in California is slowly declining while that of the total population of California is steadily increasing. In the next place, how does the status of the Japanese population in California compare with that in the continental United States? In the following table, we compare the rate of increase in California and the United States, and enumerate the percentage of the number of Japanese in California to the total number of Japanese in the United States: JAPANESE POPULATION IN THE UNITED STATES AND CALIFORNIA. ===================================================================== Census. Japanese in Decennial Rate of Rate of Percentage of Continental Increase of Decennial Decennial Japanese in United Japanese in Increase. Increase of California to States. Continental Japanese in entire Japanese United California. population of States. United States. ------------------- ----------- --------- ----------- --------------- 1880 148 ...... ....... ...... 58.1% 1890 2,039 1,891 1,277.7% 1234.0% 56.2% 1900 24,326 22,287 1,093.0% 785.0% 41.7% 1910 72,157 47,831 196.6% 307.3% 57.3% 1920 119,207 47,050 65.2% 69.7% 58.8% ====================================================================="
"""


cleaned_text = clean_text(sample_text)


print("Cleaned Text:")
print(cleaned_text)



