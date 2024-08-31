import csv
import logging
from tqdm import tqdm
from embedding2 import generateEmbedding

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_entry(entry, fieldnames, output_file):
    author = entry['author']
    book_name = entry['book']
    sample_id = entry['sample_id']
    processed_sample = entry['processed_sample']
    
    # Remove the custom delimiters from the processed_sample
    processed_sample = processed_sample.replace("#/#\\#|||#/#\\#|||#/#\\#", "")
    
    print(f"Processing sample_id: {sample_id}")
    print(f"Processed sample (first 100 chars): {processed_sample[:100]}")

    try:
        
        embedding = generateEmbedding(processed_sample)

        
        row = {
            'author': author,
            'book': book_name,
            'sample_id': sample_id
        }
        row.update(embedding)  

        
        with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)
    except Exception as e:
        logging.error(f"Error processing sample_id {sample_id}: {str(e)}")

def main():
    input_csv = '/home/aiadmin/Desktop/code/vm/embeddingGen/Thursday/results_10KSample.csv'
    output_file = 'output_embeddings.csv'

    
    with open(input_csv, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        entries = list(reader)
    
    if not entries:
        logging.error("No entries found in results.csv. Exiting.")
        return

    print(f"CSV Headers: {reader.fieldnames}")
    print(f"Total entries: {len(entries)}")

    sample_text = """
We will then cross 17th Street and examine several buildings along 17th Street as we walk north towards Pennsylvania Avenue. The total distance is about three-fourths of a kilometer (half a mile). Capital Gatehouse Site 7 [Illustration: The Capitol Gatehouse, now located at 17th Street and Constitution Avenue, is made of the same sandstone used in the White House and the center part of the Capitol, but it was left unpainted. Deterioration of this stone is due to the clay it contains, not to the effects of acid rain.] This small sandstone building was built around 1828 at the west entrance to the Capitol. In 1880 it was moved (along with a twin and four gateposts) to its present site. This building is made of the same sandstone that was used in the central part of the Capitol and in the White House. Three types of deterioration are readily visible at the gatehouse: spalling, pock marks, and preferential weathering of clay layers in the stone. This stone may be more degraded than stone in the Capitol or the White House, because of variations in stone quality and maintenance to the buildings and because it has never been painted. [Illustration: This kind of sandstone was soon found to be a poor building stone because of its tendency to spall. (detail on Capitol Gatehouse)] To continue, we will cross 17th Street and examine parts of several buildings as we walk north. Organization of American States Building Site 8 [Illustration: The Organization of American States Building is made of marble and was dedicated in 1910.] This marble building was dedicated in 1910. Two sculptures in the front of the building show some alteration crusts in sheltered areas and dissolution in exposed areas. In back of the building the marble balusters on the patio are covered with blackened crusts, especially on the sides facing the garden. In many places the crusts have blistered or spalled off, exposing new surfaces to alteration. In general, the patio sides of the balusters are in much better condition than the sides that face the garden, perhaps because washing of the patio has washed off the gypsum crusts on that side of the balusters. [Illustration: Blackened gypsum crusts may blister and spall off, exposing a crumbling stone surface to further pollution.] Continue north through the garden and parking lot and cross C Street to the Daughters of the American Revolution (DAR) Buildings. DAR Constitution Hall Site 9 [Illustration: DAR Constitution Hall is made of limestone and was built in the 1930’s.] The main damage on this limestone building, built in the 1930’s, is the blackening of the side balustrade from algae or fungi. The stone is porous and therefore retains moisture, thus encouraging growth of organisms. The limestone in this building is quite uniform and shows little preferential dissolution, except in a few places. On the top of the balustrade along C Street, for example, some of the calcite matrix has dissolved from around the fossil fragments, and some holes are filled with calcite crystals. [Illustration: Some of the blackening on limestone surfaces may be from algae or fungi that readily grow in the rough surface in Washington’s warm, humid climate.] Continue east along C Street to Memorial Continental Hall. DAR Memorial Continental Hall Site 10 [Illustration: Memorial Continental Hall, built in 1909, is part of the Daughters of the American Revolution building complex.] [Illustration: Carvings at the base of the columns on the south side of Memorial Continental Hall show that carved details and sharp edges remain on sheltered areas.] The porch area on the south side of this marble building built in 1909 is a good place to look at some contrasts in marble deterioration. Parts of the balustrade have been replaced, as shown by differences in color and surface roughness of the stone. The exposed stone surface along the top of the balustrade is rougher than the surfaces in more sheltered areas. The columns on this porch are carved around the base, so you can examine the effects of exposure to rain on the carving details. The more exposed carvings have lost their sharp edges and definition compared to the sheltered carvings. The bases of the columns contain small amounts of pyrite, which is more resistant to weathering than is the calcite in the marble surrounding the pyrite. The sheltered part of the window-sill support on the west side of the porch shows an alteration crust, a dull gray accumulation on the stone surface. [Illustration: On an exposed portion of the carving on the columns at Memorial Continental Hall, the edges of the marble have rounded and the surface has roughened.] [Illustration: Pyrite grains stand in relief where calcite and micas have weathered out of the marble at Memorial Continental Hall.] [Illustration: A dull gray surface on the marble on the window-sill support shows where an alteration crust is just beginning to develop.] At the corner of 17th and C Streets, turn left and walk north along 17th Street. On our way to the Corcoran Gallery, we will pass the Red Cross building (marble, 1917). Some of the same types of marble deterioration observed at other locations are also present here. Corcoran Gallery Site 11 [Illustration: The Corcoran Gallery is built mostly of marble.] The Corcoran Gallery is marble with a granite base. It was built in 1879 and enlarged in 1927. Ornate carvings around the roof, doors, and windows have blackened crusts of gypsum, as do parts of the marble pedestals supporting the bronze lions at the front door. The marble bases also have inclusions that stand out above the surrounding calcite, which has been dissolved away. [Illustration: Marble bases for bronze lions outside the entrance to the Corcoran Gallery have feldspar inclusions that stand in relief compared to the roughened surrounding calcite.] Continuing north along 17th Street towards Pennsylvania Avenue, you will see several modern granite office buildings and the Executive Office building (formerly the State-War and Navy building), which was built from granite and completed in 1888. These granite buildings show little deterioration. Turn right onto Pennsylvania Avenue and proceed to the Renwick Galley on the northeast corner of the intersection of 17th Streets and Pennsylvania Avenue. Renwick Gallery Site 12 [Illustration: The Renwick Gallery, made of brick and sandstone, was completed in 1859.] This building of brick and sandstone, completed in 1859, is interesting from a stone preservation point of view. The decorative sandstone panels were badly deteriorated, so in 1968 the panels were saturated with epoxy to strengthen them. This treatment actually accelerated the deterioration because when water penetrated behind the epoxy-filled area, large portions of the treated panels spalled off. A second renovation attempt was therefore necessary two years after the first, and the present panels are cast sandstone. A post of the original sandstone stands at the southeast corner of the building. [Illustration: Casts of ground sandstone and epoxy replaced the original carved sandstone decorative trim at the Renwick Gallery when a first attempt to preserve the carved stone failed.] The next part of the tour begins at 15th Street and Pennsylvania Avenue S. To get there, walk east along Pennsylvania Avenue, past Blair House and between Lafayette Park and the White House. Lafayette Park has a number of bronze statues that have been cleaned fairly recently. The White House is built of sandstone that was painted white; the paint was used in part to improve the durability of the stone. After you pass the White House, you will come to the Treasury Building. Turn right onto 15th Street and walk south, towards the Washington Monument and the Mall. The total distance from the Renwick to the corner of 15th and Pennsylvania is about three-fourths of a kilometer (half a mile). Federal Triangle Buildings Site 13 On the east side of 15th Street, beginning at E Street, is the Commerce Department building, which was constructed of limestone in the 1930’s. This building is part of the Federal Triangle, a cluster of Federal office buildings in the area bounded by Pennsylvania Avenue, Constitution Avenue, and 15th Street, built primarily during the New Deal administration of President Franklin D. Roosevelt. Some sculptures on the buildings were done by participants in the WPA program. These buildings were cleaned in the 1960’s, probably by sandblasting. Look for fossils in relief and alteration crusts in some sheltered places on the carved work. Some of the blackening on this building is from dirt and organic material trapped or growing in the rough surface of the stone. [Illustration: All of the Federal buildings that form the Federal Triangle (between Pennsylvania Avenue, Constitution Avenue, and 15th Street) are made of limestone.] Continue south on 15th Street to Constitution Avenue. From the corner of 15th Street and Constitution, follow some of the foot paths half a kilometer (three tenths of a mile) to the Washington Monument. Washington Monument Site 14 This monument was begun in 1848, but it was not finished until 1885; the change in color about 150 feet up marks a change in the type of marble used to face the monument. Although it is made of marble, its smooth, straight shape and the massive blocks used in this monument have minimized the effect of acid precipitation. Dissolution does occur in a few areas, but the amount of stone material lost from dissolution is insignificant compared to the mass of the stone. [Illustration: The straight shape and massive stones in the Washington Monument minimize the impact of acid precipitation to this important landmark.] Our tour ends here, but there are many more stone buildings and monuments in Washington and in other cities that may also show the effects of urban pollution and acid precipitatiation.

"""
    sample_embedding = generateEmbedding(sample_text)
    fieldnames = ['author', 'book', 'sample_id'] + list(sample_embedding.keys())

    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    
    for entry in tqdm(entries, desc="Processing entries"):
        process_entry(entry, fieldnames, output_file)

    logging.info(f"Processing completed. Embeddings saved to {output_file}")

if __name__ == "__main__":
    main()
