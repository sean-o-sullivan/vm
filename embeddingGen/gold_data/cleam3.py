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
    cleaned_text = ""
    position = 0

    while True:
        # Step 1: I need to find the position of the next ALL CAPS word, like TABLE.X:
        start_match = re.search(r'\b[A-Z\s.,\'\-]{3,}\b', text[position:])
        
        if not start_match:
            # No ALL CAPS word found, append remaining text and exit loop
            cleaned_text += text[position:]
            break
        start_pos = position + start_match.start()
        print(f"Start position of table: {start_pos}")

        # append the text before the current table
        cleaned_text += text[position:start_pos].strip() + "\n"

        # Step 2: Find the first occurrence of sequences of `=`, `-`, `.` or dashes after the ALL CAPS word
        first_end_match = re.search(r'([=]{3,}|[-]{3,}|[.]{3,}|-{10,})', text[start_pos:])
        
        if not first_end_match:
            # No end sequence found after this ALL CAPS word, continue searching
            position = start_pos + len(start_match.group(0))
            continue
        
        initial_end_pos = start_pos + first_end_match.end()

        # Step 3: Find the true end of the table by looking ahead 1000 characters
        true_end_pos = find_true_end(text, initial_end_pos)
        table_content = text[start_pos:true_end_pos]
        print(f"\n\n\nRemoving table from position {start_pos} to {true_end_pos}")
        print("Table content:")
        print(table_content)
        print("----\n\n\n")

        # Update the position to continue searching after the end of the removed table
        position = true_end_pos

    return cleaned_text.strip()


# Example usage with a test case
text_input = """
TEXT BEFORE

TABLE 1.--SAMPLE DATA
=====================
| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
| Data 3   | Data 4   |
----------------------
Some more text in the table
===========================
Another section of the table
---------------------------
TEXT AFTER


TABLE 1.--PROGRESS AND METHODS OF EXCAVATION IN GOOD GROUND. THIRTY-THIRD STREET. ============================================================ 1 2 3 -----------------------------+--------+--------------------+ Type of excavation. Tunnels. Worked from: -----------------------------+--------+--------------------+ Full-sized single tunnel B 1st Ave. shaft. Full-sized single tunnel A 1st Ave. shaft. Full-sized twin tunnel A and B 1st Ave. shaft. Full-sized twin tunnel A and B Intermediate shaft. (West of shaft.) Full-sized twin tunnel A and B Intermediate shaft. (East of shaft.) Full-sized twin tunnel A and B Intermediate shaft. (East of shaft.) Exploration drift A and B Intermediate shaft. (West of shaft.) Twin tunnel. Enlargement A and B West shaft. of exploration drift (East of shaft.) =============================+========+===================== ====================================================================== 4 5 6 7 ----------------------------------+--------+------------+------------+ Length Average DATES. Time tunnel advance ---------------------------------- elapsed, excavated, per day, in in in From To days. linear feet. linear feet. ----------------------------------+--------+------------+------------+ Feb. 28, 1906. May 12, 1906. 74 346 4.7 Feb. 28, 1906. Apr. 30, 1906. 62 255 4.1 Aug. 23, 1906. Jan. 5, 1907. 136 789 5.8 Apr. 4, 1906. Oct. 31, 1906. 210 730 3.5 Apr. 4, 1906. Oct. 31, 1906. 210 783 3.7 Nov. 1, 1906. Dec. 26, 1906. 56 311 5.5 Mar. 1, 1907. July 23, 1907. 145 947 6.5 Sept. 6, 1907. Dec. 4, 1907. 89 603 6.8 ===============+==================+========+============+============= ===================================================== 8 ----------------------------------------------------- Methods and conditions. ----------------------------------------------------- Top heading and bench. Muck loaded by hand. "" "" "" "" "" "" "" "" Top full-width heading and bench. Muck loaded by steam shovel. Working exclusively on this heading. Top center heading and bench. Muck loaded by steam shovel. Working alternately in headings east and west of the shaft. Top center heading and bench. Muck loaded by steam shovel. Working alternately in headings east and west of the shaft. Top full-width heading and bench. Muck loaded by steam shovel working exclusively on this heading. Exploration drift about 9 ft. by 12 ft. Mucking by hand. "" "" "" "" "" "" "" "" Top full-width heading and bench. Muck loaded by steam shovel working exclusively on this heading. =============================+========+===================== ====================================================

Text after table.

    TABLE 2.--General characteristics of the continental rise northwest Africa Sector Values measured from Profiles E-11 to E-21 ==================================================================== Depth Segment Upper edge Lower edge Gradient Width ==================================================================== Upper continental rise 1 1200 200 1500 200 1:90 30 30 10? 2 1500 200 1600 200 1:200 100 30 15? 3 1600 200 1800 100 1:100 50 25 15? Lower continental rise 1 1800 100 2000 100 1:400 200 75 50 2 2000 100 2000 100 1:1500 500 150 50 3 2000 100 2700 100 1:500 200 200 50 Abyssal plain 2700 100 3000 75 1:1250 250 200 50 --------------------------------------------------------------------

as the following table indicates: DEATH RATE OF JAPANESE IN CALIFORNIA. ======================== Year. Number. Percentage of Death per 1000. ----- ------- ---------- 1910 440 10.64% 1911 472 ..... 1912 524 ..... 1913 613 ..... 1914 628 ..... 1915 663 ..... 1916 739 ..... 1917 910 ..... 1918 1150 ..... 1919 1360 20.00% ======================== The rate of death per one thousand population increased twice during the past ten years. The rate of death per one thousand population increased twice during the past ten years. The rate of death per one thousand population increased twice during the past ten years. The rate of death per one thousand population increased twice during the past ten years. The rate of death per one thousand population increased twice during the past ten years. The rate of death per one thousand population increased twice during the past ten years. The rate of death per one thousand population increased twice during the past ten years. The rate of death per one thousand population increased twice during the past ten years. The rate of death per one thousand population increased twice during the past ten years. The rate of death per one thousand population increased twice during the past ten years. The rate of death per one thousand population increased twice during the past ten years. The rate of death per one thousand population increased twice during the past ten years. The rate of death per one thousand population increased twice during the past ten years. The rate of death per one thousand population increased twice during the past ten years. The rate of death per one thousand population increased twice during the past ten years.



JAPANESE POPULATION IN THE UNITED STATES AND CALIFORNIA. ===================================================================== Census. Japanese in Decennial Rate of Rate of Percentage of Continental Increase of Decennial Decennial Japanese in United Japanese in Increase. Increase of California to States. Continental Japanese in entire Japanese United California. population of States. United States. ------------------- ----------- --------- ----------- --------------- 1880 148 ...... ....... ...... 58.1% 1890 2,039 1,891 1,277.7% 1234.0% 56.2% 1900 24,326 22,287 1,093.0% 785.0% 41.7% 1910 72,157 47,831 196.6% 307.3% 57.3% 1920 119,207 47,050 65.2% 69.7% 58.8% =====================================================================

Text after table



Cooley__M__S___Maxwell_Stephens___1874_-70409_Vacuumcleaningsystems.txt-11,Cooley M S  Maxwell Stephens  1874 ,"Either of these values are well within the maximum variation. It is, therefore, evident that when the vacuum producer cannot be centrally located that a piping system which will give the most nearly equal length of pipe to each riser will yield the best results. A vacuum cleaning system for serving a passenger car storage yard will best illustrate the effect of long lines of piping. A typical yard having 8 tracks, each of sufficient length to accommodate 10 cars, is shown in Fig. 63. The vacuum producer in this case is located at the side of the yard at one end, which is not an unusual condition. [Illustration: FIG. 63. VACUUM CLEANING LAYOUT FOR A PASSENGER CAR STORAGE YARD.] The capacity of this yard will be 80 cars which must generally be cleaned between the hours of midnight and 6 A. M., or a period of 6 hours for cleaning. It will require one operator approximately 20 minutes to thoroughly clean the floor of one car, on account of the difficulty in getting under and around the seat legs. In addition to this, it is also necessary to clean the upholstery of the seats and their backs, which will require approximately 25 minutes more or 45 minutes for one operator to thoroughly clean one car. Therefore, one operator can clean 8 cars during the cleaning period and a ten-sweeper plant will be necessary to serve the yard. One lateral cleaning pipe must be run between every pair of tracks or four laterals in all to properly reach all cars without running the hose across tracks where it might be cut in two by the shifting of trains. Outlets should be spaced two car lengths apart in order to bring an outlet opposite the end of every second car. This will make it possible to bring the hose in through the end of the car at the door opening and clean the entire car from one end which can be done by using 100 ft. of hose. The use of double the number of outlets and 50 ft. of hose would require two attachments of the hose to clean one car resulting in a loss of time in cleaning and is not recommended. In this case, 100 ft. of hose would be the shortest length that would be likely to be used and 60 cu. ft. of free air would be the maximum to be allowed for when using 1¹ ₄-in. hose. The simplest layout for a piping system to serve this yard would be that shown in Fig. 63. When the entire yard is filled with cars and the entire force of ten operators is started to clean them it would be possible to so divide them that not over three operators would be working on any one lateral and this condition will be assumed to exist. The maximum size for the laterals between the tracks will be that for three sweepers, or 3 in., and it will not be safe to use this size beyond the second inlet from the manifold, from which point to the end of the lateral it must be made 2¹ ₂ in., the maximum size for either one or two sweepers. The total loss of pressure due to friction from the inlet at x (Fig. 63) to the separator can be readily calculated from the chart (Fig. 56) as follows: TABLE 20. PRESSURE LOSSES FROM INLET TO SEPARATOR IN SYSTEM FOR CLEANING RAILROAD CARS. --------+---------+----------+---------+--------+--------+-------- Average Friction Final Cubic Ft. Equivalent Size of Vacuum, Loss, Vacuum, Section Free Air Length, Pipe, Ins. Ins. Ins. of Pipe. per min. feet. In. Diam. Mercury. Mercury. Mercury. --------+---------+----------+---------+--------+--------+-------- x--5 60 150 2¹ ₂ 6 0.35 6.35 5--4 120 140 2¹ ₂ 7 1.35 7.70 4--2 180 280 2¹ ₂ 11 7.0 14.70 2--w 180 190 3 16 4.0 18.70 w--u 360 20 5 19 0.9 19.60 u--s 480 20 6 20 0.5 20.10 s--sep 600 20 6 20 0.4 20.50 --------+---------+----------+---------+--------+--------+-------- This loss will be the maximum that is possible under any condition as it is computed with three sweepers working on the three most remote inlets on laterals xy and vw and with two sweepers on laterals tu and rs. The pipes are the largest which will give a velocity of 40 ft. per second with the full load and at the density which will actually exist in the pipe lines with the vacuum maintained at the separator of 20 in. mercury in all cases, except the pipe from s to separator. There the size was maintained at 6 in., as it was not considered advisable to increase this on account of the reduced velocity which would occur when less than the total number of sweepers might be working. As bare floor brushes will be used for cleaning coaches it is not considered advisable to reduce the air quantity below that required by such renovators. However, when carpet renovators are used in Pullman cars and upholstery renovators are used on the cushions of both coaches and Pullmans, the air quantity will be reduced. This condition may exist at any time, also one of these carpet or upholstery renovators may be in use on one of the inlets most remote from the separator at the same time that nine floor brushes are in use on the remaining outlets. In that case a vacuum at the separator of less than 20 in. would result in a decrease in the vacuum at the inlet to which this renovator was attached. The vacuum at the separator must, therefore, be maintained at the point stated. With such a vacuum there will be variation in the vacuum at the hose cocks of from 6 in. to 20 in. or seven times the maximum allowable variation in vacuum at the hose cocks. If 1-in. hose be used, the maximum air quantities will be 40 cu. ft. per sweeper. If we start with a vacuum at the inlet x of 10-in. mercury, the vacuum at the separator will again be 20 in. and we now have a variation of 10 in. between the nearest and most remote inlet from the separator, or five times the maximum allowed. Either of these conditions is practically prohibitive, due to: 1. The excessive power consumption at the separator. 50 H. P. in case 1¹ ₄-in. hose is used, and 33 H. P. in case 1-in. hose is used. 2. The excessive capacity of the exhauster in order to handle the air at such low density, a displacement of 1,800 cu. ft. being necessary in case 1¹ ₄-in. hose is used and 1,200 cu. ft. in case 1-in hose is used. 3. The great variation in the vacuum at the hose cocks which will admit the passage of so much more air through a brush renovator on an outlet close to the separator as to render useless the calculations already made, or the high vacuum at the carpet or upholstery renovators would render their operation practically impossible. [Illustration: FIG. 64. ARRANGEMENT OF PIPING RECOMMENDED AS BEST FOR PASSENGER CAR STORAGE YARD.] Such a layout must be at once dismissed as impractical, and some other arrangement must be adapted. The arrangement of piping shown in Fig. 64 is considered by the author to be the best that can be devised for this case. With this arrangement the vacuum at the separator must be maintained at 11.50 in. mercury to insure a vacuum of 6 in. mercury at the outlet x under the most unfavorable conditions, and the maximum variation in vacuum at the inlets will be 3.45 in. mercury when 1¹ ₄-in. hose is used. This will give a maximum vacuum under a carpet renovator of 7¹ ₂ in. mercury with 37 cu. ft. of air passing and will permit 70 cu. ft. of free air per minute to pass a brush renovator when operating with 100 ft. of hose attached to the inlet at which the highest vacuum is maintained. Both of these conditions will permit satisfactory operation and the increased air quantities will not seriously affect the calculations already made. The maximum horse power required at the separator will now be 20.5 as against over 50 in the case of the piping arrangement shown in Fig. 63, and will require an exhauster having a displacement of 950 cu. ft. instead of 1,800 cu. ft. required with the former layout. If 1-in. hose is used and 10 in. mercury maintained at the outlet x under the same conditions as before, the vacuum at the separator will be 14.50 in. and the maximum variation in the vacuum at the inlets will be 3 in., which will give a maximum vacuum under a carpet renovator of 6 in. mercury with 32 cu. ft. of air passing and will permit the passage of 45 cu. ft. of free air through a brush renovator when operated at the end of 100 ft. of hose attached to the outlet at which the highest vacuum is maintained. This is a more uniform result, than was noted when 1¹ ₄-in. hose was used. The maximum horse power which will be required at the separator will now be 18.6 and the maximum displacement in the exhauster will be 740 cu. ft. It is, therefore, evident that, where very long runs of piping are necessary and where 100 ft. of hose will always be necessary, the use of 1-in. hose will require less power and a smaller displacement exhauster than would be required with 1¹ ₄-in. hose, without affecting the efficiency of the cleaning operations, and at the same time rendering the operation of the renovators on extreme ends of the system more uniform. The example cited in Figs. 63 and 64 is not by any means an extreme case to be met in cleaning systems for car yards, and the larger the system the greater will be the economy obtained with 1-in. hose. Such conditions, however, are confined almost entirely to layouts of this character and will seldom be met in layouts within any single building. This is fortunate, as the train cleaning is practically the only place where the use of 100 ft. of hose can be assured at all times. Very tall buildings offer a similar condition although the laterals are now vertical and can be kept large enough to sufficiently reduce the friction without danger of deposit of dirt in them, and the horizontal branches will be short and also large enough to keep the friction within reasonable limits without danger of deposit of dust."


"""

cleaned_text = remove_table_from_text(text_input)
print(cleaned_text)
