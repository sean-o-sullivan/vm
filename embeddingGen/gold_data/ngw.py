def process_text(text):
    #I need a far more robust approach to finding these stupid tables, otherwise gutenberg is very much unusable.......
    lines = text.split('\n')
    processed_lines = []
    in_table = False
    
    for line in lines:
        
        if '|' in line or line.strip().startswith('TABLE') or line.strip().isupper():
            in_table = True
        if in_table and line.strip() == '':
            in_table = False
        
        
        if in_table:
            processed_lines.append(line)
        else:
            
            processed_lines.append(line.strip())
    
    
    return '\n'.join(processed_lines)


test_cases = [
    ("Running test case 1: Simple table with pipes\nBefore table\n\n| Column 1 | Column 2 |\n|----------|----------|\n| Data 1   | Data 2   |\n\nAfter table", "Before table\n\n\n\nAfter table"),
    ("Running test case 2: Table with full caps title\nTEXT BEFORE\n\nTABLE 1.--SAMPLE DATA\n| Column 1 | Column 2 |\n|----------|----------|\n| Data 1   | Data 2   |\n\nTEXT AFTER", "TEXT BEFORE\n\n\n\nTEXT AFTER"),
    ("Running test case 3: Table with aligned columns using spaces\nBefore table\n\nColumn 1    Column 2\n--------    --------\nData 1      Data 2\n\nAfter table", "Before table\n\n\n\nAfter table"),
    ("Running test case 4: Multiple tables\nSome text\n\nTable 1:\n| A | B |\n|---|---|\n| 1 | 2 |\n\nTable 2:\n| X | Y |\n|---|---|\n| 3 | 4 |\n\nMore text", "Some text\n\n\n\nMore text"),
    ("Running test case 5: No tables\nThis is a test\nwith multiple lines\nbut no tables.", "This is a test\nwith multiple lines\nbut no tables."),
    ("Running test case 6: Progress and Methods of Excavation Table\nTABLE 1.--PROGRESS AND METHODS OF EXCAVATION IN GOOD GROUND. THIRTY-THIRD STREET.\n============================================================\n1                            2        3\n-----------------------------+--------+--------------------+\nType of excavation.          Tunnels. Worked from:\n-----------------------------+--------+--------------------+\nFull-sized single tunnel     B        1st Ave. shaft.\nFull-sized single tunnel     A        1st Ave. shaft.\nFull-sized twin tunnel       A and B  1st Ave. shaft.\nFull-sized twin tunnel       A and B  Intermediate shaft.\n                                      (West of shaft.)\nFull-sized twin tunnel       A and B  Intermediate shaft.\n                                      (East of shaft.)\nFull-sized twin tunnel       A and B  Intermediate shaft.\n                                      (East of shaft.)\nExploration drift            A and B  Intermediate shaft.\n                                      (West of shaft.)\nTwin tunnel. Enlargement     A and B  West shaft.\n of exploration drift                 (East of shaft.)\n=============================+========+=====================\n\nText after table.", "TABLE 1.--PROGRESS AND METHODS OF EXCAVATION IN GOOD GROUND. THIRTY-THIRD STREET.\n============================================================\n1                            2        3\n-----------------------------+--------+--------------------+\nType of excavation.          Tunnels. Worked from:\n-----------------------------+--------+--------------------+\nFull-sized single tunnel     B        1st Ave. shaft.\nFull-sized single tunnel     A        1st Ave. shaft.\nFull-sized twin tunnel       A and B  1st Ave. shaft.\nFull-sized twin tunnel       A and B  Intermediate shaft.\n                                      (West of shaft.)\nFull-sized twin tunnel       A and B  Intermediate shaft.\n                                      (East of shaft.)\nFull-sized twin tunnel       A and B  Intermediate shaft.\n                                      (East of shaft.)\nExploration drift            A and B  Intermediate shaft.\n                                      (West of shaft.)\nTwin tunnel. Enlargement     A and B  West shaft.\n of exploration drift                 (East of shaft.)\n=============================+========+=====================\n\nText after table."),
    ("Running test case 7: Continental Rise Table\nTABLE 2.--General characteristics of the continental rise northwest Africa\nSector Values measured from Profiles E-11 to E-21\n====================================================================\nDepth Segment    Upper edge    Lower edge    Gradient    Width\n====================================================================\nUpper continental rise\n    1            1200±200      1500±200      1:90        30±30±10?\n    2            1500±200      1600±200      1:200       100±30±15?\n    3            1600±200      1800±100      1:100       50±25±15?\nLower continental rise\n    1            1800±100      2000±100      1:400       200±75±50\n    2            2000±100      2000±100      1:1500      500±150±50\n    3            2000±100      2700±100      1:500       200±200±50\nAbyssal plain    2700±100      3000±75       1:1250      250±200±50\n--------------------------------------------------------------------\n\nText after table.", "TABLE 2.--General characteristics of the continental rise northwest Africa\nSector Values measured from Profiles E-11 to E-21\n====================================================================\nDepth Segment    Upper edge    Lower edge    Gradient    Width\n====================================================================\nUpper continental rise\n    1            1200±200      1500±200      1:90        30±30±10?\n    2            1500±200      1600±200      1:200       100±30±15?\n    3            1600±200      1800±100      1:100       50±25±15?\nLower continental rise\n    1            1800±100      2000±100      1:400       200±75±50\n    2            2000±100      2000±100      1:1500      500±150±50\n    3            2000±100      2700±100      1:500       200±200±50\nAbyssal plain    2700±100      3000±75       1:1250      250±200±50\n--------------------------------------------------------------------\n\nText after table."),
    ("Running test case 8: Death Rate Table\nas the following table indicates: \nDEATH RATE OF JAPANESE IN CALIFORNIA.\n========================\nYear. Number. Percentage of Death per 1000.\n----- ------- ----------\n1910   440     10.64%\n1911   472     .....\n1912   524     .....\n1913   613     .....\n1914   628     .....\n1915   663     .....\n1916   739     .....\n1917   910     .....\n1918  1150     .....\n1919  1360     20.00%\n========================\nThe rate of death per one thousand population increased twice during the past ten years.", "as the following table indicates:\nDEATH RATE OF JAPANESE IN CALIFORNIA.\n========================\nYear. Number. Percentage of Death per 1000.\n----- ------- ----------\n1910   440     10.64%\n1911   472     .....\n1912   524     .....\n1913   613     .....\n1914   628     .....\n1915   663     .....\n1916   739     .....\n1917   910     .....\n1918  1150     .....\n1919  1360     20.00%\n========================\nThe rate of death per one thousand population increased twice during the past ten years."),
    ("Running test case 9: Japanese Population Table\nJAPANESE POPULATION IN THE UNITED STATES AND CALIFORNIA.\n=====================================================================\nCensus.  Japanese in  Decennial  Rate of      Rate of    Percentage of\n         Continental  Increase   Decennial    Decennial  Japanese in\n         United       of         Increase.    Increase   California to\n         States.      Japanese                of         entire Japanese\n                      in                      Japanese   population of\n                      Continental             in         United States.\n                      United                  California.\n                      States.\n-------------------  -----------  ---------  -----------  ---------------\n1880         148        ......     .......      ......        58.1%\n1890       2,039        1,891     1,277.7%    1234.0%         56.2%\n1900      24,326       22,287     1,093.0%     785.0%         41.7%\n1910      72,157       47,831       196.6%     307.3%         57.3%\n1920     119,207       47,050        65.2%      69.7%         58.8%\n=====================================================================\n\nText after table.", "JAPANESE POPULATION IN THE UNITED STATES AND CALIFORNIA.\n=====================================================================\nCensus.  Japanese in  Decennial  Rate of      Rate of    Percentage of\n         Continental  Increase   Decennial    Decennial  Japanese in\n         United       of         Increase.    Increase   California to\n         States.      Japanese                of         entire Japanese\n                      in                      Japanese   population of\n                      Continental             in         United States.\n                      United                  California.\n                      States.\n-------------------  -----------  ---------  -----------  ---------------\n1880         148        ......     .......      ......        58.1%\n1890       2,039        1,891     1,277.7%    1234.0%         56.2%\n1900      24,326       22,287     1,093.0%     785.0%         41.7%\n1910      72,157       47,831       196.6%     307.3%         57.3%\n1920     119,207       47,050        65.2%      69.7%         58.8%\n=====================================================================\n\nText after table.")
]


passed = 0
failed = 0

for i, (input_text, expected_output) in enumerate(test_cases, 1):
    result = process_text(input_text)
    if result == expected_output:
        print(f"test case {i}: {input_text.split(':')[1].strip()}\nPASSED")
        passed += 1
    else:
        print(f"test case {i}: {input_text.split(':')[1].strip()}\nFAILED")
        print(f"Expected:\n{expected_output}")
        print(f"Got :/:\n{result}")
        failed += 1
    print("-" * 40)

print(f"Test results: {passed} passed, {failed} failed")
