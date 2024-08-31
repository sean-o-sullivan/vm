import re
table_patterns = [
    re.compile(
        r'(?sm)'
        r'(?:^[A-Z\s.,\'\-]+$\n)?'  
        r'(?:^[=\-+|_]{2,}.*$\n)+'  
        r'(?:^.*?(?:\||\+).*$\n)+'  
        r'(?:^[=\-+|_]{2,}.*$\n)?'  
        r'(?=\n*(?:[A-Z][a-z]|\Z))'  
    ),
    re.compile(
        r'(?sm)'
        r'(?:^[A-Z\s.,\'\-]+$\n)?'  
        r'(?:^[\s\d\w]+(?:\s{2,}[\s\d\w]+)+$\n)+'  
        r'(?=\n*(?:[A-Z][a-z]|\Z))'  
    )
]

def remove_tables(text):
    
    for pattern in table_patterns:
        text = pattern.sub('', text)
    
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

#taken from the project gutenberg corpus, fortunately the BAWE corpus has none of these silly tables
test_cases = [ 
    
      {
        "name": "Simple table with pipes",
        "input": """
Before table

| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
| Data 3   | Data 4   |

After table
""",
        "expected_output": """
Before table

After table
"""
    },
    {
        "name": "Table with full caps title",
        "input": """
TEXT BEFORE

TABLE 1.--SAMPLE DATA
=====================
| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
| Data 3   | Data 4   |

TEXT AFTER
""",
        "expected_output": """
TEXT BEFORE

TEXT AFTER
"""
    },
    {
        "name": "Table with aligned columns using spaces",
        "input": """
Before table

Column 1    Column 2    Column 3
Data 1      Data 2      Data 3
Data 4      Data 5      Data 6

After table
""",
        "expected_output": """
Before table

After table
"""
    },
    {
        "name": "Multiple tables",
        "input": """
Table 1:

| A | B |
|---|---|
| 1 | 2 |

Some text

Table 2:
Column 1    Column 2
Data 1      Data 2
""",
        "expected_output": """
Some text
"""
    },
    {
        "name": "No tables",
        "input": "This is a text without any tables.",
        "expected_output": "This is a text without any tables."
    },
    {
        "name": "Progress and Methods of Excavation Table",
        "input": """TABLE 1.--PROGRESS AND METHODS OF EXCAVATION IN GOOD GROUND. THIRTY-THIRD STREET. ============================================================ 1 2 3 -----------------------------+--------+--------------------+ Type of excavation. Tunnels. Worked from: -----------------------------+--------+--------------------+ Full-sized single tunnel B 1st Ave. shaft. Full-sized single tunnel A 1st Ave. shaft. Full-sized twin tunnel A and B 1st Ave. shaft. Full-sized twin tunnel A and B Intermediate shaft. (West of shaft.) Full-sized twin tunnel A and B Intermediate shaft. (East of shaft.) Full-sized twin tunnel A and B Intermediate shaft. (East of shaft.) Exploration drift A and B Intermediate shaft. (West of shaft.) Twin tunnel. Enlargement A and B West shaft. of exploration drift (East of shaft.) =============================+========+===================== ====================================================================== 4 5 6 7 ----------------------------------+--------+------------+------------+ Length Average DATES. Time tunnel advance ---------------------------------- elapsed, excavated, per day, in in in From To days. linear feet. linear feet. ----------------------------------+--------+------------+------------+ Feb. 28, 1906. May 12, 1906. 74 346 4.7 Feb. 28, 1906. Apr. 30, 1906. 62 255 4.1 Aug. 23, 1906. Jan. 5, 1907. 136 789 5.8 Apr. 4, 1906. Oct. 31, 1906. 210 730 3.5 Apr. 4, 1906. Oct. 31, 1906. 210 783 3.7 Nov. 1, 1906. Dec. 26, 1906. 56 311 5.5 Mar. 1, 1907. July 23, 1907. 145 947 6.5 Sept. 6, 1907. Dec. 4, 1907. 89 603 6.8 ===============+==================+========+============+============= ===================================================== 8 ----------------------------------------------------- Methods and conditions. ----------------------------------------------------- Top heading and bench. Muck loaded by hand. "" "" "" "" "" "" "" "" Top full-width heading and bench. Muck loaded by steam shovel. Working exclusively on this heading. Top center heading and bench. Muck loaded by steam shovel. Working alternately in headings east and west of the shaft. Top center heading and bench. Muck loaded by steam shovel. Working alternately in headings east and west of the shaft. Top full-width heading and bench. Muck loaded by steam shovel working exclusively on this heading. Exploration drift about 9 ft. by 12 ft. Mucking by hand. Fourteen timber bents were placed in March, and seven in April, 1907. Drift excavated to full width and bench. Muck loaded by steam shovel. =====================================================

Text after table.""",
        "expected_output": "Text after table."
    },
    {
        "name": "Continental Rise Table",
        "input": """TABLE 2.--General characteristics of the continental rise northwest Africa Sector Values measured from Profiles E-11 to E-21 ==================================================================== Depth Segment Upper edge Lower edge Gradient Width ==================================================================== Upper continental rise 1 1200 200 1500 200 1:90 30 30 10? 2 1500 200 1600 200 1:200 100 30 15? 3 1600 200 1800 100 1:100 50 25 15? Lower continental rise 1 1800 100 2000 100 1:400 200 75 50 2 2000 100 2000 100 1:1500 500 150 50 3 2000 100 2700 100 1:500 200 200 50 Abyssal plain 2700 100 3000 75 1:1250 250 200 50 --------------------------------------------------------------------

Text after table.""",
        "expected_output": "Text after table."
    },
    {
        "name": "Death Rate Table",
        "input": """as the following table indicates: DEATH RATE OF JAPANESE IN CALIFORNIA. ======================== Year. Number. Percentage of Death per 1000. ----- ------- ---------- 1910 440 10.64% 1911 472 ..... 1912 524 ..... 1913 613 ..... 1914 628 ..... 1915 663 ..... 1916 739 ..... 1917 910 ..... 1918 1150 ..... 1919 1360 20.00% ======================== The rate of death per one thousand population increased twice during the past ten years.""",
        "expected_output": """as the following table indicates: 

The rate of death per one thousand population increased twice during the past ten years."""
    },
    {
        "name": "Japanese Population Table",
        "input": """JAPANESE POPULATION IN THE UNITED STATES AND CALIFORNIA. ===================================================================== Census. Japanese in Decennial Rate of Rate of Percentage of Continental Increase of Decennial Decennial Japanese in United Japanese in Increase. Increase of California to States. Continental Japanese in entire Japanese United California. population of States. United States. ------------------- ----------- --------- ----------- --------------- 1880 148 ...... ....... ...... 58.1% 1890 2,039 1,891 1,277.7% 1234.0% 56.2% 1900 24,326 22,287 1,093.0% 785.0% 41.7% 1910 72,157 47,831 196.6% 307.3% 57.3% 1920 119,207 47,050 65.2% 69.7% 58.8% =====================================================================

Text after table.""",
        "expected_output": "Text after table."
    }
]

def run_tests():
    passed = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"Running test case {i}: {test_case['name']}")
        result = remove_tables(test_case['input'])
        expected = test_case['expected_output'].strip()
        
        if result == expected:
            print("PASSED")
            passed += 1
        else:
            print("FAILED")
            print("Expected:")
            print(expected)
            print("Got:")
            print(result)
            failed += 1
        print("-" * 40)

    print(f"Test results: {passed} passed, {failed} failed")

if __name__ == "__main__":
    run_tests()
