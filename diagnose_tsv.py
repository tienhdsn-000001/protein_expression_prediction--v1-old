#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_tsv.py

A utility script to debug issues with parsing large, complex TSV files.

This script reads a given TSV file line-by-line and checks for two common
problems that cause pandas `read_csv` to fail silently or truncate data:
1.  Inconsistent column counts between the header and subsequent rows.
2.  CSV parsing errors, such as unclosed quotation marks.

It will print the line number and content of the first problematic row it
encounters.
"""

import csv
import sys

# Python's csv module can sometimes hit a limit with very long fields (like DNA
# sequences). We increase the limit to prevent this from being a source of error.
# Use sys.maxsize for the maximum possible limit.
csv.field_size_limit(sys.maxsize)

FILE_TO_CHECK = "Ecoli_Annotation_v2.tsv"
HEADER_COLUMN_COUNT = 0

print(f"--- Starting Diagnostic Scan of: {FILE_TO_CHECK} ---")

try:
    with open(FILE_TO_CHECK, "r", encoding="utf-8", newline="") as f:
        # Use Python's built-in csv reader, which gives us fine-grained control.
        # We configure it to behave like our pandas parser: tab-delimited,
        # minimal quoting with double quotes.
        reader = csv.reader(f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        # Read the header row to establish the expected number of columns.
        header = next(reader)
        HEADER_COLUMN_COUNT = len(header)
        print(f"Header found with {HEADER_COLUMN_COUNT} columns.")
        print("-" * 30)

        # Iterate through the rest of the file, checking each row.
        for i, row in enumerate(reader, 2):  # Start line count from 2
            current_column_count = len(row)
            
            # Check if the number of columns in this row matches the header.
            if current_column_count != HEADER_COLUMN_COUNT:
                print(f"\n!!! COLUMN COUNT MISMATCH FOUND at line {i} !!!")
                print(f"    Expected {HEADER_COLUMN_COUNT} columns, but found {current_column_count}.")
                print("    This is the most likely cause of the parsing failure.")
                print("\n    Problematic row content (first 300 characters):")
                print(f"    {str(row)[:300]}")
                
                print("\n    Please inspect this row in the source TSV file for extra tabs or formatting errors.")
                sys.exit(1) # Exit after finding the first error.

    print("\n--- Diagnostic Scan Complete ---")
    print(f"Successfully parsed all rows. No column count mismatches were found.")
    print("If pandas is still failing, the issue might be a more subtle quoting error.")

except FileNotFoundError:
    print(f"ERROR: The file '{FILE_TO_CHECK}' was not found.")
except csv.Error as e:
    # This block will catch low-level parsing errors, like an unclosed quote.
    print(f"\n!!! CSV PARSING ERROR encountered !!!")
    print(f"    Error message: {e}")
    print(f"    This error often occurs due to an unclosed quote or a special character that breaks the parser.")
    print(f"    The error likely originates in the data on or just before line number {reader.line_num}.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
