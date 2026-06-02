import pandas as pd
import re

def remove_emojis(text):
    """Remove all emojis from text"""
    if pd.isna(text):
        return ""

    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        u"\U00002500-\U00002BEF"  
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  
        u"\u3030"
        "]+", flags=re.UNICODE
    )

    return emoji_pattern.sub(r'', str(text))

def merge_all_sheets(input_file, output_file):
    """
    Merge all 12 monthly sheets into single sheet with only text and username

    Parameters:
    -----------
    input_file : str
        Path to input Excel file with 12 monthly sheets
    output_file : str
        Path to output Excel file with merged data
    """

    print("=" * 60)
    print("📊 MERGING ALL MONTHLY SHEETS")
    print("=" * 60)

    try:
        
        print(f"📂 Reading file: {input_file}")
        excel_file = pd.ExcelFile(input_file)
        sheet_names = excel_file.sheet_names

        print(f"📋 Found {len(sheet_names)} sheets")

        all_data = []

        for sheet_name in sheet_names:
            print(f"\n🔄 Processing sheet: {sheet_name}")

            df = pd.read_excel(input_file, sheet_name=sheet_name)

            if df.empty:
                print(f"   ⚠️ Sheet '{sheet_name}' is empty")
                continue

            text_column = df.columns[0]
            username_column = df.columns[1] if len(df.columns) > 1 else None

            print(f"   Text column: '{text_column}'")
            print(f"   Username column: '{username_column}'" if username_column else "   No username column found")

            monthly_data = pd.DataFrame()

            monthly_data['text'] = df[text_column].apply(remove_emojis)

            if username_column:
                monthly_data['username'] = df[username_column]
            else:
                monthly_data['username'] = pd.NA

            monthly_data['month'] = sheet_name

            before_count = len(monthly_data)
            monthly_data = monthly_data[
                monthly_data['text'].notna() &
                (monthly_data['text'].astype(str).str.strip() != '')
            ]

            after_count = len(monthly_data)

            monthly_data = monthly_data.reset_index(drop=True)

            print(f"   📝 Rows: {before_count} → {after_count} (removed {before_count - after_count} empty rows)")

            all_data.append(monthly_data)

        if not all_data:
            print("\n❌ No valid data found in any sheet!")
            return False
        combined_df = pd.concat(all_data, ignore_index=True)

        combined_df['text'] = combined_df['text'].astype(str).str.strip()

        combined_df['username'] = combined_df['username'].fillna('')

        print(f"\n📊 COMBINED DATA SUMMARY:")
        print(f"   Total rows: {len(combined_df):,}")
        print(f"   Total months: {combined_df['month'].nunique()}")
        month_counts = combined_df['month'].value_counts().sort_index()
        print(f"\n📅 ROWS PER MONTH:")
        for month, count in month_counts.items():
            print(f"   {month:15}: {count:5,} rows")
        print(f"\n💾 Saving merged data to: {output_file}")
        combined_df.to_excel(output_file, index=False, sheet_name='All_Months_Merged')

        csv_file = output_file.replace('.xlsx', '.csv')
        combined_df.to_csv(csv_file, index=False)

        print(f"📁 Also saved as CSV: {csv_file}")

        print("\n" + "=" * 60)
        print("✅ MERGE COMPLETE!")
        print("=" * 60)
        print(f"\n📋 SAMPLE DATA (first 5 rows):")
        print("-" * 60)
        print(combined_df.head().to_string())

        return combined_df

    except FileNotFoundError:
        print(f"❌ Error: File '{input_file}' not found!")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def merge_all_sheets_with_backup(input_file, output_file):
    """
    Merge all sheets into single sheet, but keep original sheets too
    """

    print("=" * 60)
    print("📊 MERGING SHEETS WITH BACKUP")
    print("=" * 60)

    try:
        excel_file = pd.ExcelFile(input_file)
        sheet_names = excel_file.sheet_names

        all_data = []

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:

            for sheet_name in sheet_names:
                df = pd.read_excel(input_file, sheet_name=sheet_name)

                if not df.empty:
                    if len(df.columns) > 0:
                        text_col = df.columns[0]
                        df[text_col] = df[text_col].apply(remove_emojis)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            for sheet_name in sheet_names:
                df = pd.read_excel(input_file, sheet_name=sheet_name)

                if df.empty:
                    continue
                if len(df.columns) >= 1:
                    text_col = df.columns[0]
                    username_col = df.columns[1] if len(df.columns) > 1 else None

                    monthly_data = pd.DataFrame()
                    monthly_data['text'] = df[text_col].apply(remove_emojis)

                    if username_col:
                        monthly_data['username'] = df[username_col]
                    else:
                        monthly_data['username'] = ''

                    monthly_data['month'] = sheet_name
                    monthly_data = monthly_data[
                        monthly_data['text'].notna() &
                        (monthly_data['text'].astype(str).str.strip() != '')
                    ]

                    all_data.append(monthly_data)
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df['text'] = combined_df['text'].astype(str).str.strip()
                combined_df['username'] = combined_df['username'].fillna('')

                combined_df.to_excel(writer, sheet_name='All_Months_Merged', index=False)

                print(f"✅ Saved {len(sheet_names)} original sheets + 1 merged sheet")
                print(f"📊 Total merged rows: {len(combined_df):,}")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    input_excel = "dec.xlsx"  
    output_excel = "merged_comments.xlsx"   

    print("🚀 STARTING SHEET MERGING PROCESS")
    print("-" * 40)
    print(f"Input:  {input_excel}")
    print(f"Output: {output_excel}")
    print("-" * 40)
    print("\n📋 OPTION 1: Simple merge to single sheet")
    result = merge_all_sheets(input_excel, output_excel)

    if result is not False:
        print("\n🎉 PROCESS COMPLETED SUCCESSFULLY!")
        print(f"\n📄 FINAL COLUMNS IN OUTPUT:")
        print("   1. text     - Cleaned text (emojis removed)")
        print("   2. username - Username from second column")
        print("   3. month    - Original sheet name")