import pandas as pd
import re
# import matplotlib

def create_df():
    """
    Reads the CSV of the Survey Results as well as the mappings for each machines, updates 
    the index and columns on the maps to match vending machine slots such as A1.

    Returns:
        Tuple[Dataframe, Dataframe, Dataframe, Dataframe, Dataframe, Dataframe]:
        The dataframes for the survey results and 5 vending machine maps.
    
    """
    df = pd.read_csv("./Survey Results.csv")
    df.index = range(1,len(df) + 1)

    map_1 = pd.read_csv("./Machine 1.csv", names = ['A', 'B', 'C', 'D', 'E', 'F'])
    map_1.index = range(1,11)
    map_1.loc[11] = {'A': "Large", 'B': "Large", "C": "Small", "D": "Small", "E": "Large", "F": "Large"}

    map_2 = pd.read_csv("./Machine 2.csv", names = ['A', 'B', 'C', 'D', 'E', 'F'])
    map_2.index = range(1,11)
    map_2.loc[11] = {'A': "Large", 'B': "Large", "C": "Large", "D": "Large", "E": "Small", "F": "Small"}

    map_3 = pd.read_csv("./Machine 3.csv", names = ['A', 'B', 'C', 'D', 'E', 'F'])
    map_3.index = range(1,11)
    map_3.loc[11] = {'A': "Small", 'B': "Small", "C": "Large", "D": "Large", "E": "Large", "F": "Large"}

    map_4 = pd.read_csv("./Machine 4.csv", names = ['A', 'B', 'C', 'D', 'E', 'F'])
    map_4.index = range(1,11)
    map_4.loc[11] = {'A': "Large", 'B': "Small", "C": "Large", "D": "Large", "E": "Large", "F": "Small"}

    map_5 = pd.read_csv("./Machine 5.csv", names = ['A', 'B', 'C', 'D', 'E', 'F'])
    map_5.index = range(1,11)
    map_5.loc[11] = {'A': "Large", 'B': "Large", "C": "Large", "D": "Small", "E": "Large", "F": "Small"}

    return df, map_1, map_2, map_3, map_4, map_5

def preprocessing(df, map_1, map_2, map_3, map_4, map_5):
    """
    Combines the dataframe with the mappings to a new dataframe containing all necessary 
    and imporant information for processing.

    Args:
        df: The survey results
        map_1: Vending Machine 1's mapping
        map_2: Vending Machine 2's mapping
        map_3: Vending Machine 3's mapping
        map_4: Vending Machine 4's mapping
        map_5: Vending Machine 5's mapping

    Returns:
        Dataframe: A new dataframe containing the respondent, the slot they chose, the machine 
        of the slot, the rank they gave the slot, the name of the item at the slot, and the item
        size.
    """
    new_df = pd.DataFrame(columns = ["respondent", "slot", "machine", "rank", "item_name", "size"])

    maps = {
        1: map_1,
        2: map_2,
        3: map_3,
        4: map_4,
        5: map_5
    }

    for index, row in df.iterrows():
        for col in df:
            if col == "Timestamp":
                continue
            slot = row[col]
            machine = int(re.search(r"\d", col).group())
            rank = re.search(r"\d$", col).group()

            slot_col = str(slot[0])
            slot_row = int(slot[1:])

            map = maps[machine]

            item_name = map.loc[slot_row, slot_col]

            size = map.loc[11, slot_col]

            new_df.loc[len(new_df)] = {"respondent": index, "slot": slot, "machine": machine, "rank": rank, "item_name": item_name, "size": size}

    return new_df

def main(debug = False):

    if (debug): print("Creating dataframe and maps")
    df, map_1, map_2, map_3, map_4, map_5 = create_df()
    if (debug): print("Sucessfully created dataframe and maps")

    if (debug): print("Preprocessing")
    df = preprocessing(df, map_1, map_2, map_3, map_4, map_5)
    if (debug): print("Preprocessing complete")

    if (debug):
        print(df.head(5))
        print(map_1)
        print(map_2)
        print(map_3)
        print(map_4)
        print(map_5)

    

    

if __name__ == "__main__":
    main(debug=False)