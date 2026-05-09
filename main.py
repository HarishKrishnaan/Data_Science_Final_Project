import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# import matplotlib

def create_df():
    """
    Reads the CSV of the Survey Results as well as the mappings for each machines and a mapping for the snack type of items, updates 
    the index and columns on the maps to match vending machine slots such as A1.

    Returns:
        Tuple[Dataframe, Dataframe, Dataframe, Dataframe, Dataframe, Dataframe, Dataframe]:
        The dataframes for the survey results, 5 vending machine maps, and item types.
    
    """
    df = pd.read_csv("./Survey Results.csv")
    # For respondents
    df.index = range(1,len(df) + 1)

    # The 11'th row will determine whether the column was for large items or small items
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

    item_type = pd.read_csv("./item_type.csv")

    return df, map_1, map_2, map_3, map_4, map_5, item_type

def preprocessing(df, map_1, map_2, map_3, map_4, map_5, item_type):
    """
    Using the dataframe from the respondents, process all the selected data into a list, then for every respondents in every machine
    create a table describing whether the user selected the item or not.

    Args:
        df: The survey results
        map_1: Vending Machine 1's mapping
        map_2: Vending Machine 2's mapping
        map_3: Vending Machine 3's mapping
        map_4: Vending Machine 4's mapping
        map_5: Vending Machine 5's mapping
        item_type: The item types

    Returns:
        Dataframe: A new dataframe containing the respondent, the slot they chose, the machine 
        of the slot, the rank they gave the slot, the name of the item at the slot, and the item
        size.
    """
    new_rows = []

    maps = {
        1: map_1,
        2: map_2,
        3: map_3,
        4: map_4,
        5: map_5
    }

    # Checking for NA's 
    df.dropna(inplace=True)
    df = df.drop(columns = ["Timestamp"])

    selected_lookup = {}

    # Loop through respondents and add them to the lookup
    for index, row in df.iterrows():
        for col in df.columns:

            slot = row[col]
            machine = int(re.search(r"\d", col).group())

            slot_col = slot[0]
            slot_row = int(slot[1:])

            item_name = maps[machine].loc[slot_row, slot_col]

            key = (index, machine)
            selected_lookup.setdefault(key, set()).add(item_name)

    # 77 respondents
    for index, row in df.iterrows():
        # 5 machines
        for machine_id, map_df in maps.items():

            selected_items = selected_lookup.get((index, machine_id), set())

            # 40 items
            for c in range(1, 11):
                for r in ['A','B','C','D','E','F']:

                    item_name = map_df.loc[c, r]
                    if (pd.isna(item_name)): continue

                    size = map_df.loc[11, r]

                    row_dic = {'A':1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}

                    center_points = [(3,5), (3,6), (4,5), (4,6)]

                    type_ = item_type[item_type["item_name"] == item_name]["type"].values[0]

                    new_rows.append({
                        "respondent": index,
                        "machine": machine_id,
                        "item_name": item_name,
                        "size": size,
                        "row": row_dic[r],
                        "col": c,
                        "edge": (r == 'A' or r == 'F' or c == 1 or c == 10),

                        # Idk I just arbitrarily picked B as eye level
                        "eye_level": (r == 'B'),

                        # Center is described at C5,C6,D5,D6, and the distance to it is the minimum value
                        "center_distance": min(
                            np.sqrt((row_dic[r] - cr)**2 + (c - cc)**2) for cr, cc in center_points
                            ),

                        "type": type_,
                        "selected": 1 if item_name in selected_items else 0
                    })

    return pd.DataFrame(new_rows)

def split(df, train_size=0.2):
    """
    Splits the data into testing and training data, setting categorical data into numerical data.

    Args:
        df: The dataframe to split

    Returns:
        Tuple[Dataframe, Dataframe, Dataframe, Dataframe, Dataframe, Dataframe]:
            The Training and Testing dataframes
    """
    df_encoded = pd.get_dummies(df, columns=["size", "type"])

    # df_encoded = df_encoded.drop(columns=["item_name"])
    df_encoded = df_encoded.drop(columns=["item_name", "respondent", "machine", "row", "col"])

    x = df_encoded.drop(columns=["selected"])
    y = df_encoded["selected"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=42)

    return x, y, x_train, x_test, y_train, y_test

def random_forest(x_train, y_train, n = 200):
    """
    Fits the random forest with the training dataframes

    Args:
        x_train: The not-selected training data
        y_train: The selected training data

    Returns:
        RandomForestClassifier: The random forest that was fitted with the training dataframes. 
    """
    rf = RandomForestClassifier(
        n_estimators=n,
        random_state=42
    )

    rf.fit(x_train, y_train)

    return rf

def predictions(rf, x_test):
    """
    Predictions on the random forests

    Args:
        rf: The random forest
        x_test: The non-selected testing data

    Returns:
        y_pred: The predictions from the forest
        y_prob: The probability from the forest
    """

    y_pred = rf.predict(x_test)
    y_prob = rf.predict_proba(x_test)[:, 1]

    return y_pred, y_prob

def evaluation(y_test, y_pred, display = False):
    """
    Evaluates and prints the accuracy score from the forest

    Args: 
        y_test: The selected testing data
        y_pred: The predictions from the forest

    Returns:
        accuracy: The accuracy of the forest
    """

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    if (display):
        print(accuracy)
        print(cm)

    return accuracy, cm

def size_loop(df, train_size = [0.8]):
    """
    Tests the random forest with different testing sizes and different n_sizes. Prints out a plot of the resulting accuracy

    Args:
        df: The pre-processed df to use
        train_size: A list of test sizes to try out, as a %
        n_size: A list of random forest estimators to try out
    """

    accuracy_list = []

    importance = pd.DataFrame()

    if (len(train_size) == 0): train_size.append(0.8)

    for train in train_size:
        x, y, x_train, x_test, y_train, y_test = split(df, train)

        importance["features"] = x.columns
        
        rf = random_forest(x_train, y_train, 200)

        y_pred, y_prob = predictions(rf, x_test)

        accuracy, cr = evaluation(y_test, y_pred)

        print(f"Training Size: {train * 100}% | Accuracy: {accuracy:.4f}")

        importance[f"importance.{train}"] = rf.feature_importances_

        # accuracy_list.append((accuracy, train))

    # df_results = pd.DataFrame(accuracy_list, columns=["accuracy","train_size"])

    print(importance)

    

def rank_weight(rank):
    rank_weight = {
        1: 3,
        2: 2,
        3: 1
    }
    try:
        return rank_weight[rank]
    except:
        return None

def main(debug = False):
    """
    Using a random forest algorithm, we should find the importance of the data based on the characteristics we give it.
    More characteristics in the data the more better results we get. 
    
    """
    if (debug): print("Creating dataframe and maps")
    df, map_1, map_2, map_3, map_4, map_5, item_type = create_df()
    if (debug): print("Sucessfully created dataframe and maps")

    if (debug):
        print(df.head(5))
        print(map_1)
        print(map_2)
        print(map_3)
        print(map_4)
        print(map_5)
        print(item_type.head(5))

    if (debug): print("Preprocessing")
    df_processed = preprocessing(df, map_1, map_2, map_3, map_4, map_5, item_type)
    if (debug): print("Preprocessing complete")

    if (debug): print(df_processed.head(5))

    if (debug): print("Splitting")
    x, y, x_train, x_test, y_train, y_test = split(df_processed, 0.2)
    if (debug): print("Splitting complete")

    if (debug): print("Fitting random forest")
    rf = random_forest(x_train, y_train)
    if (debug): print("Fitting random forest complete")

    if (debug): print("Predictions")
    y_pred, y_prob = predictions(rf, x_test)
    if (debug): print("Predicitons complete")

    if (debug): print("Evaluation")
    _, cm = evaluation(y_test, y_pred, True)
    if (debug): print("Evaluation complete")

    importance = pd.DataFrame({
        "feature": x.columns,
        "importance": rf.feature_importances_
    }).sort_values(by="importance", ascending=False)

    importance = importance.sort_values("importance", ascending=True)

    plt.figure(figsize=(8,6))
    plt.barh(importance["feature"], importance["importance"])
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix Heatmap")
    plt.show()

    plt.figure()
    plt.hist(y_prob, bins=20)
    plt.title("Prediction Probability Distribution")
    plt.xlabel("Predicted probability of selection")
    plt.ylabel("Count")
    plt.show()

    # df_copy = df_processed.copy()

    # expand_size = {
    #     1: (1, 2),
    #     2: (3, 4),
    #     3: (5, 6),
    #     4: (7, 8),
    #     5: (9, 10)
    # }

    # expanded_rows = []

    # for _, row in df_copy.iterrows():

    #     positions = [(row["row"], row["col"])]

    #     if row["size"] == "Large":
    #         col = row["col"]

    #         for k, (c1, c2) in expand_size.items():
    #             if col == k:
    #                 positions = [(row["row"], c1), (row["row"], c2)]
    #                 break

    #     for r, c in positions:
    #         new_row = row.copy()
    #         new_row["row"] = r
    #         new_row["col"] = c
    #         expanded_rows.append(new_row)

    # pivot = pd.DataFrame(expanded_rows)

    # print(pivot.head(10))

    # heatmap_data = pivot.groupby(["row", "col"])["selected"].mean().unstack()

    # plt.figure(figsize=(8,5))
    # sns.heatmap(heatmap_data, cmap="YlOrRd", annot=False)
    # plt.title("Selection Rate by Machine Position (Expanded Large Items)")
    # plt.xlabel("Column")
    # plt.ylabel("Row")
    # plt.show()






    # Todo Graphics
    # df["rank_weight"] = df["rank"].map(rank_weight)

    # if (debug): print(df.head(5))

    # df_summary = df.groupby("item_name").agg(
    #     count = ("item_name", "count"),
    #     rank_score = ("rank_weight", "sum")
    # ).sort_index()

    # if (debug): print(df_summary.head(5))

    # df_combined_map = pd.concat([map_1.drop(11),map_2.drop(11),map_3.drop(11),map_4.drop(11),map_5.drop(11)]).stack().reset_index(drop=True).to_frame(name="item_name")

    # df_map_summary = df_combined_map.groupby(by="item_name").agg(
    #     appearance = ("item_name", "count")
    # )

    # if (debug): print(df_map_summary.head(5))

    # df_summary = pd.merge(df_summary, df_map_summary, on="item_name", how="right").fillna(0)

    # if (debug): print(df_summary.head(11))


    

    

if __name__ == "__main__":
    main(debug=True)