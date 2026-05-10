import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def create_df():
    """
    Reads the CSV of the Survey Results as well as the mappings for each machines and a mapping for the snack type of items, updates 
    the index and columns on the maps to match vending machine slots such as A1.

    Returns:
        Tuple[Dataframe, Dataframe, Dataframe, Dataframe, Dataframe, Dataframe, Dataframe]:
        The dataframes for the survey results, 5 vending machine maps, and item types.
    
    """
    df = pd.read_csv("./data/Survey Results.csv")
    # For respondents
    df.index = range(1,len(df) + 1)

    # The 11'th row will determine whether the column was for large items or small items
    map_1 = pd.read_csv("./data/Machine 1.csv", names = ['A', 'B', 'C', 'D', 'E', 'F'])
    map_1.index = range(1,11)
    map_1.loc[11] = {'A': "Large", 'B': "Large", "C": "Small", "D": "Small", "E": "Large", "F": "Large"}

    map_2 = pd.read_csv("./data/Machine 2.csv", names = ['A', 'B', 'C', 'D', 'E', 'F'])
    map_2.index = range(1,11)
    map_2.loc[11] = {'A': "Large", 'B': "Large", "C": "Large", "D": "Large", "E": "Small", "F": "Small"}

    map_3 = pd.read_csv("./data/Machine 3.csv", names = ['A', 'B', 'C', 'D', 'E', 'F'])
    map_3.index = range(1,11)
    map_3.loc[11] = {'A': "Small", 'B': "Small", "C": "Large", "D": "Large", "E": "Large", "F": "Large"}

    map_4 = pd.read_csv("./data/Machine 4.csv", names = ['A', 'B', 'C', 'D', 'E', 'F'])
    map_4.index = range(1,11)
    map_4.loc[11] = {'A': "Large", 'B': "Small", "C": "Large", "D": "Large", "E": "Large", "F": "Small"}

    map_5 = pd.read_csv("./data/Machine 5.csv", names = ['A', 'B', 'C', 'D', 'E', 'F'])
    map_5.index = range(1,11)
    map_5.loc[11] = {'A': "Large", 'B': "Large", "C": "Large", "D": "Small", "E": "Large", "F": "Small"}

    item_type = pd.read_csv("./data/item_type.csv")

    return df, map_1, map_2, map_3, map_4, map_5, item_type

def preprocessing(df, map_1, map_2, map_3, map_4, map_5, item_type):
    """
    Using the dataframe from the respondents, process all the selected data into a list, then for every respondents in every machine
    create a table describing whether the user selected the item or not. Additionally, add in any important values such as type,
    location predicates, and others.

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
    df_encoded = df_encoded.drop(columns=["item_name", "respondent", "machine"])

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
        random_state=42,
        class_weight="balanced"
    )

    rf.fit(x_train, y_train)

    return rf

def logistic_regression(x_train, y_train, iter = 1000):
    """
    Fits a logistic regression model with the training dataframes

    Args:
        x_train: The not-selected training data
        y_train: The selected training data

    Returns:
        LogisticRegression: The logistic regression
    """

    lr = LogisticRegression(
        max_iter=iter,
        random_state=42,
        class_weight="balanced"
        )

    lr.fit(x_train, y_train)

    return lr

def decision_tree(x_train, y_train, depth = 5):
    """
    Fits a decision tree model with the training dataframes

    Args:
        x_train: The not-selected training data
        y_train: The selected training data

    Returns:
        DecisionTreeClassifier: The decision tree
    """
    dt = DecisionTreeClassifier(
        random_state=42,
        max_depth=depth,
        class_weight="balanced"
    )

    dt.fit(x_train, y_train)

    return dt

def predictions(model, x_test):
    """
    Predictions on the model

    Args:
        model: The model to use
        x_test: The non-selected testing data

    Returns:
        y_pred: The predictions from the forest
        y_prob: The probability from the forest
    """

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

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
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    if (display):
        print(accuracy)
        print(cm)

    return accuracy, precision, recall, f1, cm

def hyperparameter_experiment(df_processed):
    """
    Tests the three models with different n, iteration, and depths

    Args:
        df_processed: The processed dataframe to use

    Returns:
        Tuple(List, List, List)
        

    """
    rf_results = []
    lr_results = []
    dt_results = []

    x, y, x_train, x_test, y_train, y_test = split(df_processed, 0.8)

    # Random Forest: number of trees
    for n in [50, 100, 200, 300, 500]:
        rf = RandomForestClassifier(n_estimators=n, random_state=42)
        rf.fit(x_train, y_train)
        rf_pred = rf.predict(x_test)
        rf_results.append((n, accuracy_score(y_test, rf_pred)))

    # Logistic Regression: max iterations
    for max_iter in [100, 250, 500, 1000, 2000]:
        lr = LogisticRegression(max_iter=max_iter)
        lr.fit(x_train, y_train)
        lr_pred = lr.predict(x_test)
        lr_results.append((max_iter, accuracy_score(y_test, lr_pred)))

    # Decision Tree: max depth
    for depth in [1, 2, 3, 5, 10, 15, 20]:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(x_train, y_train)
        dt_pred = dt.predict(x_test)
        dt_results.append((depth, accuracy_score(y_test, dt_pred)))

    return rf_results, lr_results, dt_results

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

    # Random Forest
    if (debug): print("Fitting random forest")
    rf = random_forest(x_train, y_train)
    if (debug): print("Fitting random forest complete")

    if (debug): print("Random forest predictions")
    rf_pred, rf_prob = predictions(rf, x_test)
    if (debug): print("Random forest predicitons complete")

    if (debug): print("Random forest evaluation")
    rf_accuracy, rf_precision, rf_recall, rf_f1, rf_cm = evaluation(y_test, rf_pred, True)
    if (debug): print("Random forest evaluation complete")

    # Logistic regression
    if (debug): print("Fitting logistic regression")
    lr = logistic_regression(x_train, y_train)
    if (debug): print("Fitting logistic regression complete")

    if (debug): print("Logistic regression predictions")
    lr_pred, lr_prob = predictions(lr, x_test)
    if (debug): print("Logistic regression predictions complete")

    if (debug): print("Logistic regression evaluation")
    lr_accuracy, lr_precision, lr_recall, lr_f1, lr_cm = evaluation(y_test, lr_pred, True)
    if (debug): print("Logistic regression evaluation complete")

    # Decision Tree
    if (debug): print("Decision tree regression")
    dt = decision_tree(x_train, y_train)
    if (debug): print("Fitting decision tree complete")

    if (debug): print("Decision tree predictions")
    dt_pred, dt_prob = predictions(dt, x_test)
    if (debug): print("Decision tree predictions complete")

    if (debug): print("Decision tree evaluation")
    dt_accuracy, dt_precision, dt_recall, dt_f1, dt_cm = evaluation(y_test, dt_pred, True)
    if (debug): print("Decision tree evaluation complete")

    rf_mean_prob = rf_prob.mean()
    lr_mean_prob = lr_prob.mean()
    dt_mean_prob = dt_prob.mean()

    # Graphs

    metrics = pd.DataFrame({
        "Model": ["Random Forest", "Logistic Regression", "Decision Tree"],
        "Accuracy": [rf_accuracy, lr_accuracy, dt_accuracy],
        "Precision": [rf_precision, lr_precision, dt_precision],
        "Recall": [rf_recall, lr_recall, dt_recall],
        "F1-Score": [rf_f1, lr_f1, dt_f1],
    })

    print(metrics)

    # Random Forest
    importance_rf = pd.DataFrame({
        "feature": x.columns,
        "importance": rf.feature_importances_
    })
    
    importance_rf = importance_rf.sort_values("importance", ascending=True)

    plt.figure(figsize=(8,6))
    plt.barh(importance_rf["feature"], importance_rf["importance"])
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

    sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Random Forest Confusion Matrix Heatmap")
    plt.show()

    plt.figure()
    plt.hist(rf_prob, bins=20)
    plt.title("Random Forest Prediction Probability Distribution")
    plt.xlabel("Predicted probability of selection")
    plt.ylabel("Count")
    plt.show()

    # Logistic Regression
    importance_lr = pd.DataFrame({
        "feature": x.columns,
        'coefficient': lr.coef_[0],
        'odds_ratio': np.exp(lr.coef_[0])
    })

    importance_lr = importance_lr.sort_values("coefficient", ascending=True)

    plt.figure(figsize=(8,6))
    plt.barh(importance_lr["feature"], importance_lr["coefficient"])
    plt.title("Logistic Regression Coefficient Importance")
    plt.xlabel("Importance")
    plt.ylabel("Coefficient")
    plt.show()

    sns.heatmap(lr_cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Logistic Regression Confusion Matrix Heatmap")
    plt.show()

    plt.figure()
    plt.hist(lr_prob, bins=20)
    plt.title("Logistic Regression Prediction Probability Distribution")
    plt.xlabel("Predicted probability of selection")
    plt.ylabel("Count")
    plt.show()

    # Decision Tree
    importance_dt = pd.DataFrame({
        "feature": x.columns,
        "importance": dt.feature_importances_
    })

    importance_dt = importance_dt.sort_values("importance", ascending=True)

    plt.figure(figsize=(8,6))
    plt.barh(importance_dt["feature"], importance_dt["importance"])
    plt.title("Decision Tree Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

    sns.heatmap(dt_cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Decision Tree Confusion Matrix Heatmap")
    plt.show()

    plt.figure()
    plt.hist(dt_prob, bins=20)
    plt.title("Decision Tree Prediction Probability Distribution")
    plt.xlabel("Predicted probability of selection")
    plt.ylabel("Count")
    plt.show()

    models = ["Random Forest", "Logistic Regression", "Decision Tree"]

    accuracy_scores = [rf_accuracy, lr_accuracy, dt_accuracy]
    mean_probs = [rf_mean_prob, lr_mean_prob, dt_mean_prob]

    plt.figure(figsize=(8, 5))
    x = np.arange(len(models))
    width = 0.35

    bars1 = plt.bar(x - width/2, accuracy_scores, width, label="Accuracy")
    bars2 = plt.bar(x + width/2, mean_probs, width, label="Mean Probability")

    plt.bar_label(bars1, labels=[f"{v:.4f}" for v in accuracy_scores], padding=3)
    plt.bar_label(bars2, labels=[f"{v:.4f}" for v in mean_probs], padding=3)

    plt.xticks(x, models)
    plt.ylabel("Score")
    plt.title("Model Accuracy and Mean Predicted Probability")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()


    # Experimentation
    # rf_results, lr_results, dt_results = hyperparameter_experiment(df_processed)

    # rf_df = pd.DataFrame(rf_results, columns=["value", "accuracy"])
    # lr_df = pd.DataFrame(lr_results, columns=["value", "accuracy"])
    # dt_df = pd.DataFrame(dt_results, columns=["value", "accuracy"])

    # plt.figure(figsize=(8,5))
    # plt.plot(rf_df["value"], rf_df["accuracy"], marker="o")
    # plt.xlabel("Number of Trees")
    # plt.ylabel("Accuracy")
    # plt.title("Random Forest Accuracy by Number of Trees")
    # plt.ylim(0, 1)
    # plt.show()

    # plt.figure(figsize=(8,5))
    # plt.plot(lr_df["value"], lr_df["accuracy"], marker="o")
    # plt.xlabel("Max Iterations")
    # plt.ylabel("Accuracy")
    # plt.title("Logistic Regression Accuracy by Max Iterations")
    # plt.ylim(0, 1)
    # plt.show()

    # plt.figure(figsize=(8,5))
    # plt.plot(dt_df["value"], dt_df["accuracy"], marker="o")
    # plt.xlabel("Max Depth")
    # plt.ylabel("Accuracy")
    # plt.title("Decision Tree Accuracy by Max Depth")
    # plt.ylim(0, 1)
    # plt.show()


    

    

if __name__ == "__main__":
    main(debug=False)