from six import StringIO
from pydot import graph_from_dot_data
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from scipy import stats
from utils import read_data


def dec_tree(x, y):
    dt = DecisionTreeClassifier(criterion="gini", min_impurity_decrease=0.1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
    dt.fit(X_train, y_train)
    dot_data = StringIO()
    export_graphviz(dt, out_file=dot_data, feature_names=["LONGITUD", "LATITUDE"], class_names=["BARO", "TROP"])
    (graph, ) = graph_from_dot_data(dot_data.getvalue())
    graph.write_png("data/barotrop_tree.png")
    print(f"test_results = {dt.score(X_test, y_test)}")
    dt = DecisionTreeClassifier(criterion="gini", min_impurity_decrease=0.1)
    print(cross_val_score(dt, x, y).mean())


def ttest(a, b):
    t, p = stats.ttest_ind(a, b)
    print("t = ", t)
    print("p = ", p)


if __name__ == '__main__':
    data = read_data()
    print("t-stat for LONGITUD")
    ttest(a=data.loc[data["CLASS"] == "BARO", ["LONGITUD"]],
          b=data.loc[data["CLASS"] != "BARO", ["LONGITUD"]])
    print("t-stat for LATITUDE")
    ttest(a=data.loc[data["CLASS"] == "BARO", ["LATITUDE"]],
          b=data.loc[data["CLASS"] != "BARO", ["LATITUDE"]])
    dec_tree(data.iloc[:, :-1].to_numpy(), data.iloc[:, -1].to_numpy())
