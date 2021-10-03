import pandas as pd



def carre(x):
    return x*x


def cube(x):
    return x*x*x



df = pd.DataFrame([
    {"function" : carre, "name_function" : "carre"},
    {"function" : cube, "name_function" : "cube"},
    ])


print(df.head())

f = df.loc["carre", "function"]
print(f)
print(f(3))