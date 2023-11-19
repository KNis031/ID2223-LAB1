import os
import modal

LOCAL = False

if LOCAL == False:
    stub = modal.Stub("wine_daily")  # The app/stub
    image = modal.Image.debian_slim().pip_install(["hopsworks"])  # the virtual env/image

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("my-custom-secret"))
    def f():
       g()


# def generate_flower(name, sepal_len_max, sepal_len_min, sepal_width_max, sepal_width_min,
#                     petal_len_max, petal_len_min, petal_width_max, petal_width_min):
#     """
#     Returns a single iris flower as a single row in a DataFrame
#     """
#     import pandas as pd
#     import random
#
#     df = pd.DataFrame({ "sepal_length": [random.uniform(sepal_len_max, sepal_len_min)],
#                        "sepal_width": [random.uniform(sepal_width_max, sepal_width_min)],
#                        "petal_length": [random.uniform(petal_len_max, petal_len_min)],
#                        "petal_width": [random.uniform(petal_width_max, petal_width_min)]
#                       })
#     df['variety'] = name
#     return df
#
#
# def get_random_iris_flower():
#     """
#     Returns a DataFrame containing one random iris flower
#     """
#     import pandas as pd
#     import random
#
#     virginica_df = generate_flower("Virginica", 8, 5.5, 3.8, 2.2, 7, 4.5, 2.5, 1.4)
#     versicolor_df = generate_flower("Versicolor", 7.5, 4.5, 3.5, 2.1, 3.1, 5.5, 1.8, 1.0)
#     setosa_df =  generate_flower("Setosa", 6, 4.5, 4.5, 2.3, 1.2, 2, 0.7, 0.3)
#
#     # randomly pick one of these 3 and write it to the featurestore
#     pick_random = random.uniform(0,3)
#     if pick_random >= 2:
#         iris_df = virginica_df
#         print("Virginica added")
#     elif pick_random >= 1:
#         iris_df = versicolor_df
#         print("Versicolor added")
#     else:
#         iris_df = setosa_df
#         print("Setosa added")
#
#     return iris_df
#
# def g():
#     import hopsworks
#     import pandas as pd
#
#     project = hopsworks.login()
#     fs = project.get_feature_store()
#
#     iris_df = get_random_iris_flower()
#
#     iris_fg = fs.get_feature_group(name="iris",version=1)
#     iris_fg.insert(iris_df)


def generate_wine(max_df, min_df, type):
    import pandas as pd
    import random
    dfd = {'type': type}
    for feature,_ in max_df.items():
        dfd[feature] = [random.uniform(min_df[feature], max_df[feature])]

    new_wine_df = pd.DataFrame(dfd)
    new_wine_df['quality'] = new_wine_df['quality'].astype(int)
    return new_wine_df


def get_random_wine_quality():
    """
    Returns a DataFrame containing one random wine quality
    """
    import pandas as pd
    import random
    import hopsworks

    project = hopsworks.login()
    fs = project.get_feature_store()

    fg = fs.get_feature_group(name="wine_quality", version=2)

    stats = fg.get_statistics()
    st_dict = stats.content
    st_cols = st_dict['columns']
    # print(st_cols[0])
    min_quality = st_cols[0]['minimum']
    max_quality = st_cols[0]['maximum']

    random_quality = random.randint(min_quality, max_quality)

    df = fg.read()
    df = df[df['quality'] == random_quality]

    if (df['type'].nunique() > 1):  # if more than a single type
        t_counts = df['type'].value_counts()
        p_red = t_counts['red']/(t_counts['red']+t_counts['white'])
        pick_random = random.uniform(0, 1)
        random_type = 'red' if (pick_random < p_red) else 'white'
    else:
        random_type = df['type'].iloc[0]

    df = df[df['type'] == random_type]

    max_df = df.max(numeric_only=True, axis=0)
    min_df = df.min(numeric_only=True, axis=0)

    new_wine_df = generate_wine(max_df, min_df, random_type)
    print(f"Wine of quality {random_quality} added!")

    return new_wine_df

def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    new_wine_df = get_random_wine_quality()

    fg = fs.get_feature_group(name="wine_quality", version=2)
    fg.insert(new_wine_df)


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("wine_daily")
        with stub.run():
            f()
