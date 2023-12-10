import pandas as pd


def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here
    df = df.pivot(index='id_1', columns='id_2', values='car')

    # Replace diagonal values with 0
    #df.values[[range(len(df))]*2] = 0
    df.fillna(0,inplace= True)
    print(df)
    
    return df


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here
    conditions = [
        (df['car'] <= 15),
        (df['car'] > 15) & (df['car'] <= 25),
        (df['car'] > 25)
    ]
    choices = ['low', 'medium', 'high']
    df['car_type'] = pd.Series(pd.cut(df['car'], bins=[0, 15, 25, float('inf')], labels=choices))

    # Calculate the count of occurrences for each 'car_type' category and return as a dictionary
    type_counts = df['car_type'].value_counts().to_dict()
    print(dict(sorted(type_counts.items())))
    

    return dict(sorted(type_counts.items()))


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here
    # Calculate the mean value of the 'bus' column
    mean_bus = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean value
    bus_indexes = df[df['bus'] > (2 * mean_bus)].index.tolist()
    
    # Sort the indices in ascending order
    bus_indexes.sort()
    print(list(bus_indexes))
    return list(bus_indexes)


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where average 'truck' values are greater than 7
    filtered_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sort the list of routes in ascending order
    filtered_routes.sort()
    print(filtered_routes)
    return list(filtered_routes)


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here
    # Apply the specified logic to modify values in the DataFrame
    modified_df = matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)
    
    # Round the values to 1 decimal place
    matrix = modified_df.round(1)
    print(matrix)

    return matrix


def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    return pd.Series()

df = pd.read_csv(r"C:\Users\ramni\Downloads\MapUp-Data-Assessment-F\datasets\dataset-1.csv")

matrix = generate_car_matrix(df)
#get_type_count(df)
#get_bus_indexes(df)
#filter_routes(df)
multiply_matrix(matrix)