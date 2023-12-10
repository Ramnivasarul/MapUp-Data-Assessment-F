import pandas as pd
from datetime import time

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    data = pd.read_csv(r"C:\Users\ramni\Downloads\MapUp-Data-Assessment-F\datasets\dataset-3.csv")

    # Create a dictionary to store distances between locations
    distances = {}

    # Iterate through the dataset to populate the distances dictionary
    for index, row in data.iterrows():
        location1 = row['id_start']
        location2 = row['id_end']
        distance = row['distance']

        # Populate distances for both directions (A to B and B to A)
        distances[(location1, location2)] = distance
        distances[(location2, location1)] = distance

    # Create a set of unique locations
    locations = set(data['Location1'].unique()).union(set(data['Location2'].unique()))

    # Initialize the distance matrix DataFrame
    distance_matrix = pd.DataFrame(index=locations, columns=locations)

    # Calculate cumulative distances along known routes
    for loc1 in locations:
        for loc2 in locations:
            if loc1 == loc2:
                distance_matrix.loc[loc1, loc2] = 0  # Diagonal values set to 0
            elif (loc1, loc2) in distances:
                distance_matrix.loc[loc1, loc2] = distances[(loc1, loc2)]
            else:
                # Find intermediate locations to calculate cumulative distances
                intermediate_locations = [k for k in locations if (loc1, k) in distances and (k, loc2) in distances]
                if intermediate_locations:
                    cumulative_distance = distances[(loc1, intermediate_locations[0])] + distances[(intermediate_locations[0], loc2)]
                    distance_matrix.loc[loc1, loc2] = cumulative_distance

    # Ensure symmetry in the distance matrix
    distance_matrix = distance_matrix.fillna(0)  # Fill NaN values with 0
    df = distance_matrix.astype(float)  # Convert values to float

    
    return df


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    upper_triangular = distance_matrix.where(~distance_matrix.isna(), other=distance_matrix.T)
    upper_triangular = upper_triangular.where(pd.DataFrame(np.tri(*distance_matrix.shape), index=distance_matrix.index, columns=distance_matrix.columns).astype(bool))

    # Create a DataFrame to store unrolled distance data
    unrolled_distances = pd.DataFrame(columns=['id_start', 'id_end', 'distance'])

    # Iterate through the upper triangular matrix to extract distances
    for i, row in upper_triangular.iterrows():
        for j, distance in row.items():
            if i != j:  # Exclude same id_start and id_end combinations
                unrolled_distances = unrolled_distances.append({'id_start': i, 'id_end': j, 'distance': distance}, ignore_index=True)
    df=unrolled_distances
    return df

    


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    # Filter the DataFrame by the reference value in id_start column
    reference_data = dataframe[dataframe['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    avg_distance = reference_data['distance'].mean()

    # Calculate the lower and upper bounds within 10% of the average distance
    lower_bound = avg_distance * 0.9
    upper_bound = avg_distance * 1.1

    # Filter the DataFrame for id_start values within the threshold range
    filtered_ids = dataframe[(dataframe['id_start'] != reference_value) & 
                             (dataframe['distance'] >= lower_bound) &
                             (dataframe['distance'] <= upper_bound)]

    # Get unique id_start values within the threshold range and sort them
    df = sorted(filtered_ids['id_start'].unique())

    return df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    df = dataframe.copy()

    # Define rate coefficients for different vehicle types
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type based on the distance
    for vehicle, coefficient in rate_coefficients.items():
        df[vehicle] = df['distance'] * coefficient

    return df


    


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
     # Copy the input DataFrame to avoid modifying the original DataFrame
    df = dataframe.copy()

    # Define time ranges for weekdays and weekends
    weekday_discounts = {
        (time(0, 0, 0), time(10, 0, 0)): 0.8,
        (time(10, 0, 0), time(18, 0, 0)): 1.2,
        (time(18, 0, 0), time(23, 59, 59)): 0.8
    }
    weekend_discount = 0.7

    # Create empty lists to store generated data
    rows = []
    
    # Iterate through each unique (id_start, id_end) pair
    for start_id, end_id in dataframe[['id_start', 'id_end']].drop_duplicates().values:
        for day in range(7):  # 7 days in a week
            for start_time, end_time in weekday_discounts.keys():
                start_datetime = pd.Timestamp('2023-01-02') + pd.Timedelta(days=day) + pd.Timedelta(hours=start_time.hour, minutes=start_time.minute, seconds=start_time.second)
                end_datetime = pd.Timestamp('2023-01-02') + pd.Timedelta(days=day) + pd.Timedelta(hours=end_time.hour, minutes=end_time.minute, seconds=end_time.second)

                # Apply weekday or weekend discounts based on day and time ranges
                if day < 5:  # Weekdays (Monday - Friday)
                    discount_factor = weekday_discounts[(start_time, end_time)]
                else:  # Weekends (Saturday and Sunday)
                    discount_factor = weekend_discount

                # Append the data to the list
                rows.append({
                    'id_start': start_id,
                    'id_end': end_id,
                    'start_day': start_datetime.strftime('%A'),
                    'start_time': start_datetime.time(),
                    'end_day': end_datetime.strftime('%A'),
                    'end_time': end_datetime.time(),
                    'discount_factor': discount_factor
                })

    # Convert the list of dictionaries to a DataFrame
    time_based_toll_rates = pd.DataFrame(rows)

    # Merge the generated DataFrame with the original DataFrame based on (id_start, id_end)
    df = pd.merge(df, time_based_toll_rates, on=['id_start', 'id_end'])

    # Calculate vehicle columns with discount factors
    vehicle_columns = ['moto', 'car', 'rv', 'bus', 'truck']
    for vehicle in vehicle_columns:
        df[vehicle] *= df['discount_factor']

    # Drop the temporary 'discount_factor' column
    df.drop(columns='discount_factor', inplace=True)

    return df

    
df = pd.read_csv(r"C:\Users\ramni\Downloads\MapUp-Data-Assessment-F\datasets\dataset-3.csv")
calculate_distance_matrix(df)
