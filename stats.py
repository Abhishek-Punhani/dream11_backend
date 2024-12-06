import polars as pl
import pandas as pd

def generate_fantasy_points_data(filtered_df: pl.DataFrame, player_id: str) -> pd.DataFrame:
    """
    Generates Fantasy Points metrics data for a specified player based on the provided filtered dataset.

    Parameters:
    - filtered_df (pl.DataFrame): A Polars DataFrame filtered to the specific player_id.
    - player_id (str): The unique identifier of the player.

    Returns:
    - pd.DataFrame: A DataFrame containing 'start_date', 'Floor', 'Ceiling',
      'Batting_First_FP', and 'Chasing_FP'.
    """
    
    def determine_batting_order(df: pl.DataFrame) -> pl.DataFrame:
        """
        Determine if the team batted first or chased based on 'win_by_run' and 'win_by_wickets'.
        Adds a new column 'batting_order' with values 'Batting First', 'Chasing', or 'Unknown'.
        """
        return df.with_columns(
            pl.when(pl.col('win_by_run') > 0)
            .then(pl.lit('Batting First'))
            .when(pl.col('win_by_wickets') > 0)
            .then(pl.lit('Chasing'))
            .otherwise(pl.lit('Unknown'))
            .alias('batting_order')
        )
    
    def calculate_metrics(df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Batting First Fantasy Points, Chasing Fantasy Points, Rolling Average,
        Rolling Standard Deviation, Floor, and Ceiling using window functions.
        """
        window_size = 3
    
        # Calculate rolling average and standard deviation
        df = df.with_columns([
            pl.col("Dream11_Points").rolling_mean(window_size).alias("Rolling_Avg"),
            pl.col("Dream11_Points").rolling_std(window_size).alias("Rolling_STD")
        ])
    
        # Calculate Floor and Ceiling
        df = df.with_columns([
            (pl.col("Rolling_Avg") - 2 * pl.col("Rolling_STD")).alias("Floor"),
            (pl.col("Rolling_Avg") + 2 * pl.col("Rolling_STD")).alias("Ceiling")
        ])
    
        # Ensure Floor is not negative
        df = df.with_columns(
            pl.when(pl.col("Floor") < 0)
            .then(pl.lit(0))
            .otherwise(pl.col("Floor"))
            .alias("Floor")
        )
    
        # Create conditional columns for Batting First and Chasing Fantasy Points
        df = df.with_columns([
            pl.when(pl.col("batting_order") == "Batting First")
            .then(pl.col("Dream11_Points"))
            .otherwise(pl.lit(0))
            .alias("Batting_First_Dream11_Points"),
            
            pl.when(pl.col("batting_order") == "Chasing")
            .then(pl.col("Dream11_Points"))
            .otherwise(pl.lit(0))
            .alias("Chasing_Dream11_Points")
        ])
    
        # Calculate rolling sums for Batting First Fantasy Points
        df = df.with_columns([
            pl.col("Batting_First_Dream11_Points").rolling_sum(window_size).alias("Batting_First_FP"),
            pl.col("Chasing_Dream11_Points").rolling_sum(window_size).alias("Chasing_FP")
        ])
    
        return df
    
    # Verify that the filtered_df contains only the specified player_id
    if not all(filtered_df['player_id'] == player_id):
        print(f"Warning: The filtered DataFrame contains data for multiple player_ids.")
        print(f"Expected player_id: {player_id}")
        print(f"Found player_ids: {filtered_df['player_id'].unique().to_list()}")
    
    # Step 1: Sort by start_date
    filtered_df = filtered_df.sort('start_date')
    
    # Step 2: Determine batting order
    filtered_df = determine_batting_order(filtered_df)
    
    # Step 3: Calculate metrics
    filtered_df = calculate_metrics(filtered_df)
    
    # Step 4: Select relevant columns
    data_points = filtered_df.select([
        pl.col("start_date"),
        pl.col("Floor"),
        pl.col("Ceiling"),
        pl.col("Batting_First_FP"),
        pl.col("Chasing_FP")
    ])
    
    # Step 5: Convert to Pandas DataFrame for easier handling
    data_points_pd = data_points.to_pandas()
    
    return data_points_pd

# Example Usage
def main():
    # Define the dataset path
    dataset_path = r'/home/manav/dev_ws/src/dream11_backend/filtered_dataset.csv'  # Update this path as needed

    # Step 1: Get player input
    player_name = input("Enter the name of the Player: ").strip()
    player_id = input("Enter the player_id: ").strip()  # Prompt for player_id

    # Step 2: Load the dataset
    try:
        df = pl.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: The file '{dataset_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    # Step 3: Parse 'start_date' column to datetime
    if 'start_date' in df.columns:
        try:
            df = df.with_columns(
                pl.col("start_date").str.strptime(pl.Date, "%Y-%m-%d").alias("start_date")
            )
        except Exception as e:
            print(f"Error parsing 'start_date': {e}")
            return
    else:
        print("Error: 'start_date' column not found in the dataset.")
        return

    # Step 4: Filter the DataFrame based on player_id (case-insensitive)
    filtered_df = df.filter(pl.col('player_id') == player_id)

    if filtered_df.is_empty():
        print(f"No data found for the specified Player ID '{player_id}'. Please check the ID and try again.")
        return

    # Step 5: Generate Fantasy Points metrics data
    data_points = generate_fantasy_points_data(filtered_df, player_id)

    # Step 6: Check if data_points is not empty
    if not data_points.empty:
        # Display the data points
        print("\nFantasy Points Metrics Data:")
        print(data_points.to_string(index=False))
    else:
        print("No data to display.")

if __name__ == "__main__":
    main()
